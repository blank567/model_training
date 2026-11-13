"""统一只读后训练评测；冻结权重与测试身份，逐批保存结果，可安全接续。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import canonical_hash, read_json, sha256_file, write_json, write_jsonl
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_lab.tokenization.bpe import load_tokenizer
from llm_lora.data import AuxiliaryCorpus
from llm_reading_data.verifier import answer_f1, normalize_answer, reward
from llm_rl.generation import prefill
from llm_sft.data import SFTCorpus
import evaluate_sft_models as legacy
import train_sft as locking


def state_digest(state):
    """只对模型张量求摘要；优化器/路径差异不会冒充不同模型。"""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def cached_greedy(model, prefixes, max_new_tokens, device='cpu', forbidden=()):
    """[B,可变长提示] -> 只含新Token；各题独立EOS，右PAD不会被当成输入。"""
    if not prefixes or any(not p or len(p) + max_new_tokens > model.config.max_seq_len for p in prefixes):
        raise ValueError('Generation must preserve the entire nonempty prompt')
    model.eval(); state = prefill(model, prefixes, device)
    answers = [[] for _ in prefixes]
    active = torch.ones(len(prefixes), dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        selected = active.nonzero().flatten()
        scores = state.logits.index_select(0, selected).clone()
        if forbidden: scores[:, list(forbidden)] = -torch.inf
        if not torch.isfinite(scores).any(dim=1).all(): raise FloatingPointError('Invalid decode logits')
        chosen = scores.argmax(dim=-1)
        tokens = torch.full((len(prefixes),), 2, dtype=torch.long, device=device)
        tokens[selected] = chosen; was_active = active.clone()
        for index, token in zip(selected.tolist(), chosen.tolist()):
            answers[index].append(token)
            if token == 2: active[index] = False
        if not active.any() or step + 1 == max_new_tokens: break
        state.advance(tokens, was_active)
    return [{'generated_ids': ids, 'stop_reason': 'eos' if ids and ids[-1] == 2 else 'max_new_tokens'} for ids in answers]


def reading_metrics(row, generated, tokenizer):
    ids = generated['generated_ids']; visible = ids[:-1] if ids and ids[-1] == 2 else ids
    text = tokenizer.decode(visible, skip_special_tokens=False)
    em = reward(text, row['answers'], row['language'])
    f1 = answer_f1(text, row['answers'], row['language'])
    value = normalize_answer(text, row['language']) or ''
    context = normalize_answer(row['context'], row['language']) or ''
    special = any(token in {0, 1, 2} for token in visible)
    extractive = bool(value) and value in context
    if em: error = 'correct'
    elif not text.strip(): error = 'empty'
    elif special: error = 'special_token'
    elif generated['stop_reason'] != 'eos': error = 'length_limit'
    elif f1 > 0: error = 'partial_overlap'
    elif extractive: error = 'wrong_material_span'
    else: error = 'not_a_material_span'
    grams = [tuple(visible[i:i+3]) for i in range(max(0, len(visible)-2))]
    return {'id': row['id'], 'language': row['language'], 'task': row.get('task', 'reading_comprehension'),
            'source': row.get('source', 'unknown'), 'group_id': row.get('group_id', row['id']),
            'question': row['question'], 'context': row['context'], 'prompt': row['prompt'],
            'references': row['answers'], 'answer': text, **generated,
            'exact_match': em, 'f1': f1, 'extractive': extractive, 'empty': not text.strip(),
            'special_token': special, 'error_category': error,
            'prompt_tokens': row['prompt_token_count'], 'gold_answer_tokens': row['answer_token_count'],
            'generated_tokens': len(ids), 'visible_tokens': len(visible),
            'repeated_3gram_fraction': 1-len(set(grams))/len(grams) if grams else 0.}


def summarize_reading(rows):
    def group(values):
        return {'examples': len(values), 'correct': sum(r['exact_match'] for r in values),
                'exact_match': np.mean([r['exact_match'] for r in values]).item(),
                'f1': np.mean([r['f1'] for r in values]).item(),
                'eos_rate': np.mean([r['stop_reason'] == 'eos' for r in values]).item(),
                'empty_rate': np.mean([r['empty'] for r in values]).item(),
                'extractive_rate': np.mean([r['extractive'] for r in values]).item(),
                'special_token_rate': np.mean([r['special_token'] for r in values]).item(),
                'mean_generated_tokens': np.mean([r['generated_tokens'] for r in values]).item(),
                'mean_repeated_3gram_fraction': np.mean([r['repeated_3gram_fraction'] for r in values]).item(),
                'errors': dict(Counter(r['error_category'] for r in values))}
    out = group(rows)
    out['by_language'] = {key: group([r for r in rows if r['language'] == key]) for key in sorted({r['language'] for r in rows})}
    out['by_source'] = {key: group([r for r in rows if r['source'] == key]) for key in sorted({r['source'] for r in rows})}
    out['macro_exact_match'] = float(np.mean([v['exact_match'] for v in out['by_language'].values()]))
    out['macro_f1'] = float(np.mean([v['f1'] for v in out['by_language'].values()]))
    out['by_prompt_length'] = {}
    for key, lower, upper in [('0_255', 0, 256), ('256_511', 256, 512), ('512_767', 512, 768), ('768_plus', 768, 10**9)]:
        values = [r for r in rows if lower <= r['prompt_tokens'] < upper]
        if values: out['by_prompt_length'][key] = group(values)
    return out


def cluster_interval(reference, contender, metric='exact_match', samples=2000, seed=20261104):
    """相同题配对；同文章题作为整群重采样，按语言保留macro定义。"""
    right = {r['id']: r for r in contender}
    if len(right) != len(contender) or set(right) != {r['id'] for r in reference}: raise ValueError('Paired IDs differ')
    rng = np.random.default_rng(seed); distributions = []; diagnostics = {}
    observed = []
    for language in sorted({r['language'] for r in reference}):
        groups = defaultdict(lambda: [0., 0])
        values = [r for r in reference if r['language'] == language]
        for r in values:
            other = right[r['id']]
            if (other['language'], other.get('group_id')) != (r['language'], r.get('group_id')): raise ValueError('Pair metadata differs')
            key = r.get('group_id', r['id'])
            groups[key][0] += other[metric] - r[metric]; groups[key][1] += 1
        array = np.asarray(list(groups.values()), dtype=float)
        draws = rng.integers(0, len(array), size=(samples, len(array)))
        distributions.append(array[draws, 0].sum(1) / array[draws, 1].sum(1))
        observed.append(array[:, 0].sum() / array[:, 1].sum())
        diagnostics[language] = {'examples': len(values), 'clusters': len(groups)}
    distribution = np.mean(distributions, axis=0)
    return {'metric': metric, 'delta_contender_minus_reference': float(np.mean(observed)),
            'percentile_95': np.quantile(distribution, [.025, .975]).tolist(), 'replicates': samples,
            'method': 'paired language-stratified passage-cluster percentile bootstrap', 'groups': diagnostics,
            'note': 'Sampling uncertainty conditional on these trained weights; no seed or pretraining variance'}


def exact_mcnemar(reference, contender):
    right = {r['id']: r for r in contender}
    gained = sum(r['exact_match'] == 0 and right[r['id']]['exact_match'] == 1 for r in reference)
    lost = sum(r['exact_match'] == 1 and right[r['id']]['exact_match'] == 0 for r in reference)
    n = gained + lost; k = min(gained, lost)
    if n:
        logs = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)]
        pivot = max(logs); p = min(1., 2*math.exp(pivot)*sum(math.exp(x-pivot) for x in logs))
    else: p = 1.
    return {'gained_correct': gained, 'lost_correct': lost, 'discordant': n, 'two_sided_exact_p': p,
            'limitation': 'Question-level McNemar assumes independent question pairs; cluster bootstrap is preferred for shared passages'}


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); out = [None]*len(values); maximum = 0.
    for position, index in enumerate(order):
        maximum = max(maximum, min(1., values[index]*(len(values)-position))); out[index] = maximum
    return out


def get_corpus(spec):
    if sha256_file(spec['manifest']) != spec['manifest_sha256']: raise ValueError('Frozen test manifest changed')
    if spec['corpus_kind'] == 'auxiliary': corpus = AuxiliaryCorpus(spec['manifest'], 'test', spec['manifest_sha256'])
    else: corpus = SFTCorpus(spec['manifest'], 'test', allow_test=True)
    # 阅读数据没有历史指令任务标签；补充的仅为内存中的诊断标签，不改文本、Token或Mask。
    for row in corpus.rows:
        row.setdefault('task', 'reading_comprehension')
    if len(corpus) != spec['expected_examples']: raise ValueError('Incomplete test corpus')
    if spec.get('test_sha256') and corpus.identity['split_sha256'] != spec['test_sha256']: raise ValueError('Wrong final test split')
    if any(r['prompt_token_count'] + spec['max_new_tokens'] > 1024 for r in corpus.rows): raise ValueError('Prompt would be truncated')
    return corpus


def checkpoint_signature(item, tokenizer_sha):
    if sha256_file(item['checkpoint']) != item['checkpoint_sha256']: raise ValueError('Checkpoint changed after plan freeze')
    payload = torch.load(item['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    token_sha = payload.get('tokenizer', {}).get('sha256') or payload['identity'].get('tokenizer_sha256')
    if token_sha != tokenizer_sha: raise ValueError('Model and test tokenizer differ')
    if item['kind'] == 'lora':
        signature = canonical_hash({'base': payload['identity']['base_checkpoint_sha256'], 'adapter': state_digest(payload['adapter_state'])})
    else:
        signature = state_digest(payload['model'])
    metadata = {'tensor_sha256': signature, 'checkpoint_step': payload.get('optimizer_step', payload.get('step')),
                'checkpoint_kind': payload.get('checkpoint_kind'), 'tokenizer_sha256': token_sha}
    del payload
    return signature, metadata


def load_model(item, manifest):
    if item['kind'] == 'lora':
        from llm_lora.inference import load_model as load_lora
        model, tokenizer, metadata = load_lora(item['checkpoint'], 'cuda')
        return model, metadata
    payload = torch.load(item['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if payload.get('stage') != 'reading_rl':
        del payload
        from chat_sft import load_model as load_dense
        model, tokenizer, metadata = load_dense(item['checkpoint'], 'cuda',
            data_manifest=manifest if item['stage'] == 'pretrain' else None, threads=8)
        return model, metadata
    spec = ModelConfig(**payload['model_config']); spec.activation_checkpointing = False
    model = Decoder(spec); model.load_state_dict(payload['model'], strict=True)
    model.to('cuda').eval(); del payload
    return model, {'architecture': asdict(spec)}


@torch.inference_mode()
def numerical_probe(model, corpus, spec):
    """记录真实GPU缓存与原始前向的误差，不把BF16等价误写成逐位相等。"""
    order = sorted(range(len(corpus)), key=lambda i: corpus.rows[i]['prompt_token_count'])
    indices = [order[len(order)//8], order[len(order)//2], order[7*len(order)//8]]
    prefixes = [corpus.rows[i]['input_ids'][:corpus.rows[i]['prompt_token_count']] for i in indices]
    state = prefill(model, prefixes, 'cuda'); errors = []; matches = []
    for offset, prefix in enumerate(prefixes):
        with torch.autocast('cuda', dtype=torch.bfloat16): expected = model(torch.tensor([prefix], device='cuda')).logits[0, -1].float()
        error = ((state.logits[offset]-expected).norm()/expected.norm().clamp_min(1e-12)).item()
        errors.append(error); matches.append(bool(state.logits[offset].argmax() == expected.argmax()))
    del state
    if max(errors) > .04 or not all(math.isfinite(e) for e in errors): raise ValueError('Cached/native numerical probe failed')
    # 两种实现的首16个Token作为数值诊断，所有正式比较仍统一使用缓存协议。
    cached = cached_greedy(model, prefixes, 16, 'cuda', spec['forbidden_tokens'])
    native = legacy.greedy_batch(model, prefixes, 16, device='cuda', dtype='bf16', forbidden=tuple(spec['forbidden_tokens']))
    return {'examples': len(indices), 'ids': [corpus.rows[i]['id'] for i in indices],
            'prefill_relative_l2': errors, 'prefill_argmax_matches': matches,
            'first16_tokens_exactly_match_native': [a['generated_ids'] == b['generated_ids'] for a, b in zip(cached, native)],
            'bf16_not_bitwise_identical': True}


def run(plan_path):
    plan_path = Path(plan_path); plan = read_json(plan_path); root = Path(plan['project_root']); output = Path(plan['output_dir'])
    if not output.resolve().is_relative_to(root/'runs/assistant/inference'): raise ValueError('Wrong output scope')
    if canonical_hash({k:v for k,v in plan.items() if k != 'plan_identity'}) != plan['plan_identity']: raise ValueError('Plan changed')
    source_sha = sha256_file(__file__); receipt_path = output/'evaluator.json'
    if receipt_path.exists() and read_json(receipt_path)['sha256'] != source_sha: raise ValueError('Resume with the same evaluator source')
    write_json(receipt_path, {'path': str(Path(__file__).resolve()), 'sha256': source_sha})
    models = {m['key']: m for m in plan['models']}; started = time.monotonic(); results = {}
    torch.set_num_threads(8); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    with locking.training_lock(root):
        locking.gpu_idle()
        guard = BudgetGuard(plan['budget'], str(output.resolve()), root/'runs/budget_ledger.json')
        try:
            for suite in plan['suite_order']:
                spec = plan['suites'][suite]; corpus = get_corpus(spec); tokenizer = load_tokenizer(corpus.tokenizer_path)
                generation_batch = spec.get('decode_batch_size', plan['decode']['batch_size'])
                directory = output/suite; directory.mkdir(exist_ok=True); cache = {}; suite_results = []
                write_json(directory/'test_identity.json', corpus.identity)
                write_jsonl(directory/'test_metadata.jsonl', [{k:r.get(k) for k in ['id','language','task','source','group_id','question','context','prompt','system','response','answers','prompt_token_count','answer_token_count']} for r in corpus.rows])
                for key in spec['models']:
                    item = models[key]; dest = directory/key; dest.mkdir(exist_ok=True)
                    result_file = dest/'metrics.json'
                    signature, metadata = checkpoint_signature(item, corpus.identity['tokenizer_sha256'])
                    if result_file.exists():
                        result = read_json(result_file)
                        if result['checkpoint_sha256'] != item['checkpoint_sha256'] or result['tensor_sha256'] != signature: raise ValueError('Completed evaluation identity changed')
                        if result['test_identity'] != corpus.identity: raise ValueError('Completed evaluation belongs to another test split')
                        cache[signature] = dest; suite_results.append(result); continue
                    if signature in cache:
                        source = cache[signature]; original = read_json(source/'metrics.json')
                        result = {**original, 'key': key, 'checkpoint': item['checkpoint'], 'checkpoint_sha256': item['checkpoint_sha256'],
                                  'model': item, 'metadata': metadata, 'reused_identical_model_results': str(source),
                                  'elapsed_seconds': 0., 'actual_gpu_evaluation_repeated': False}
                        for name in ['test_rows.jsonl', 'generations.jsonl']:
                            (dest/name).write_bytes((source/name).read_bytes())
                        write_json(result_file, result); suite_results.append(result); continue
                    tick = time.monotonic(); write_json(output/'state.json', {'status':'loading','suite':suite,'model':key})
                    model, loaded = load_model(item, plan['suites']['reading_test_2000']['manifest'])
                    model.requires_grad_(False); before = state_digest(model.state_dict()); torch.cuda.reset_peak_memory_stats()
                    probe = numerical_probe(model, corpus, spec)
                    eval_cfg = {'train': {'device':'cuda','dtype':'bf16','eval_micro_batch_size':plan['loss']['micro_batch_size']}}
                    test, loss_rows = legacy.evaluate_loss(model, corpus, eval_cfg, guard)
                    write_jsonl(dest/'test_rows.jsonl', loss_rows)
                    predictions = []; part = dest/'generations.partial.jsonl'
                    if part.exists():
                        predictions = [json.loads(line) for line in part.read_text(encoding='utf-8').splitlines()]
                    seen = {r['id'] for r in predictions}
                    if len(seen) != len(predictions) or not seen <= {r['id'] for r in corpus.rows}: raise ValueError('Invalid partial evaluation')
                    order = sorted([i for i,r in enumerate(corpus.rows) if r['id'] not in seen], key=lambda i:corpus.rows[i]['prompt_token_count'])
                    with part.open('a', encoding='utf-8') as stream:
                        for start in range(0, len(order), generation_batch):
                            rows = [corpus.rows[i] for i in order[start:start+generation_batch]]
                            prefixes = [r['input_ids'][:r['prompt_token_count']] for r in rows]
                            values = cached_greedy(model, prefixes, spec['max_new_tokens'], 'cuda', spec['forbidden_tokens'])
                            for row, value in zip(rows, values):
                                if suite == 'reading_test_2000': prediction = reading_metrics(row, value, tokenizer)
                                else:
                                    from llm_lora.metrics import supported_constraints
                                    prediction = legacy.generation_metrics(row, value, tokenizer)
                                    prediction['constraint_checks'] = supported_constraints(row['system']+'\n'+row['prompt'], prediction['answer'])
                                    prediction['generated_tokens'] = len(value['generated_ids'])
                                predictions.append(prediction);stream.write(json.dumps(prediction, ensure_ascii=False)+'\n')
                            stream.flush()
                            if start % (generation_batch*5) == 0 or len(predictions) == len(corpus):
                                state = {'status':'generating','suite':suite,'model':key,'generated':len(predictions),'expected':len(corpus),'elapsed_seconds':time.monotonic()-started}
                                write_json(output/'state.json', state); print(json.dumps(state), flush=True)
                            if guard.should_stop(): raise TimeoutError('Evaluation budget/time reached; partial predictions retained')
                    if len(predictions) != len(corpus) or len({r['id'] for r in predictions}) != len(corpus): raise ValueError('Incomplete generation')
                    write_jsonl(dest/'generations.jsonl', predictions)
                    if state_digest(model.state_dict()) != before: raise ValueError('Read-only evaluation changed weights')
                    if suite == 'reading_test_2000': generation = summarize_reading(predictions)
                    else:
                        from llm_lora.metrics import summarize_constraints
                        generation = legacy.summarize_generations(predictions)
                        generation['decoding'] = 'greedy BF16 read-only KV; full prompt; no repetition penalty; forbid PAD/UNK only'
                        generation['supported_constraints'] = summarize_constraints(predictions)
                    result = {'key':key,'model':item,'checkpoint':item['checkpoint'],'checkpoint_sha256':item['checkpoint_sha256'],
                              **metadata,'loaded_metadata':loaded,'test':test,'generation':generation,'numerical_probe':probe,
                              'generated_examples':len(predictions),'test_identity':corpus.identity,'weights_unchanged':True,
                              'elapsed_seconds':time.monotonic()-tick,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                              'decode_batch_size':generation_batch,
                              'test_not_used_to_change_config':True,'actual_gpu_evaluation_repeated':True}
                    write_json(result_file, result); cache[signature] = dest; suite_results.append(result)
                    del model; gc.collect(); torch.cuda.empty_cache()
                results[suite] = suite_results; write_json(directory/'comparison.json', {'status':'completed','results':suite_results})
            write_json(output/'comparison.json', {'status':'completed','plan_identity':plan['plan_identity'],'results':results,
                'source_sha256':source_sha,'elapsed_seconds':time.monotonic()-started,'budget':guard.update(),
                'model_or_optimizer_updates':0,'no_checkpoints_deleted':True})
            write_json(output/'state.json', {'status':'completed','suite_models':{k:len(v) for k,v in results.items()}})
        except BaseException as error:
            write_json(output/'failure.json', {'type':type(error).__name__,'error':str(error)}); traceback.print_exc(); raise
        finally: guard.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
