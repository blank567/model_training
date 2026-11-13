"""冻结测试集上的统一SFT比较；推理、Loss和生成指标均不更新参数。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import copy
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch.nn import functional as F

from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import canonical_hash, read_json, sha256_file, write_json, write_jsonl
from llm_lab.config import ModelConfig
from llm_lab.engine.corpus import BinaryCorpus
from llm_lab.models import Decoder
from llm_lab.tokenization.bpe import load_tokenizer
from llm_sft.config import runtime_fingerprint
from llm_sft.data import SFTCorpus
from llm_sft.engine import autocast, evaluate_retention
import train_sft as launcher


def normalized(text):
    return re.sub(r'[\W_]+', '', text.casefold())


def overlap_units(text, language):
    return list(normalized(text)) if language == 'zh' else re.findall(r"\w+(?:['’-]\w+)*", text.casefold())


def rouge_l_f1(answer, reference, language):
    """中英文分别按规范化字符/单词计算LCS；仅是参考重叠指标。"""
    left, right = overlap_units(answer, language), overlap_units(reference, language)
    if not left or not right: return 0.
    previous = [0] * (len(right) + 1)
    for a in left:
        current = [0]
        for index, b in enumerate(right, 1):
            current.append(previous[index-1]+1 if a == b else max(previous[index], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(left) + len(right))


def generation_indices(rows, count, seed, context, max_new_tokens):
    """按语言/任务轮流抽取同一批题；抽样不依赖任何模型输出。"""
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        if row['prompt_token_count'] + max_new_tokens > context: continue
        buckets[(row['language'], row['task'])].append(index)
    for key in buckets:
        buckets[key].sort(key=lambda index: canonical_hash({'seed': seed, 'id': rows[index]['id']}))
    chosen = []
    while len(chosen) < min(count, sum(len(values) for values in buckets.values()) + len(chosen)):
        changed = False
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop(0)); changed = True
        if not changed: break
    return chosen


@torch.inference_mode()
def greedy_batch(model, prefixes, max_new_tokens, *, device='cpu', dtype='fp32', pad_id=0, eos_id=2, forbidden=(0,1)):
    """右补齐并逐样本取最后有效位置；不把PAD位置当成生成位置。"""
    if not prefixes or any(not values or len(values)+max_new_tokens > model.config.max_seq_len for values in prefixes):
        raise ValueError('Empty prefixes or generation exceeds context')
    histories = [list(values) for values in prefixes]; generated = [[] for _ in prefixes]
    active = list(range(len(prefixes))); reason = ['max_new_tokens'] * len(prefixes)
    model.eval()
    for _ in range(max_new_tokens):
        width = max(len(histories[index]) for index in active)
        values = torch.full((len(active), width), pad_id, dtype=torch.long, device=device)
        lengths = []
        for offset, index in enumerate(active):
            history = histories[index]; lengths.append(len(history)-1)
            values[offset, :len(history)] = torch.tensor(history, device=device)
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' and dtype == 'bf16' else contextlib.nullcontext()
        with precision: output = model(values).logits
        scores = output[torch.arange(len(active), device=device), torch.tensor(lengths, device=device)].float()
        scores[:, list(forbidden)] = -torch.inf
        if not torch.isfinite(scores).any(dim=1).all(): raise FloatingPointError('No finite decode score')
        identifiers = scores.argmax(dim=-1).tolist(); remaining = []
        for index, identifier in zip(active, identifiers):
            histories[index].append(identifier); generated[index].append(identifier)
            if identifier == eos_id: reason[index] = 'eos'
            else: remaining.append(index)
        active = remaining
        if not active: break
    return [{'generated_ids': values, 'stop_reason': stop} for values, stop in zip(generated, reason)]


@torch.inference_mode()
def evaluate_loss(model, corpus, config, guard=None):
    rows = []; batch_size = config['train']['eval_micro_batch_size']; device = config['train']['device']
    order = sorted(range(len(corpus.rows)), key=lambda index: len(corpus.rows[index]['input_ids']))
    model.eval()
    for offset in range(0, len(order), batch_size):
        indices = order[offset:offset+batch_size]; batch = corpus.collate(indices, device=device)
        with autocast(config): logits = model(batch['x']).logits
        losses = F.cross_entropy(logits.float().transpose(1,2), batch['y'], ignore_index=-100, reduction='none')
        mask = batch['y'] != -100
        correct = ((logits.argmax(dim=-1) == batch['y']) & mask).sum(dim=1).tolist()
        totals = losses.sum(dim=1).tolist(); tokens = mask.sum(dim=1).tolist()
        for index, loss_sum, token_count, hit in zip(indices, totals, tokens, correct):
            if not math.isfinite(loss_sum): raise FloatingPointError('Nonfinite test loss')
            row = corpus.rows[index]
            rows.append({'id':row['id'], 'language':row['language'], 'task':row['task'],
                         'loss_sum':loss_sum, 'tokens':token_count, 'correct_tokens':hit})
        if guard and offset % (batch_size*20) == 0 and guard.should_stop():
            raise TimeoutError('Evaluation budget reached')
    result = {}
    for language in ('en','zh'):
        values = [row for row in rows if row['language']==language]
        total = sum(row['tokens'] for row in values); loss = sum(row['loss_sum'] for row in values)/total
        result[language] = {'loss':loss, 'perplexity':math.exp(min(loss,80)), 'examples':len(values),
                            'supervised_tokens':total, 'teacher_forced_token_accuracy':sum(row['correct_tokens'] for row in values)/total}
    result['macro_loss'] = (result['en']['loss']+result['zh']['loss'])/2
    result['tasks'] = {}
    for task in sorted({row['task'] for row in rows}):
        values = [row for row in rows if row['task']==task]
        result['tasks'][task] = {'examples':len(values), 'loss':sum(row['loss_sum'] for row in values)/sum(row['tokens'] for row in values)}
    return result, rows


def generation_metrics(row, result, tokenizer):
    identifiers = result['generated_ids']; visible = identifiers[:-1] if identifiers and identifiers[-1]==2 else identifiers
    answer = tokenizer.decode(visible, skip_special_tokens=True)
    grams = [tuple(visible[index:index+3]) for index in range(max(0,len(visible)-2))]
    return {'id':row['id'], 'language':row['language'], 'task':row['task'], 'prompt':row['prompt'],
            'system':row['system'], 'reference':row['response'], 'answer':answer, **result,
            'normalized_exact_match':normalized(answer)==normalized(row['response']),
            'reference_overlap_rouge_l_f1':rouge_l_f1(answer,row['response'],row['language']),
            'repeated_3gram_fraction':1-len(set(grams))/len(grams) if grams else 0.,
            'empty':not answer.strip()}


def summarize_generations(rows):
    return {'examples':len(rows), 'eos_stop_fraction':sum(row['stop_reason']=='eos' for row in rows)/len(rows),
            'empty_fraction':sum(row['empty'] for row in rows)/len(rows),
            'mean_reference_overlap_rouge_l_f1':sum(row['reference_overlap_rouge_l_f1'] for row in rows)/len(rows),
            'mean_repeated_3gram_fraction':sum(row['repeated_3gram_fraction'] for row in rows)/len(rows),
            'normalized_exact_match_fraction':sum(row['normalized_exact_match'] for row in rows)/len(rows),
            'not_task_accuracy':True, 'decoding':'greedy; no repetition penalties; full prefix recompute; shared prompts'}


def select_model(results):
    """事前固定为中英文测试回答Loss均值最低；生成与保持单列，不造综合分。"""
    eligible = [row for row in results if row.get('candidate') and math.isfinite(row['test']['macro_loss'])]
    if not eligible: raise ValueError('No completed candidate with finite test loss')
    return min(eligible, key=lambda row:(row['test']['macro_loss'],row['key']))


def paired_interval(reference, contender, seed=42, samples=1000):
    """按语言配对重采样，估计两个模型macro回答Loss差的区间。"""
    right = {row['id']:row for row in contender}; rng = np.random.default_rng(seed); components = []
    if set(right) != {row['id'] for row in reference}: raise ValueError('Bootstrap samples differ')
    for language in ('en','zh'):
        left = [row for row in reference if row['language']==language]
        losses = np.array([row['loss_sum'] for row in left]); other = np.array([right[row['id']]['loss_sum'] for row in left])
        tokens = np.array([row['tokens'] for row in left])
        indices = rng.integers(0,len(left),size=(samples,len(left)))
        components.append((other[indices].sum(axis=1)-losses[indices].sum(axis=1))/tokens[indices].sum(axis=1))
    differences = (components[0]+components[1])/2
    return {'delta_is_other_minus_selected':True, 'percentile_95':np.quantile(differences,[.025,.975]).tolist(),
            'replicates':samples, 'note':'Descriptive paired resampling, not a multiple-comparison correction'}


def run(plan_path):
    plan = read_json(plan_path); root = Path(plan['project_root']).resolve(); output = Path(plan['output_dir'])
    if not output.resolve().is_relative_to(root/'runs/assistant/inference'): raise ValueError('Wrong assistant output scope')
    if output.exists(): raise FileExistsError('Refusing to overwrite evaluation')
    if runtime_fingerprint(root) != plan['runtime_sha256']: raise ValueError('Frozen training source changed')
    if sha256_file(plan['data_manifest']) != plan['data_manifest_sha256']: raise ValueError('Evaluation manifest changed')
    output.mkdir(parents=True,exist_ok=False); write_json(output/'plan.json',plan)
    config = {'train':{'device':'cuda','dtype':'bf16','eval_micro_batch_size':plan['loss_batch_size'],
                        'seed':42,'retention_eval_batches':2,'retention_eval_seq_len':512}}
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False
    corpus = SFTCorpus(plan['data_manifest'],'test',allow_test=True)
    write_json(output/'test_identity.json',corpus.identity)
    base_payload = torch.load(plan['base_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if sha256_file(plan['base_checkpoint']) != plan['base_checkpoint_sha256']: raise ValueError('Base changed')
    base_config = base_payload['config']; old_manifest = Path(base_config['data_manifest'])
    if not old_manifest.is_absolute(): old_manifest = root/old_manifest
    if sha256_file(old_manifest) != base_payload['identity']['manifest_sha256']: raise ValueError('Retention data changed')
    retention = BinaryCorpus(old_manifest,'validation',512)
    model_spec = base_config['model']; del base_payload
    token_path = Path(plan['data_manifest']).parent/corpus.manifest['tokenizer']['path']
    tokenizer = load_tokenizer(token_path)
    sample_indices = generation_indices(corpus.rows,plan['generation_examples'],42,model_spec['max_seq_len'],plan['max_new_tokens'])
    write_json(output/'generation_sample_ids.json',{'ids':[corpus.rows[index]['id'] for index in sample_indices],
        'test_content_is_selection_data_this_round':True,'auxiliary_test_not_read':True})
    results = []; row_losses = {}; started = time.monotonic()
    with launcher.training_lock(root):
        launcher.gpu_idle(); guard = BudgetGuard(plan['budget'],str(output.resolve()),root/'runs/budget_ledger.json')
        try:
            for number, candidate in enumerate(plan['candidates']):
                phase = {'status':'evaluating','candidate':candidate['key'],'index':number,'total':len(plan['candidates'])}
                write_json(output/'state.json',phase); print(json.dumps(phase),flush=True)
                if sha256_file(candidate['checkpoint']) != candidate['checkpoint_sha256']: raise ValueError('Candidate changed')
                payload = torch.load(candidate['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
                if payload['identity']['tokenizer_sha256'] != corpus.identity['tokenizer_sha256']: raise ValueError('Tokenizer changed')
                if candidate['candidate'] and (payload.get('stage')!='sft' or payload['identity']['data_identity']!=corpus.identity['data_identity']):
                    raise ValueError('Candidate belongs to different SFT data')
                spec = ModelConfig(**(payload['model_config'] if payload.get('stage')=='sft' else payload['config']['model']))
                spec.activation_checkpointing=False; model = Decoder(spec); model.load_state_dict(payload['model'],strict=True)
                step = payload.get('step'); del payload; model.to('cuda').eval()
                before = time.monotonic(); test, losses = evaluate_loss(model,corpus,config,guard)
                held_retention = evaluate_retention(model,retention,config)
                predictions = []
                for offset in range(0,len(sample_indices),plan['generation_batch_size']):
                    indices=sample_indices[offset:offset+plan['generation_batch_size']]
                    prefixes=[corpus.rows[index]['input_ids'][:corpus.rows[index]['prompt_token_count']] for index in indices]
                    generated=greedy_batch(model,prefixes,plan['max_new_tokens'],device='cuda',dtype='bf16')
                    predictions.extend(generation_metrics(corpus.rows[index],value,tokenizer) for index,value in zip(indices,generated))
                    if guard.should_stop(): raise TimeoutError('Evaluation budget reached')
                report={**candidate,'checkpoint_step':step,'test':test,'retention_validation':held_retention,
                        'generation':summarize_generations(predictions),'elapsed_seconds':time.monotonic()-before}
                folder=output/candidate['key']; folder.mkdir(); write_json(folder/'metrics.json',report)
                write_jsonl(folder/'test_rows.jsonl',losses); write_jsonl(folder/'generations.jsonl',predictions)
                results.append(report); row_losses[candidate['key']]=losses
                write_json(output/'partial_results.json',results)
                del model; gc.collect(); torch.cuda.empty_cache()
            selected=select_model(results)
            intervals={row['key']:paired_interval(row_losses[selected['key']],row_losses[row['key']])
                       for row in results if row['candidate'] and row['key']!=selected['key']}
            report={'status':'completed','selection_rule':plan['selection_rule'],'selected':selected,'results':results,
                    'paired_test_loss_intervals':intervals,'test_identity':corpus.identity,
                    'test_used_for_selection':True,'auxiliary_test_not_read':True,'training_started':False,
                    'elapsed_seconds':time.monotonic()-started,'budget':guard.update(),
                    'limits':['Test-based selection is not an untouched final test.',
                              'Teacher-forced token accuracy and ROUGE overlap are not task correctness.',
                              'Generated outputs use a finite stratified sample; retention uses fixed small validation windows.']}
            write_json(output/'comparison.json',report); write_json(output/'state.json',{'status':'completed','selected':selected['key']})
            return report
        except BaseException as error:
            write_json(output/'state.json',{'status':'failed','error_type':type(error).__name__});raise
        finally:guard.update()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().plan),ensure_ascii=False),flush=True)
