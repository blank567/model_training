"""按原题分组发布公开中英数学 SFT；不使用合成算术模板补足配额。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from importlib import metadata
import json
from pathlib import Path
import random
import re
import shutil

from llm_lab.common.io import canonical_hash, read_json, sha256_file, write_json, write_jsonl
from llm_math_data.verifier import numeric_value
from .rules import TOPICS, classify, reject_reason, final_answer, answer_key, numeric_literals, question_hash

PROTOCOL = {'prefix_template': 'System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n',
            'empty_system_prefix_template': 'User:\n{prompt}\n\nAssistant:\n',
            'encoding': 'prefix_and_response_separately', 'max_sequence_tokens': 1024,
            'labels': 'unshifted same-position loss mask; training adapter must shift exactly once',
            'eos_supervised': True}
IDENTITY_FIELDS = ('schema_version','kind','splits','tokenizer','tokenization','chinese_normalization')
WEIGHTS = {'algebra_equations':.20, 'geometry':.17, 'functions_trigonometry':.12,
           'probability_statistics':.10, 'combinatorics':.10, 'number_theory':.10,
           'calculus':.08, 'linear_algebra':.07, 'arithmetic_word':.06}


def clean_text(text, converter=None):
    text = text.replace('\r\n','\n').strip()
    # 仅修复文字中的转义换行；不得破坏 LaTeX 的 \nu、\nabla 等命令。
    text = re.sub(r'\\n(?![A-Za-z])', '\n', text).replace('\\"','"')
    if converter is not None:
        for _ in range(8):
            newer = converter.convert(text)
            if newer == text: break
            text = newer
        else: raise ValueError('OpenCC did not reach a fixed point')
    return text


def prefix(prompt):
    return PROTOCOL['empty_system_prefix_template'].format(prompt=prompt)


def training_text(question,response,language):
    answer=final_answer(response)
    instruction=('请给出解题过程，并在最后一行写“答案：最终答案”。' if language=='zh' else
                 'Show your solution and put "Answer: final answer" on the last line.')
    ending=('答案：' if language=='zh' else 'Answer: ')+answer if answer else None
    return question+'\n'+instruction, response+('\n'+ending if ending and not response.endswith(ending) else '')


def allocate(total, capacities, weights=WEIGHTS, minimum=0):
    """有容量上限的确定性配额；稀缺学科缺口不由算术题填满。"""
    if total > sum(capacities.values()) or any(capacities[k] < minimum for k in weights):
        raise ValueError('Insufficient subject coverage or total candidate capacity')
    result = {k: minimum for k in weights}
    if sum(result.values()) > total: raise ValueError('Minimum quotas exceed requested total')
    while sum(result.values()) < total:
        eligible = [k for k in weights if result[k] < capacities[k]]
        key = max(eligible, key=lambda k: (total*weights[k]-result[k], weights[k], k))
        result[key] += 1
    return result


def exclusion_hashes(root):
    old = root/'artifacts/research/math_alignment_sources_20261101'
    values = set(read_json(old/'project_exclusions.json')['question_hashes'])
    origins = read_json(old/'sources.lock.json')['sources']
    files = {}
    # 仅用既有官方测试题的题面构建去污染索引，不计算任何模型测试指标。
    for spec in origins:
        if spec['role'] != 'external_test_only': continue
        path = old/spec['path']
        if sha256_file(path) != spec['sha256']: raise ValueError('Frozen external source changed')
        files[spec['path']] = spec['sha256']
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            question = row.get('question') or row.get('problem')
            if isinstance(question,str): values.add(question_hash(question))
    return values, files


def candidates(root, cache, token_file, work):
    import pyarrow.parquet as pq
    from opencc import OpenCC
    from tokenizers import Tokenizer
    if metadata.version('OpenCC') != '1.1.9': raise ValueError('Use official OpenCC==1.1.9')
    converter, tokenizer = OpenCC('t2s'), Tokenizer.from_file(str(token_file))
    lock = read_json(cache/'sources.lock.json')
    for item in lock['sources']:
        if sha256_file(cache/item['local_path']) != item['sha256']: raise ValueError('Pinned source checksum changed')
    english = read_json(cache/'TIGER-Lab__MathInstruct/MathInstruct.json')
    chinese = pq.read_table(cache/'ALmonster__MathInstruct-Chinese/mathinstruct-zh.parquet').to_pylist()
    exclusions, test_sources = exclusion_hashes(root)
    rejected, seen, seen_zh, pool = Counter(), set(), set(), defaultdict(list)
    literal_alignment, checked_alignment = 0, 0
    for position, row in enumerate(chinese):
        if position % 20000 == 0: print(json.dumps({'phase':'curation','chinese_scanned':position,'candidate_pairs':sum(map(len,pool.values()))}),flush=True)
        try:
            idx = int(row['id'])-1
            if idx < 0: raise ValueError()
            original = english[idx]
            if [m['role'] for m in row['messages']] != ['user','assistant']: raise ValueError()
        except (ValueError,IndexError,KeyError): rejected['invalid_source_alignment']+=1;continue
        ep, ea = clean_text(original['instruction']), clean_text(original['output'])
        zp, za = clean_text(row['messages'][0]['content'],converter), clean_text(row['messages'][1]['content'],converter)
        source = original['source']
        reason = reject_reason(ep,ea,source,'en') or reject_reason(zp,za,source,'zh')
        if reason: rejected[reason]+=1;continue
        if re.search(r'\(A\).+\(B\)',ep,re.S) and not re.search(r'\(?A\s*[)）.、:：].+\(?B\s*[)）.、:：]',zp,re.S):
            rejected['lost_multiple_choice_options']+=1;continue
        a1,a2 = answer_key(final_answer(ea)),answer_key(final_answer(za))
        if a1 and a2 and a1 != a2:
            rejected['explicit_final_answer_disagreement']+=1;continue
        if len(numeric_literals(ep))>=2:
            checked_alignment += 1
            if len(numeric_literals(ep)&numeric_literals(zp))/len(numeric_literals(ep))<.8:
                rejected['prompt_numeral_alignment_uncertain']+=1;continue
            literal_alignment += 1
        # 过度转义的 LaTeX 不自动改写含义，留在原始储备中，不进入训练。
        if re.search(r'\\{2,}(?=[A-Za-z])',zp+za):
            rejected['overescaped_latex_translation']+=1;continue
        topic = classify(ep,source)
        if topic is None: rejected['unclassified_topic']+=1;continue
        group = question_hash(ep)
        if group in exclusions or question_hash(zp) in exclusions:
            rejected['known_project_or_external_test_question']+=1;continue
        if group in seen or question_hash(zp) in seen_zh:
            rejected['duplicate_original_or_translated_question']+=1;continue
        eq,ez=ep,zp;es,zs=ea,za
        ep,ea=training_text(ep,ea,'en');zp,za=training_text(zp,za,'zh')
        lengths, invalid = [], False
        for p,a in ((ep,ea),(zp,za)):
            pids = tokenizer.encode(prefix(p),add_special_tokens=False).ids
            aids = tokenizer.encode(a,add_special_tokens=False).ids+[2]
            if len(pids)+len(aids)>1024 or len(aids)<8 or any(i in {0,1,2} for i in pids+aids[:-1]):
                invalid=True;break
            lengths.append({'prompt':len(pids),'answer':len(aids),'total':len(pids)+len(aids)})
        if invalid: rejected['context_or_token_quality']+=1;continue
        seen.add(group);seen_zh.add(question_hash(ez))
        pool[topic].append({'group_id':group,'upstream_id':row['id'],'upstream_source':source,'topic':topic,
            'en':{'question':eq,'source_response':es,'prompt':ep,'response':ea,'final_answer':final_answer(es),'counts':lengths[0]},
            'zh':{'question':ez,'source_response':zs,'prompt':zp,'response':za,'final_answer':final_answer(zs),'counts':lengths[1]}})
    result = {'raw_english_rows':len(english),'raw_chinese_rows':len(chinese),'rejected_pairs':dict(rejected),
        'eligible_unique_pairs':{key:len(pool[key]) for key in TOPICS},'explicit_numeral_alignment_checks':checked_alignment,
        'passed_numeral_alignment_checks':literal_alignment,'source_lock_sha256':sha256_file(cache/'sources.lock.json'),
        'external_question_exclusion_sources':test_sources,'known_question_hashes':len(exclusions),
        'translation_alignment':'public id -> English one-based index; numerals/explicit final answers checked, not full semantic certification'}
    write_json(work/'curation_report.json',result)
    return pool,result,tokenizer,converter


def build(root,cache,token_file,output,work,train_records=50000,validation_records=2000,test_records=2000):
    if output.exists(): raise ValueError('Refusing to overwrite any previous publication')
    if any(n%2 for n in (train_records,validation_records,test_records)): raise ValueError('Parallel counts must be even')
    pool,report,tokenizer,converter=candidates(root,cache,token_file,work)
    rng=random.Random(42061101)
    for values in pool.values(): rng.shuffle(values)
    capacities={key:len(pool[key]) for key in TOPICS}
    # 先为开发和最终测试保留每个学科的原题组，再分配正式训练。
    val=allocate(validation_records//2,capacities,minimum=10)
    remain={k:capacities[k]-val[k] for k in TOPICS}
    test=allocate(test_records//2,remain,minimum=10)
    remain={k:remain[k]-test[k] for k in TOPICS}
    train=allocate(train_records//2,remain,minimum=100)
    if train['arithmetic_word']>train_records//2*.10: raise ValueError('Arithmetic word-problem share exceeds 10%')
    output.mkdir(parents=True)
    shutil.copyfile(token_file,output/'tokenizer.json')
    specs, statistics, selections = {},{},{}
    offsets=Counter()
    for split,quotas in [('validation',val),('test',test),('train',train)]:
        selected=[]
        for topic,n in quotas.items():
            selected.extend(pool[topic][offsets[topic]:offsets[topic]+n]);offsets[topic]+=n
        rng.shuffle(selected);selections[split]=selected
    selections['pilot']=selections['train'][:1000]
    for split,selected in selections.items():
        lang_tokens=Counter();lang_rows=Counter();topics=Counter();sources=Counter();answer_types=Counter();totals=[]
        def rows():
            for pair in selected:
                for language in ['zh','en']:
                    item=pair[language];prompt_ids=tokenizer.encode(prefix(item['prompt']),add_special_tokens=False).ids
                    answer_ids=tokenizer.encode(item['response'],add_special_tokens=False).ids+[2]
                    answer=item['final_answer'];kind='numeric_exact' if numeric_value(answer) is not None else 'choice_exact' if isinstance(answer,str) and re.fullmatch('[A-H]',answer.strip()) else None
                    record={'schema_version':1,'split':split,'id':'public_math_'+pair['group_id'][:24]+'_'+language,
                        'group_id':pair['group_id'],'language':language,'system':'','question':item['question'],
                        'prompt':item['prompt'],'response':item['response'],
                        'input_ids':prompt_ids+answer_ids,'loss_mask':[0]*len(prompt_ids)+[1]*len(answer_ids),
                        'prompt_token_count':len(prompt_ids),'answer_token_count':len(answer_ids),
                        'topic':pair['topic'],'topic_assignment':'heuristic_original_english_prompt_v1',
                        'source_difficulty':None,'final_answer':answer,'available_terminal_verifier':kind,
                        'provenance':{'original_dataset':'TIGER-Lab/MathInstruct','original_row_one_based':pair['upstream_id'],
                            'subset':pair['upstream_source'],'translation_dataset':'ALmonster/MathInstruct-Chinese' if language=='zh' else None,
                            'translation':'public_llm_translation' if language=='zh' else None},
                        'normalized_text_sha256':canonical_hash([language,'',item['prompt'],item['response']])}
                    lang_tokens[language]+=len(answer_ids);lang_rows[language]+=1;topics[pair['topic']]+=1;sources[pair['upstream_source']]+=1
                    answer_types[kind or 'requires_symbolic_or_other_verifier']+=1;totals.append(len(prompt_ids)+len(answer_ids))
                    yield record
        path=output/(split+'.jsonl');write_jsonl(path,rows())
        specs[split]={'path':path.name,'examples':len(selected)*2,'sha256':sha256_file(path)}
        statistics[split]={'records':sum(lang_rows.values()),'groups':len(selected),'languages':dict(lang_rows),
            'answer_tokens':dict(lang_tokens),'zh_answer_token_fraction':lang_tokens['zh']/sum(lang_tokens.values()),
            'topics':dict(topics),'upstream_subsets':dict(sources),'terminal_verifier_coverage':dict(answer_types),
            'min_sequence_tokens':min(totals),'max_sequence_tokens':max(totals),'mean_sequence_tokens':sum(totals)/len(totals)}
    manifest={'schema_version':1,'kind':'sft_tokenized_conversations','fixture':False,
        'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'status':'awaiting_independent_audit',
        'splits':specs,'tokenizer':{'path':'tokenizer.json','sha256':sha256_file(output/'tokenizer.json'),
            'vocab_size':32000,'pad_id':0,'unk_id':1,'eos_id':2},'tokenization':PROTOCOL,
        'chinese_normalization':{'mode':'t2s','package':'OpenCC','package_version':'1.1.9',
            'algorithm':'bounded_fixed_point','max_passes':8,'text_cleanup':'literal_escaped_newline_without_changing_latex_n_commands'},
        'origin':{'source_lock_sha256':report['source_lock_sha256'],'public_sources':read_json(cache/'sources.lock.json')['sources'],
            'local_template_generation_used':False,'translation_is_human_verified':False},
        'statistics':statistics,'quality_report':report,'training_policy':{'epochs':1,'stop_epochs':None},
        'rl_status':{'sft_data_includes_worked_solutions':True,'not_all_answers_are_numeric':True,
            'dpo_pairs':'not_created; sample policy responses and verify before constructing pairs',
            'proofs_require_separate_verification':True},
        'limits':['Heuristic subject tags, not gold taxonomy or difficulty labels',
                  'Publisher solutions and public translations are not independently proved correct',
                  'Exact known-question exclusion is not a guarantee against semantic or pretraining contamination'],
        'training_started':False}
    manifest['data_identity']=canonical_hash({key:manifest[key] for key in IDENTITY_FIELDS})
    write_json(output/'manifest.json',manifest)
    write_json(work/'build_report.json',{'output':str(output),'statistics':statistics,'curation':report})
    print(json.dumps({'status':'awaiting_independent_audit','manifest':str(output/'manifest.json'),'statistics':statistics},ensure_ascii=False),flush=True)
    return output/'manifest.json'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    args=ap.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    build(args.root.resolve(),args.cache.resolve(),args.tokenizer.resolve(),args.output.resolve(),args.work.resolve())
