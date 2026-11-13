"""从从未用于训练/开发/旧测试的候选余量冻结LoRA阶段最终测试。"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import time

from llm_lab.common.io import read_json,sha256_file,write_json,write_jsonl
from llm_lab.data.text_normalization import build_chinese_normalizer
from llm_sft.config import runtime_fingerprint
from tokenizers import Tokenizer
import prepare_sft_expansion as expansion
import audit_sft_data


def freeze(root,stamp):
    root=Path(root).resolve();pool=root/'artifacts/posttrain/sft_expansion_20261027_215627'
    target=root/'artifacts/posttrain'/('lora_evaluation_'+stamp)
    work=root/'runs/assistant/data_prepare'/('lora_evaluation_'+stamp)
    if target.exists() or work.exists():raise FileExistsError('Evaluation version already exists')
    before=runtime_fingerprint();started=time.monotonic()
    report=read_json(pool/'pool_manifest.json')
    if sha256_file(pool/'candidates.jsonl')!=report['candidates_sha256']:raise ValueError('Candidate pool changed')
    train_manifest=read_json(pool/'sft_100k/manifest.json');primary=root/'artifacts/posttrain/sft_300m_20261026_170531'
    auxiliary=pool/'auxiliary_evaluation';dev_manifest=auxiliary/'manifest.json';dev=read_json(dev_manifest)
    ids=set();groups=set();keys={name:set() for name in ('prompt','input_context')};excluded_files=[]
    def protect(path):
        excluded_files.append({'path':str(path),'sha256':sha256_file(path)})
        for row in expansion.row_stream(path):
            ids.add(row['id']);groups.add(row['group_id'])
            for key in keys:
                value=expansion.fingerprint(row[key])
                if value:keys[key].add(value)
    protect(pool/'sft_100k/train.jsonl')
    for split in ('validation','test'):protect(primary/(split+'.jsonl'))
    for split in ('validation','test'):protect(auxiliary/dev['files'][split]['path'])
    review=read_json(pool/'quality_review.json');ids.update(row['id'] for row in review['reviewed'])
    buckets=defaultdict(list);rejected=Counter();eligible=0
    for row in expansion.row_stream(pool/'candidates.jsonl'):
        if row['id'] in ids or row['group_id'] in groups:rejected['used_id_or_group']+=1;continue
        if any(expansion.fingerprint(row[key]) in keys[key] for key in keys if row[key]):
            rejected['used_question_or_material']+=1;continue
        if row['prompt_token_count']+256>1024:rejected['generation_context_budget']+=1;continue
        eligible+=1;rank=int(hashlib.sha256(('lora-final-20261028:'+row['id']).encode()).hexdigest(),16)
        item=(-rank,row['id'],row);bucket=buckets[(row['language'],row['task'])]
        if len(bucket)<2000:heapq.heappush(bucket,item)
        elif item[0]>bucket[0][0]:heapq.heapreplace(bucket,item)
    candidates=[item[2] for bucket in buckets.values() for item in bucket]
    weights=read_json(pool/'build_config.json')['task_weights']
    selected,shortfalls=expansion.nested_selection([],candidates,[1000],weights,20261028)
    if shortfalls:raise ValueError('Insufficient independent balanced final test candidates')
    rows=[{**row,'split':'test'} for row in selected[1000]]
    tokenizer=Tokenizer.from_file(str(pool/'sft_100k/tokenizer.json'));normalizer=build_chinese_normalizer('t2s')
    seen=set();seen_groups=set();exact={key:set() for key in keys}
    for index,row in enumerate(rows,1):
        audit_sft_data._check_row(row,'test',index,tokenizer,train_manifest,normalizer)
        if row['id'] in ids or row['id'] in seen or row['group_id'] in groups or row['group_id'] in seen_groups:raise ValueError('Final test overlap')
        seen.add(row['id']);seen_groups.add(row['group_id'])
        for key in keys:
            value=expansion.fingerprint(row[key])
            if value and (value in keys[key] or value in exact[key]):raise ValueError('Final test exact question/material overlap')
            if value:exact[key].add(value)
    statistics=expansion.base.split_statistics(rows)
    zh=statistics['languages']['zh']['answer_tokens'];en=statistics['languages']['en']['answer_tokens']
    if not .45<=zh/(zh+en)<=.55:raise ValueError('Final test language Token balance failed')
    target.mkdir(parents=True,exist_ok=False);work.mkdir(parents=True,exist_ok=False)
    write_jsonl(target/'test.jsonl',rows);shutil.copy2(pool/'sft_100k/tokenizer.json',target/'tokenizer.json')
    manifest={'schema_version':1,'kind':'sft_auxiliary_evaluation','files':{'test':{'path':'test.jsonl','examples':1000,'sha256':sha256_file(target/'test.jsonl')}},
        'tokenizer':train_manifest['tokenizer'],'tokenization':train_manifest['tokenization'],'chinese_normalization':train_manifest['chinese_normalization'],
        'statistics':statistics,'pool_sha256':report['candidates_sha256'],'excluded_files':excluded_files,
        'new_test_before_lora_tuning':True,'semantic_gold':False,'builder_sha256':sha256_file(Path(__file__))}
    write_json(target/'manifest.json',manifest)
    write_json(target/'acceptance.json',{'status':'passed','training_review_and_split_isolation':True,'retokenization':True,
        'answer_mask_and_eos':True,'chinese_fixed_point':True,'generation_budget_preserves_all_prompts':True,'semantic_gold':False})
    protocol={'schema_version':1,'kind':'lora_evaluation_protocol','status':'frozen_before_lora_tuning',
        'training_manifest':str(pool/'sft_100k/manifest.json'),'training_manifest_sha256':sha256_file(pool/'sft_100k/manifest.json'),
        'primary_validation_manifest':str(primary/'manifest.json'),'primary_validation_manifest_sha256':sha256_file(primary/'manifest.json'),
        'development':{'manifest':str(dev_manifest),'manifest_sha256':sha256_file(dev_manifest),'split':'validation','examples':2000},
        'final_test':{'manifest':str(target/'manifest.json'),'manifest_sha256':sha256_file(target/'manifest.json'),
                      'test_sha256':manifest['files']['test']['sha256'],'examples':1000},
        'selection_rule':'Choose smallest auxiliary-development macro answer loss at the same 3125-update screen budget; checkpoints inside a run use original primary validation. Retention/generation are reported separately.',
        'final_generations_per_model':1000,'generation_budget':256,'historical_tests_are_not_final':True,
        'final_test_content_exposed_to_tuning':False,'deduplication_limits':report['deduplication_limits']}
    write_json(target/'protocol.json',protocol)
    result={'status':'completed','protocol':str(target/'protocol.json'),'protocol_sha256':sha256_file(target/'protocol.json'),
        'eligible_candidates':eligible,'final_test_examples':1000,'tasks':statistics['tasks_heuristic'],
        'chinese_answer_token_fraction':zh/(zh+en),'rejections':dict(rejected),'output_dir':str(target),'work_dir':str(work),
        'test_text_exposed_to_agent':False,'gpu_training_started':False,'elapsed_seconds':time.monotonic()-started}
    assert runtime_fingerprint()==before
    write_json(work/'report.json',result);write_json(root/'artifacts/posttrain/latest_lora_evaluation.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stamp',required=True)
    args=parser.parse_args();print(json.dumps(freeze(args.root,args.stamp),ensure_ascii=False),flush=True)
