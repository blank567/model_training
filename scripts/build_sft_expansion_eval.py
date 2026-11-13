"""从扩充池余量冻结辅助开发/测试集，不更改主SFT训练与评测文件。"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

import prepare_sft_expansion as expansion
import audit_sft_data
from llm_lab.data.text_normalization import build_chinese_normalizer
from tokenizers import Tokenizer


def unused_candidates(rows,train_ids,review_ids):
    """内容抽审曾见过的题也不进入辅助评测，防止筛选反馈影响评测。"""
    seen=set();result=[]
    for row in rows:
        if row['id'] in seen:raise ValueError('Repeated candidate ID')
        seen.add(row['id'])
        if row['id'] not in train_ids and row['id'] not in review_ids:result.append(row)
    return result


def verify_auxiliary_rows(datasets,train_ids,review_ids,manifest,tokenizer,normalizer):
    seen=set();groups=set();exact={key:set() for key in ('prompt','input_context')}
    reports={}
    for split,rows in datasets.items():
        for index,row in enumerate(rows,1):
            if row['id'] in train_ids or row['id'] in review_ids or row['id'] in seen:
                raise ValueError('Auxiliary evaluation overlaps training, content review or another split')
            if row['group_id'] in groups:raise ValueError('Auxiliary evaluation group overlap')
            audit_sft_data._check_row(row,split,index,tokenizer,manifest,normalizer)
            seen.add(row['id']);groups.add(row['group_id'])
            for key in exact:
                value=expansion.fingerprint(row[key])
                if value and value in exact[key]:raise ValueError('Auxiliary exact question/context overlap')
                if value:exact[key].add(value)
        statistics=expansion.base.split_statistics(rows)
        zh=statistics['languages']['zh']['answer_tokens'];en=statistics['languages']['en']['answer_tokens']
        if not .45<=zh/(zh+en)<=.55:raise ValueError('Auxiliary language Token balance failed')
        reports[split]=statistics
    return reports


def build(directory,validation_count=2000,test_count=1000):
    directory=Path(directory).resolve();release=expansion.read_json(directory/'release.json')
    if (directory/'auxiliary_evaluation').exists():raise ValueError('Auxiliary evaluation is already frozen')
    review=expansion.read_json(directory/'quality_review.json');review_ids={row['id'] for row in review['reviewed']}
    train_ids=set();train_groups=set();train_exact={key:set() for key in ('prompt','input_context')};parents=[]
    for item in release['datasets']:
        path=Path(item['manifest']);manifest=expansion.read_json(path)
        parents.append({'manifest':str(path),'sha256':expansion.base.sha256(path),'data_identity':manifest['data_identity']})
        for row in expansion.row_stream(path.parent/manifest['splits']['train']['path']):
            train_ids.add(row['id']);train_groups.add(row['group_id'])
            for key in train_exact:
                value=expansion.fingerprint(row[key])
                if value:train_exact[key].add(value)
    unused=unused_candidates(expansion.row_stream(directory/'candidates.jsonl'),train_ids,review_ids)
    selected,shortfalls=expansion.nested_selection([],unused,[validation_count,validation_count+test_count],
                                                  expansion.read_json(directory/'build_config.json')['task_weights'],20261027)
    if shortfalls:raise ValueError('Insufficient unused balanced candidates for auxiliary evaluation')
    validation=selected[validation_count];validation_ids={row['id'] for row in validation}
    test=[row for row in selected[validation_count+test_count] if row['id'] not in validation_ids]
    datasets={'validation':[{**row,'split':'validation'} for row in validation],
              'test':[{**row,'split':'test'} for row in test]}
    reference=Path(release['datasets'][0]['manifest']);manifest=expansion.read_json(reference)
    tokenizer=Tokenizer.from_file(str(reference.parent/manifest['tokenizer']['path']))
    normalizer=build_chinese_normalizer('t2s')
    report=verify_auxiliary_rows(datasets,train_ids,review_ids,manifest,tokenizer,normalizer)
    for rows in datasets.values():
        for row in rows:
            if row['group_id'] in train_groups:raise ValueError('Auxiliary group overlaps a training group')
            for key in train_exact:
                value=expansion.fingerprint(row[key])
                if value and value in train_exact[key]:raise ValueError('Auxiliary question/context overlaps training')
    target=directory/'auxiliary_evaluation';target.mkdir(exist_ok=False)
    files={}
    for split,rows in datasets.items():
        path=target/(split+'.jsonl');expansion.base.write_jsonl(path,rows)
        files[split]={'path':path.name,'examples':len(rows),'sha256':expansion.base.sha256(path)}
    shutil.copy2(reference.parent/manifest['tokenizer']['path'],target/'tokenizer.json')
    spec={'schema_version':1,'kind':'sft_auxiliary_evaluation','files':files,'tokenizer':manifest['tokenizer'],
          'tokenization':manifest['tokenization'],'chinese_normalization':manifest['chinese_normalization'],
          'train_parents':parents,'pool_sha256':expansion.base.sha256(directory/'candidates.jsonl'),
          'excludes_all_training_and_content_review_ids':True,'primary_validation_and_test_unchanged':True,
          'semantic_gold':False,'test_content_used_for_review':False,
          'builder_sha256':expansion.base.sha256(Path(__file__)),'statistics':report}
    expansion.base.atomic_json(target/'manifest.json',spec)
    expansion.base.atomic_json(target/'acceptance.json',{'status':'passed','file_hashes':True,
        'retokenization':True,'answer_mask_and_eos':True,'chinese_fixed_point':True,
        'training_review_and_split_isolation':True,'language_token_balance':True,'semantic_gold':False})
    return {'status':'passed','directory':str(target),'manifest':str(target/'manifest.json'),
            'validation_examples':validation_count,'test_examples':test_count,'test_content_exposed_to_review':False,
            'statistics':report}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',required=True)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false')
    if hasattr(os,'sched_getaffinity'):
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[-2:]))
    if hasattr(os,'nice'):os.nice(10)
    print(json.dumps(build(args.data_dir),ensure_ascii=False,indent=2))
