"""按冻结抽审清单生成不可覆盖的 SFT 子版本；不改已构建父数据。"""
from __future__ import annotations
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import shutil

from prepare_sft_data import atomic_json,canonical,sha256,split_statistics,take_balanced,under,write_jsonl


def publish(config_path: Path):
    settings=json.loads(config_path.read_text(encoding='utf-8'))
    root=(config_path.resolve().parent/settings['project_root']).resolve()
    source=under(root,settings['parent_dataset']);target=under(root,settings['output_dir'])
    parent=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    identity_fields=('schema_version','kind','splits','tokenizer','tokenization','chinese_normalization')
    if canonical({key:parent[key] for key in identity_fields})!=parent['data_identity']:
        raise ValueError('Parent manifest data identity is stale or modified')
    if parent['data_identity']!=settings['parent_data_identity'] or parent['acceptance']['status']!='passed':
        raise ValueError('Curated publication parent identity/acceptance mismatch')
    if target.exists():
        raise ValueError('Refusing to overwrite a curated SFT dataset')
    exclusions={(item['dataset'],item['row_index']):item for item in settings['exclusions']}
    datasets={};removed=[];seen=set()
    for split in ('train','validation','test'):
        spec=parent['splits'][split];path=source/spec['path']
        if sha256(path)!=spec['sha256']:raise ValueError('Parent split changed')
        keep=[]
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);key=(row['source']['dataset'],row['source']['row_index'])
                if key in exclusions:
                    if split=='test':raise ValueError('Spot-check curation must not target test content')
                    seen.add(key);removed.append({'id':row['id'],'split':split,**exclusions[key]})
                else:keep.append(row)
        datasets[split]=keep
    if seen!=set(exclusions):raise ValueError('Curation exclusion was not found in train/validation')
    queues={language:deque(row for row in datasets['train'] if row['language']==language) for language in ('zh','en')}
    datasets['pilot']=[{**row,'split':'pilot'} for row in take_balanced(queues,2000,42)]
    target.mkdir(parents=True,exist_ok=False)
    manifest=copy.deepcopy(parent)
    stats={}
    for split,rows in datasets.items():
        path=target/(split+'.jsonl')
        if split=='test':
            # 不读取测试题作内容反馈；发布阶段按字节保留已经冻结的测试文件。
            shutil.copy2(source/parent['splits']['test']['path'],path)
        else:write_jsonl(path,rows)
        manifest['splits'][split]={'path':path.name,'sha256':sha256(path),'examples':len(rows)}
        stats[split]=split_statistics(rows)
    for name in ('tokenizer.json','build_config.json'):
        shutil.copy2(source/name,target/name)
    if sha256(target/'tokenizer.json')!=parent['tokenizer']['sha256']:raise ValueError('Tokenizer changed')
    atomic_json(target/'curation_config.json',settings)
    retained={row['id'] for split in ('train','validation') for row in datasets[split]}
    review=[json.loads(line) for line in (source/'review_sample.jsonl').read_text(encoding='utf-8').splitlines()]
    write_jsonl(target/'review_sample.jsonl',[row for row in review if row['id'] in retained])
    atomic_json(target/'statistics.json',{'splits':stats,'curation_removed':removed,
                                         'parent_statistics_path':settings['parent_dataset']+'/statistics.json'})
    manifest['curation']={'parent_dataset':settings['parent_dataset'],'parent_data_identity':parent['data_identity'],
                          'config':'curation_config.json','config_sha256':sha256(target/'curation_config.json'),
                          'publisher_sha256':sha256(Path(__file__)),'removed_examples':len(removed),
                          'test_bytes_unchanged':manifest['splits']['test']['sha256']==parent['splits']['test']['sha256']}
    manifest['quality']['status']='automatic_heuristics_plus_documented_llm_spot_checks'
    manifest['quality']['semantic_gold']=False
    manifest['quality']['spot_check_examples']=len(settings['reviewed_examples'])
    manifest['quality']['spot_check_rejections']=len(removed)
    manifest['acceptance']={'status':'pending_independent_audit'}
    keys=('schema_version','kind','splits','tokenizer','tokenization','chinese_normalization')
    manifest['data_identity']=canonical({key:manifest[key] for key in keys})
    atomic_json(target/'manifest.json',manifest)
    atomic_json(target/'quality_review.json',{'data_identity':manifest['data_identity'],
               'method':'stratified LLM qualitative review, not expert human gold',
               'reviewed_examples':settings['reviewed_examples'],'removed':removed,
               'test_content_used_for_selection':False,'semantic_correctness_of_all_rows_verified':False,
               'limits':'Deliberate source/task strata are not a random estimate of population error rate.'})
    print(json.dumps({'status':'published_pending_audit','output_dir':str(target),
                     'data_identity':manifest['data_identity'],'statistics':stats,'removed_examples':len(removed)},ensure_ascii=False,indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    publish(parser.parse_args().config)
