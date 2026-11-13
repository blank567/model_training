"""冻结协议、防止测试混入调参，以及命名/配置不可变性。"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from llm_lab.common.io import write_json,write_jsonl,sha256_file
from llm_lora.config import read_config
from llm_lora.data import AuxiliaryCorpus
import train_lora
from test_sft_runtime_data import corpus
from test_sft_engine import fixture_config


def test_freeze_names_and_source_identity_protect_adapter_configuration(corpus):
    config,_=fixture_config(corpus,output_name='template');config['method']='lora'
    config['adapter']={'rank':2,'alpha':4,'dropout':0.,'target_scope':'all','bias':'none'}
    path=Path(config['project_root'])/'lora_template.json';write_json(path,config)
    prepared=train_lora.prepare(path,'20261028_120001')
    assert '_data2_r2_a4_all_' in prepared['RUN_NAME']
    frozen=train_lora.frozen_config(prepared['RUN_CONFIG'])
    assert frozen['method']=='lora' and frozen['adapter']['rank']==2
    changed=json.loads(Path(prepared['RUN_CONFIG']).read_text(encoding='utf-8'));changed['adapter']['rank']=4
    write_json(prepared['RUN_CONFIG'],changed)
    with pytest.raises(ValueError,match='config changed'):train_lora.frozen_config(prepared['RUN_CONFIG'])


def test_auxiliary_development_does_not_open_a_final_test(corpus):
    directory,manifest,rows,tokenizer=corpus;target=directory/'evaluation';target.mkdir()
    shutil.copy2(directory/'tokenizer.json',target/'tokenizer.json')
    selected=deepcopy(rows['validation']);write_jsonl(target/'validation.jsonl',selected)
    tokenization={**manifest['tokenization'],'prefix_template':'System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n',
        'empty_system_prefix_template':'User:\n{prompt}\n\nAssistant:\n'}
    value={'kind':'sft_auxiliary_evaluation','fixture':True,'tokenizer':manifest['tokenizer'],'tokenization':tokenization,
        'files':{'validation':{'path':'validation.jsonl','sha256':sha256_file(target/'validation.jsonl'),'examples':len(selected)},
                 'test':{'path':'sealed-and-nonexistent.jsonl','sha256':'0'*64,'examples':1000}}}
    write_json(target/'manifest.json',value);write_json(target/'acceptance.json',{'status':'passed','training_review_and_split_isolation':True})
    dataset=AuxiliaryCorpus(target/'manifest.json','validation',sha256_file(target/'manifest.json'),allow_fixture=True)
    assert len(dataset)==2 and not (target/'sealed-and-nonexistent.jsonl').exists()
    assert dataset.collate([0,1])['supervised_tokens']==sum(row['answer_token_count'] for row in selected)
    with pytest.raises(ValueError,match='manifest changed'):
        AuxiliaryCorpus(target/'manifest.json','validation','0'*64,allow_fixture=True)
