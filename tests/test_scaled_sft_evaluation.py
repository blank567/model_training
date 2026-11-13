"""辅助测试采用独立协议，拒绝身份变化、泄漏的文件路径和Mask变更。"""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_scaled_sft import AuxiliaryCorpus
from llm_lab.common.io import sha256_file,write_json,write_jsonl
from test_sft_data_audit import corpus


def auxiliary(corpus):
    directory,manifest,rows,tokenizer=corpus;target=directory/'auxiliary';target.mkdir()
    shutil.copy2(directory/manifest['tokenizer']['path'],target/'tokenizer.json')
    write_jsonl(target/'test.jsonl',rows['test'])
    spec={'kind':'sft_auxiliary_evaluation','fixture':True,'tokenizer':manifest['tokenizer'],
        'tokenization':manifest['tokenization'],'files':{'test':{'path':'test.jsonl',
        'sha256':sha256_file(target/'test.jsonl'),'examples':len(rows['test'])}}}
    write_json(target/'manifest.json',spec)
    write_json(target/'acceptance.json',{'status':'passed','training_review_and_split_isolation':True})
    return target/'manifest.json'


def test_auxiliary_reader_retokenizes_and_uses_same_shifted_mask(corpus):
    path=auxiliary(corpus);data=AuxiliaryCorpus(path,sha256_file(path),allow_fixture=True)
    batch=data.collate(list(range(len(data.rows))))
    assert batch['supervised_tokens']==sum(row['answer_token_count'] for row in data.rows)
    assert data.identity['split']=='test'
    with pytest.raises(ValueError,match='Fixture'):
        AuxiliaryCorpus(path,sha256_file(path))


def test_auxiliary_test_file_change_is_rejected(corpus):
    path=auxiliary(corpus);digest=sha256_file(path)
    with (path.parent/'test.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='test changed'):
        AuxiliaryCorpus(path,digest,allow_fixture=True)


def test_auxiliary_manifest_change_is_rejected(corpus):
    path=auxiliary(corpus)
    with pytest.raises(ValueError,match='manifest changed'):
        AuxiliaryCorpus(path,'0'*64,allow_fixture=True)
