"""SFT 编码边界、语言配额和 holdout 规范化的对抗性回归。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from collections import deque

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import prepare_sft_data as build


def test_encoding_supervises_answer_and_eos_but_never_role_prompt(tmp_path):
    from llm_lab.tokenization.bpe import train_bpe
    tokenizer=train_bpe(['System: Guide\nUser: cats\nAssistant: dogs']*10,tmp_path/'tokenizer.json',vocab_size=300,min_frequency=1)
    row={'system':'Guide','prompt':'cats','response':'dogs'}
    ids,mask,prefix_count,answer_count=build.encode_row(row,tokenizer,2)
    assert mask[:prefix_count]==[0]*prefix_count
    assert mask[prefix_count:]==[1]*answer_count
    assert ids[-1]==2 and mask[-1]==1
    assert tokenizer.decode(ids[:prefix_count],skip_special_tokens=False).endswith('Assistant:\n')
    assert tokenizer.decode(ids[prefix_count:-1])=='dogs'


def test_balancing_uses_supervision_tokens_instead_of_equal_sample_counts():
    queues={'zh':deque({'id':str(i),'answer_token_count':10} for i in range(100)),
            'en':deque({'id':'en'+str(i),'answer_token_count':30} for i in range(100))}
    rows=build.take_balanced(queues,40,42)
    zh=sum(row['answer_token_count'] for row in rows if not row['id'].startswith('en'))
    en=sum(row['answer_token_count'] for row in rows if row['id'].startswith('en'))
    assert zh==en==300
    assert len([r for r in rows if r['id'].startswith('en')])==10


def test_balanced_selection_rejects_missing_language():
    with pytest.raises(ValueError,match='Insufficient'):
        build.take_balanced({'zh':deque(),'en':deque()},2,42)


def test_holdout_compares_compatible_new_normalization_after_original_sha_check(tmp_path):
    # 原文是全角字符；原始身份不能先NFKC后再校验，否则合法holdout会被误拒。
    text='ＡＢＣＤＥＦＧＨＩＪ'*15
    path=tmp_path/'documents.jsonl'
    path.write_text(json.dumps({'split':'test','language':'en','text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest()}),encoding='utf-8')
    class Identity:
        def convert(self,value,language):return value
    overlaps,report=build.holdout_guard(path,Identity())
    assert report['documents']==1
    assert overlaps('prefix '+('abcdefghij'*15)+' suffix')
    assert not overlaps('unrelated short input')


def test_modified_holdout_row_fails_before_leakage_check(tmp_path):
    path=tmp_path/'documents.jsonl'
    path.write_text(json.dumps({'split':'validation','language':'zh','text':'changed','text_sha256':'0'*64}),encoding='utf-8')
    with pytest.raises(ValueError,match='SHA'):
        build.holdout_guard(path,None)


def test_path_escape_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='under'):
        build.under(tmp_path,'../outside')


def test_dolly_grounding_context_is_required():
    source={'name':'dolly','format':'dolly_jsonl','category_allowlist':['closed_qa'],'require_context':True}
    row,reason=build.extract({'instruction':'Who?','context':'','response':'Someone','category':'closed_qa'},source)
    assert row is None and reason=='missing_grounding_context'


def test_smol_followup_is_not_silently_treated_as_self_contained():
    source={'name':'smol','format':'messages_parquet','source_allowlist':['short']}
    row,reason=build.extract({'source':'short','messages':[{'role':'assistant','content':'Earlier answer'},{'role':'user','content':'Why?'}]},source)
    assert row is None and reason=='not_self_contained_first_turn'
