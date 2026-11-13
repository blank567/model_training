from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from supplement_reading_dpo_pairs import combine_unique,select_matched,unresolved_chinese,exchange_balanced


def row(i,language):return {'id':str(i),'language':language,'answer_token_count':2}


def test_explicit_minimum_is_checked_when_not_in_hundred_step_sequence():
    old=[row(i,'zh' if i<2 else 'en') for i in range(5)]
    new=[row(i+10,'en' if i<2 else 'zh') for i in range(5)]
    left,right,balance=select_matched(old,new,minimum=4)
    assert len(left)==len(right)==4 and balance['old']['zh_answer_token_fraction']==.5
    assert len({r['id'] for r in left})==4


def test_supplement_cannot_duplicate_original_questions():
    with pytest.raises(ValueError,match='duplicate'):combine_unique([row(1,'zh')],[row(1,'zh')])


def test_only_original_failed_chinese_prompts_are_targeted():
    arm=[row(1,'zh'),row(2,'en'),row(3,'zh')]
    assert unresolved_chinese(arm,{'1'})==[2]


def test_length_exchanges_retain_count_unique_ids_and_token_balance():
    pool=[{'id':'z'+str(i),'language':'zh','answer_token_count':10} for i in range(80)]
    pool += [{'id':'e'+str(i),'language':'en','answer_token_count':9 if i<75 else 5} for i in range(150)]
    selected,balance=exchange_balanced(pool,200)
    assert len(selected)==len({r['id'] for r in selected})==200
    assert .495<=balance['zh_answer_token_fraction']<=.505


def test_impossible_token_capacity_is_not_silently_relaxed():
    pool=[{'id':'z','language':'zh','answer_token_count':1}]
    pool += [{'id':'e'+str(i),'language':'en','answer_token_count':10} for i in range(10)]
    assert exchange_balanced(pool,10)==(None,None)
