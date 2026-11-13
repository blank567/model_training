"""禁止不足预算候选选模，客观约束只覆盖实际识别到的要求。"""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_lora_experiments import screen_result,select_candidate
from llm_lora.metrics import supported_constraints,summarize_constraints


def test_incomplete_or_missing_development_cannot_enter_selection():
    for value in ({'status':'completed','step':3125,'latest_development':{'macro_loss':1.}},
                  {'status':'paused','step':3000,'latest_development':{'macro_loss':1.}},
                  {'status':'paused','step':3125}):
        with pytest.raises(ValueError):screen_result(value,3125)
    assert screen_result({'status':'paused','step':3125,'latest_development':{'macro_loss':2.}},3125)==2
    with pytest.raises(ValueError):select_candidate([{'key':'a','development_loss':1.}])
    assert select_candidate([{'key':key,'development_loss':loss} for key,loss in [('a',2.),('b',1.),('c',3.)]])['key']=='b'


def test_supported_constraints_do_not_treat_unchecked_answers_as_correct():
    unknown=supported_constraints('Explain clouds.','A cloud.')
    assert unknown['all_supported_passed'] is None
    bad=supported_constraints('All lowercase. Exactly 2 bullet points.','* First\n* second')
    assert bad['supported_check_count']==2 and bad['all_supported_passed'] is False
    good=supported_constraints('Return only JSON.','{"key":1}')
    assert good['all_supported_passed'] is True
    total=summarize_constraints([{'constraint_checks':value} for value in (unknown,bad,good)])
    assert total['covered_examples']==2 and total['all_supported_constraints_pass_fraction']==.5


def test_keyword_frequency_counts_whole_words():
    value=supported_constraints("The word 'important' should appear at least 3 times.",'important importance important')
    assert value['all_supported_passed'] is False
