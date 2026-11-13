"""测试错位、右补齐生成、EOS终止、固定抽样和比较指标语义。"""
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_sft_models as evaluation


class NextToken(torch.nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(max_seq_len=40)
    def forward(self,values):
        logits=torch.full((*values.shape,12),-100.)
        target=(values+1).clamp(max=11)
        target[values==6]=2
        logits.scatter_(2,target.unsqueeze(-1),100.)
        return SimpleNamespace(logits=logits)


def test_ragged_greedy_matches_independent_generation_and_stops_on_eos():
    model=NextToken();prefixes=[[3],[3,4,5],[4,5,6]]
    actual=evaluation.greedy_batch(model,prefixes,8)
    expected=[evaluation.greedy_batch(model,[prefix],8)[0] for prefix in prefixes]
    assert actual==expected
    assert actual[0]['generated_ids']==[4,5,6,2]
    assert all(row['stop_reason']=='eos' for row in actual)


def test_generation_budget_does_not_truncate_prompt_or_exceed_context():
    with pytest.raises(ValueError,match='context'):
        evaluation.greedy_batch(NextToken(),[[3]*36],8)
    result=evaluation.greedy_batch(NextToken(),[[3]],2)[0]
    assert result=={'generated_ids':[4,5],'stop_reason':'max_new_tokens'}


def test_selection_excludes_base_and_does_not_relabel_token_accuracy_as_task_accuracy():
    rows=[{'key':'base','candidate':False,'test':{'macro_loss':0.1}},
          {'key':'a','candidate':True,'test':{'macro_loss':2.2}},
          {'key':'b','candidate':True,'test':{'macro_loss':2.1}}]
    assert evaluation.select_model(rows)['key']=='b'


def test_generation_sample_balances_strata_and_is_stable_without_model_outputs():
    rows=[{'id':str(index),'language':language,'task':task,'prompt_token_count':8}
          for index,(language,task) in enumerate([('en','summary'),('zh','rewrite')]*5)]
    selected=evaluation.generation_indices(rows,6,42,128,16)
    assert selected==evaluation.generation_indices(rows,6,42,128,16)
    assert len(selected)==len(set(selected))==6
    assert sum(rows[index]['language']=='en' for index in selected)==3


def test_rouge_is_only_reference_overlap_and_handles_empty_output():
    assert evaluation.rouge_l_f1('小猫。','小猫！','zh')==1
    assert evaluation.rouge_l_f1('a red cat','a blue cat','en')==pytest.approx(2/3)
    assert evaluation.rouge_l_f1('','reference','en')==0


def test_paired_bootstrap_preserves_pairing():
    rows=[{'id':language+str(index),'language':language,'tokens':10,'loss_sum':20}
          for language in ('en','zh') for index in range(4)]
    other=[{**row,'loss_sum':21} for row in reversed(rows)]
    interval=evaluation.paired_interval(rows,other,samples=20)
    assert interval['percentile_95']==pytest.approx([.1,.1])
    with pytest.raises(ValueError,match='differ'):
        evaluation.paired_interval(rows,other[:-1],samples=20)


def test_loss_counts_shifted_answer_tokens_and_ignores_prefix_and_padding():
    class Corpus:
        rows=[{'id':'en','language':'en','task':'qa','input_ids':[3,4,5,6,2]},
              {'id':'zh','language':'zh','task':'qa','input_ids':[3,4,5]}]
        def collate(self,indices,device):
            x=torch.tensor([[3,4,5,6] if index==0 else [3,4,0,0] for index in indices])
            y=torch.tensor([[-100,5,6,2] if index==0 else [-100,5,-100,-100] for index in indices])
            return {'x':x,'y':y}
    result,rows=evaluation.evaluate_loss(NextToken(),Corpus(),
        {'train':{'device':'cpu','dtype':'fp32','eval_micro_batch_size':2}})
    assert result['macro_loss']==0
    assert result['en']['supervised_tokens']==3 and result['zh']['supervised_tokens']==1
    assert result['en']['teacher_forced_token_accuracy']==1
