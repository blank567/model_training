"""只读评测的错位长度、EOS、模型复用及配对统计检查。"""
from copy import deepcopy
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from evaluate_posttrain_comprehensive import cached_greedy, cluster_interval, exact_mcnemar, holm, state_digest
from evaluate_sft_models import greedy_batch


def test_cached_decode_matches_native_for_variable_lengths_and_forbidden_tokens():
    torch.manual_seed(991)
    m=Decoder(ModelConfig(vocab_size=40,dim=16,n_layers=2,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq_len=32,dropout=.2)).eval()
    before=state_digest(m.state_dict());prefixes=[[4,5],[6,7,8,9,10],[4,5]]
    actual=cached_greedy(m,prefixes,8,forbidden=(0,1))
    expected=greedy_batch(m,prefixes,8,forbidden=(0,1))
    assert actual==expected
    assert state_digest(m.state_dict())==before
    assert all(not any(t in {0,1} for t in r['generated_ids']) for r in actual)
    with pytest.raises(ValueError):cached_greedy(m,[[4]*30],8)


def test_cluster_interval_pairs_ids_not_input_order_and_detects_missing_examples():
    a=[{'id':str(i),'language':'zh' if i<4 else 'en','group_id':str(i//2),'exact_match':0.} for i in range(8)]
    b=[{**r,'exact_match':1.} for r in reversed(a)]
    result=cluster_interval(a,b,samples=40)
    assert result['delta_contender_minus_reference']==1.
    assert result['percentile_95']==[1.,1.]
    assert result['groups']['zh']['clusters']==2
    assert exact_mcnemar(a,b)['two_sided_exact_p']==pytest.approx(2/256)
    with pytest.raises(ValueError):cluster_interval(a,b[:-1],samples=40)


def test_holm_is_monotone_in_sorted_pvalues_and_preserves_original_order():
    assert holm([.04,.001,.03])==pytest.approx([.06,.003,.06])
