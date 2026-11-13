"""真实tiny Decoder验证LoRA：零更新等价、冻结、导出与完整恢复。"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_lora.adapters import inject,adapter_state,load_adapter,frozen_digest,assert_frozen,merge_for_inference
from llm_lora.config import validate,runtime_identity
from llm_lora.engine import run_lora
from test_sft_runtime_data import corpus
from test_sft_engine import fixture_config,assert_state_equal


@pytest.fixture(autouse=True)
def single_thread():
    before=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(before)


def settings(rank=2,scope='all'):
    return {'rank':rank,'alpha':rank*2,'dropout':0.,'target_scope':scope,'bias':'none'}


def tiny(dropout=0.):
    torch.manual_seed(7)
    return Decoder(ModelConfig(vocab_size=32,dim=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq_len=32,dropout=dropout))


def test_injection_is_initially_exact_and_keeps_config_attributes():
    model=tiny().eval();x=torch.tensor([[3,4,5]]);before=model(x).logits.detach();digest=frozen_digest(model)
    model,info=inject(model,settings());model.eval()
    assert info['target_modules']==7 and info['trainable_parameters']==512
    assert torch.equal(before,model(x).logits)
    assert frozen_digest(model)==digest and model.config.max_seq_len==32
    assert model.config.to_dict()['tie_word_embeddings'] is True
    assert model.embedding.weight is model.lm_head.weight
    assert_frozen(model)


@pytest.mark.parametrize('scope,targets', [('all',7),('qv',2)])
def test_only_adapter_updates_and_base_tensors_remain_identical(scope,targets):
    model,info=inject(tiny(),settings(scope=scope));digest=frozen_digest(model)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    initial=adapter_state(model)
    for _ in range(3):
        optimizer.zero_grad();loss=model(torch.tensor([[3,4,5]]),torch.tensor([[4,5,2]])).loss
        loss.backward();assert_frozen(model);optimizer.step()
    assert info['target_modules']==targets and frozen_digest(model)==digest
    assert any(not torch.equal(initial[key],value) for key,value in adapter_state(model).items())


def test_adapter_roundtrip_and_merge_preserve_predictions():
    model,_=inject(tiny(),settings());optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    model(torch.tensor([[3,4,5]]),torch.tensor([[4,5,2]])).loss.backward();optimizer.step()
    model.eval();state=adapter_state(model);x=torch.tensor([[4,5]]);expected=model(x).logits.detach()
    other,_=inject(tiny(),settings());load_adapter(other,state);other.eval()
    assert torch.equal(expected,other(x).logits)
    merged=merge_for_inference(other)
    assert torch.allclose(expected,merged(x).logits,atol=1e-6,rtol=1e-5)
    bad=deepcopy(state);bad.pop(next(iter(bad)))
    with pytest.raises(ValueError,match='keys'):load_adapter(model,bad)
    bad=deepcopy(state);bad[next(iter(bad))].fill_(float('nan'))
    with pytest.raises(ValueError,match='finiteness'):load_adapter(model,bad)


def make_config(corpus,name):
    config,_=fixture_config(corpus,output_name=name,tail=True,dropout=.2)
    config['method']='lora';config['adapter']=settings();validate(config)
    return config


def test_resume_is_exact_for_adapter_optimizer_scheduler_rng_and_sampler(corpus):
    config=make_config(corpus,'continuous');first=run_lora(config)
    alternate=deepcopy(config);alternate['output_dir']=str(Path(config['output_dir']).with_name('resumed'))
    partial=run_lora(alternate,stop_after_steps=2)
    assert partial['status']=='paused' and partial['step']==2
    resumed=run_lora(alternate,resume=Path(alternate['output_dir'])/'last.pt')
    assert resumed['status']=='completed' and resumed['step']==first['step']==6
    left=torch.load(Path(config['output_dir'])/'last.pt',weights_only=False)
    right=torch.load(Path(alternate['output_dir'])/'last.pt',weights_only=False)
    for key in ('adapter_state','optimizer','scheduler','scaler','rng','progress','latest_validation','best_validation_loss'):
        assert_state_equal(left[key],right[key])
    assert left['base']['sha256']==right['base']['sha256']
    assert left['identity']['base_frozen_weights_sha256']==right['identity']['base_frozen_weights_sha256']
    assert 'model' not in right and right['checkpoint_kind']=='adapter_full'
    best=torch.load(Path(alternate['output_dir'])/'best.pt',weights_only=False)
    assert 'optimizer' not in best and 'model' not in best and best['checkpoint_kind']=='adapter_only'
    assert (Path(alternate['output_dir'])/'adapter_model.safetensors').is_file()


def test_resume_rejects_changed_rank_or_learning_rate_and_best_checkpoint(corpus):
    config=make_config(corpus,'identity');run_lora(config,stop_after_steps=1);last=Path(config['output_dir'])/'last.pt'
    for key,value in [('learning_rate',.007)]:
        changed=deepcopy(config);changed['train'][key]=value
        with pytest.raises(ValueError,match='identity'):run_lora(changed,resume=last)
    changed=deepcopy(config);changed['adapter']=settings(rank=4)
    with pytest.raises(ValueError,match='identity'):run_lora(changed,resume=last)
    with pytest.raises(ValueError,match='last.pt'):run_lora(config,resume=last.with_name('best.pt'))


@pytest.mark.parametrize('rank', [0,-1,257,True])
def test_invalid_rank_rejected(corpus,rank):
    config=make_config(corpus,'config');config['adapter']['rank']=rank
    with pytest.raises(ValueError,match='rank'):validate(config)


def test_new_package_does_not_change_old_runtime_fingerprint():
    assert runtime_identity()['full_sft_runtime_sha256']=='0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872'
