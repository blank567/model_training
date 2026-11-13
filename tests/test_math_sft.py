"""真实 tiny Decoder 从 SFT 权重初始化以及单轮精确恢复，全部为 CPU fixture。"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from llm_lab.common.io import canonical_hash,sha256_file
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_sft.config import read_config
from llm_align.math_sft import run,runtime_identity,validate
from llm_math_public.build import PROTOCOL
from test_sft_runtime_data import corpus,repin

ROOT=Path(__file__).resolve().parents[1]


def config_for(corpus,name):
    directory,manifest,rows,tokenizer=corpus
    manifest['tokenization']=deepcopy(PROTOCOL)
    repin(directory,manifest,rows)
    parent=directory/'parent_sft.pt'
    if not parent.exists():
        torch.manual_seed(700)
        spec=ModelConfig(vocab_size=tokenizer.get_vocab_size(),dim=16,n_layers=2,n_heads=2,n_kv_heads=1,
                         ffn_dim=32,max_seq_len=1024,dropout=.1,activation_checkpointing=False)
        model=Decoder(spec)
        torch.save({'schema_version':1,'stage':'sft','checkpoint_kind':'model_only','step':4,
            'config':{'data_manifest':str(directory/'manifest.json'),'project_root':str(ROOT),'method':'full'},
            'model_config':asdict(spec),'model':model.state_dict(),
            'identity':{'manifest_sha256':sha256_file(directory/'manifest.json'),'data_identity':manifest['data_identity'],
                        'tokenizer_sha256':manifest['tokenizer']['sha256']}},parent)
    raw={'project_root':str(ROOT),'base_checkpoint':str(parent),'data_manifest':str(directory/'manifest.json'),
        'output_dir':str(directory/name),'fixture':True,'min_free_bytes':0,
        'base_checkpoint_sha256':sha256_file(parent),'data_manifest_sha256':sha256_file(directory/'manifest.json'),
        'data_identity':manifest['data_identity'],'runtime_sha256':canonical_hash(runtime_identity(ROOT)),
        'train':{'device':'cpu','dtype':'fp32','epochs':1,'stop_epochs':None,'learning_rate':.002,
            'micro_batch_size':1,'gradient_accumulation_steps':1,'eval_interval':1,'save_interval':1,'log_interval':1,
            'warmup_ratio':.5,'activation_checkpointing':False,'retention_eval_batches':0,'tensorboard':False,'wandb':False},
        'budget':{'hourly_rate_cny':0,'run_limit_cny':1,'project_limit_cny':500,'reserve_cny':100,
                  'max_runtime_seconds':60,'ledger_path':str(directory/'ledger.json')}}
    path=directory/(name+'.json');path.write_text(json.dumps(raw),encoding='utf-8')
    return read_config(path),parent


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def test_initial_weights_are_completed_sft_not_fresh_parameters(corpus):
    cfg,parent=config_for(corpus,'initial');parent_hash=sha256_file(parent)
    report=run(cfg,stop_after_steps=0)
    got=load(Path(cfg['output_dir'])/'last.pt')['model'];expected=load(parent)['model']
    assert report['step']==0 and report['plan']['epochs']==1
    assert all(torch.equal(got[k],expected[k]) for k in got)
    assert sha256_file(parent)==parent_hash


def test_one_epoch_resume_matches_uninterrupted(corpus):
    continuous,_=config_for(corpus,'continuous');first=run(continuous)
    resumed,_=config_for(corpus,'resumed');pause=run(resumed,stop_after_steps=1)
    assert pause['step']==1 and pause['status']=='bounded_smoke_stop'
    second=run(resumed,resume=Path(resumed['output_dir'])/'last.pt')
    assert first['step']==second['step']==2 and second['examples_seen']==2
    a,b=(load(Path(cfg['output_dir'])/'last.pt') for cfg in [continuous,resumed])
    assert all(torch.equal(a['model'][key],b['model'][key]) for key in a['model'])
    assert a['scheduler']==b['scheduler']
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])


def test_new_math_sft_rejects_extra_epochs(corpus):
    cfg,_=config_for(corpus,'bad_epochs');cfg['train']['epochs']=2
    with pytest.raises(ValueError,match='epochs=1'):validate(cfg)
