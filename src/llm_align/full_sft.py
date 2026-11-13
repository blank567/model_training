"""从已完成的全参数 SFT 初始化领域 SFT；复用冻结数值内核，支持真实单轮和精确恢复。"""
from __future__ import annotations

from dataclasses import asdict
import copy
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

import numpy as np
import torch

from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_lab.common.tracking import start_tracker
from llm_lab.engine.checkpoint import save_checkpoint
from llm_lab.engine.pretrain import lr_factor
from llm_sft.config import validate_config,runtime_fingerprint as frozen_full_runtime
from llm_sft.data import SFTCorpus,deterministic_epoch_batches
from llm_sft.engine import train_update,evaluate_sft_model,_rng,_restore_rng,_environment,_console_json

FROZEN_FULL='0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872'


def runtime_identity(root):
    root=Path(root)
    names=['src/llm_align/full_sft.py','src/llm_align/math_sft.py','scripts/train_math_sft.py',
           'scripts/train_reading_sft.py','scripts/chat_sft.py']
    return {'files':{name:sha256_file(root/name) for name in names},'frozen_full_runtime':frozen_full_runtime(root)}


def validate(config):
    validate_config(config)
    if config['train']['epochs']!=1 or config['train']['stop_epochs'] is not None:
        raise ValueError('New domain SFT must use epochs=1 and stop_epochs=null')
    if config['train_split']!='train':raise ValueError('Formal domain SFT uses the accepted training split only')
    if config['train']['retention_eval_batches']!=0:raise ValueError('This entry does not implicitly mix in pretraining data')


def run(config,resume=None,stop_after_steps=None):
    validate(config);config=copy.deepcopy(config);root=Path(config['project_root']);out=Path(config['output_dir'])
    train=config['train'];device=train['device'];started=time.monotonic()
    if resume is None and out.exists() and any(out.iterdir()):raise ValueError('Output is nonempty; use its own resume')
    if resume is not None and Path(resume).resolve()!=(out/'last.pt').resolve():raise ValueError('Resume must use this run last.pt')
    runtime=runtime_identity(root)
    if not config['fixture'] and runtime['frozen_full_runtime']!=FROZEN_FULL:raise ValueError('Frozen historical runtime changed')
    if canonical_hash(runtime)!=config['runtime_sha256']:raise ValueError('Domain SFT runtime changed')
    training=SFTCorpus(config['data_manifest'],'train');validation=SFTCorpus(config['data_manifest'],'validation')
    if training.fixture!=config['fixture']:raise ValueError('Fixture/production mismatch')
    if training.identity['manifest_sha256']!=config['data_manifest_sha256']:raise ValueError('Published dataset changed')
    if sha256_file(config['base_checkpoint'])!=config['base_checkpoint_sha256']:raise ValueError('Selected Full-SFT start changed')
    if device=='cuda':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 unavailable')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.backends.cuda.matmul.allow_tf32=False
    random.seed(train['seed']);np.random.seed(train['seed']);torch.manual_seed(train['seed'])
    if device=='cuda':torch.cuda.manual_seed_all(train['seed'])
    torch.use_deterministic_algorithms(train['deterministic'])
    guard=BudgetGuard(config['budget'],str(out.resolve()),config['budget']['ledger_path'])
    sys.path.insert(0,str(root/'scripts'))
    from chat_sft import load_model
    model,tokenizer,start_identity=load_model(config['base_checkpoint'],device,threads=8 if device=='cuda' else 1)
    if start_identity['stage']!='sft' or start_identity['tokenizer_sha256']!=training.identity['tokenizer_sha256']:
        raise ValueError('Start must be the selected matching full-SFT model')
    if not config['fixture'] and model.parameter_count()!=303350784:raise ValueError('Wrong production model')
    model.config.activation_checkpointing=train['activation_checkpointing'];model.requires_grad_(True)
    model_config=asdict(model.config)
    updates=deterministic_epoch_batches(len(training),train['micro_batch_size'],train['gradient_accumulation_steps'],train['seed'],0)
    total=len(updates);warmup=math.ceil(total*train['warmup_ratio'])
    plan={'epochs':1,'stop_epochs':None,'train_examples':len(training),'max_steps':total,'warmup_steps':warmup,
          'effective_batch_size':train['micro_batch_size']*train['gradient_accumulation_steps']}
    identity={'config':canonical_hash(config),'source':canonical_hash(runtime),'manifest_sha256':training.identity['manifest_sha256'],
        'data_identity':training.identity['data_identity'],'tokenizer_sha256':training.identity['tokenizer_sha256'],
        'base_checkpoint_sha256':start_identity['checkpoint_sha256'],'environment':_environment(device)}
    optimizer=torch.optim.AdamW(model.parameters(),lr=train['learning_rate'],betas=tuple(train['betas']),weight_decay=train['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(step,warmup,total,train['min_lr_ratio']))
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    step=0;examples=0;best_loss=float('inf');best_step=0;latest=None;previous=None
    if resume:
        previous=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
        if previous.get('identity')!=identity or previous.get('run_id')!=str(out.resolve()) or previous.get('checkpoint_kind')!='full':
            raise ValueError('Resume identity mismatch')
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['scaler']);step=previous['step'];examples=previous['examples_seen']
        best_loss=previous['best_validation_loss'];best_step=previous['best_step'];latest=previous['latest_validation']
        if not 0<=step<=total:raise ValueError('Resume cursor outside original one-epoch plan')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'config.json',config)
    write_json(out/'run_manifest.json',{'schema_version':1,'stage':'sft','method':'full','domain':config.get('experiment',{}).get('domain','broad_math'),
        'initialization':'selected completed Full-SFT model','identity':identity,'runtime':runtime,'plan':plan,
        'parameters':model.parameter_count(),'initial_model':start_identity,'test_used_for_selection':False})
    tracker=None;writer=None;stop_requested=False;handlers={};status='failed'

    def stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True

    def persist(kind):
        incoming=sum(p.numel()*p.element_size() for p in model.parameters())*(3 if kind=='full' else 1)+16*2**20
        if shutil.disk_usage(out).free<config['min_free_bytes']+incoming:raise OSError('Checkpoint would violate reserved disk space')
        payload={'schema_version':1,'stage':'sft','method':'full','checkpoint_kind':kind,'config':config,
            'model_config':model_config,'identity':identity,'run_id':str(out.resolve()),'model':model.state_dict(),
            'step':step,'examples_seen':examples,'best_step':best_step,'best_validation_loss':best_loss,
            'latest_validation':latest,'plan':plan,'initialization':'completed Full-SFT model'}
        if kind=='full':payload.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=_rng(device))
        save_checkpoint(out/('last.pt' if kind=='full' else 'best.pt'),payload)

    try:
        tracker=start_tracker(train,out,root,job_type='domain-sft-full',run_config={'train':train,'plan':plan},resume=resume is not None)
        if train['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(str(out/'tensorboard'),purge_step=step+1 if resume else None)
        if previous:_restore_rng(previous['rng'],device)
        del previous
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,stop)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        with (out/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            def record(value):
                row={'optimizer_step':step,'step':step,'examples_seen':examples,'epochs_equivalent':examples/len(training),
                     'elapsed_seconds':time.monotonic()-started,**value}
                stream.write(__import__('json').dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                tracker.log(row,step);_console_json(row)
                if writer:
                    for key in ['train_loss','learning_rate','grad_norm']:
                        if key in value:writer.add_scalar(key,value[key],step)
                    if 'validation' in value:writer.add_scalar('validation/macro_loss',value['validation']['macro_loss'],step)
            if latest is None:
                latest=evaluate_sft_model(model,validation,config);best_loss=latest['macro_loss'];persist('model_only')
                record({'event':'baseline','validation':latest})
            status='completed'
            while step<total:
                if stop_requested or (out/'STOP').exists():status='user_stop';break
                if stop_after_steps is not None and step>=stop_after_steps:status='bounded_smoke_stop';break
                if guard.should_stop():status='budget_or_time_limit';break
                start=time.monotonic();lr=optimizer.param_groups[0]['lr']
                result=train_update(model,optimizer,scaler,training,updates[step],config)
                if not result['updated']:raise FloatingPointError('Optimizer skipped update')
                scheduler.step();step+=1;examples+=result['examples']
                evaluated=step%train['eval_interval']==0 or step==total
                if evaluated:
                    latest=evaluate_sft_model(model,validation,config)
                    if latest['macro_loss']<best_loss:best_loss=latest['macro_loss'];best_step=step;persist('model_only')
                if step==1 or step%train['log_interval']==0 or evaluated:
                    record({'event':'train','train_loss':result['loss'],'learning_rate':lr,'grad_norm':result['grad_norm'],
                        'input_tokens_per_second':result['input_tokens']/(time.monotonic()-start),
                        **({'validation':latest} if evaluated else {})})
                if step%train['save_interval']==0:persist('full')
            persist('full')
            summary={'status':status,'step':step,'examples_seen':examples,'plan':plan,'best_step':best_step,
                'best_validation_loss':best_loss,'latest_validation':latest,'budget':guard.update(),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None,
                'best_checkpoint':str(out/'best.pt'),'last_checkpoint':str(out/'last.pt')}
            write_json(out/'summary.json',summary);_console_json(summary)
        return summary
    except BaseException as error:
        status='failed'
        write_json(out/'failure.json',{'status':status,'error_type':type(error).__name__,'last_finished_step':step,
            'resumable_checkpoint':str(out/'last.pt') if (out/'last.pt').is_file() else None})
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        guard.update()
        if writer:writer.close()
        if tracker:tracker.finish(0 if status in {'completed','user_stop','bounded_smoke_stop'} else 1)
