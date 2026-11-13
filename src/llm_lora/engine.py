"""Adapter-only更新与完整断点恢复；数据/梯度归一化复用冻结SFT实现。"""
from __future__ import annotations

import copy
from dataclasses import asdict
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import threading
import time

import numpy as np
import torch

from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_lab.common.tracking import start_tracker
from llm_lab.config import ModelConfig
from llm_lab.engine.checkpoint import save_checkpoint
from llm_lab.engine.corpus import BinaryCorpus
from llm_lab.engine.pretrain import lr_factor,source_fingerprint
from llm_lab.models import Decoder
from llm_sft.config import plan_details
from llm_sft.data import SFTCorpus
from llm_sft.engine import train_update,evaluate_sft_model,evaluate_retention,update_plan,_rng,_restore_rng,_environment,_console_json
from .adapters import inject,assert_frozen,frozen_digest,adapter_state,load_adapter,export_best
from .config import validate,runtime_fingerprint,runtime_identity,FROZEN_FULL_RUNTIME
from .data import development,protocol


def load_base(config):
    path=Path(config['base_checkpoint']);before=path.stat();digest=sha256_file(path)
    if config.get('base_checkpoint_sha256') and config['base_checkpoint_sha256']!=digest:raise ValueError('Base checksum changed')
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('stage') not in {None,'pretrain'} or payload.get('schema_version')!=1:raise ValueError('LoRA initializes from pretrained Base only')
    spec=ModelConfig(**payload['config']['model']);spec.activation_checkpointing=config['train']['activation_checkpointing']
    model=Decoder(spec);model.load_state_dict(payload['model'],strict=True)
    original_manifest=Path(payload['config']['data_manifest'])
    if not original_manifest.is_absolute():original_manifest=Path(payload['config'].get('project_root',config['project_root']))/original_manifest
    if sha256_file(original_manifest)!=payload['identity']['manifest_sha256']:raise ValueError('Base data identity changed')
    if not config['fixture'] and payload['identity'].get('source')!=source_fingerprint():raise ValueError('Pretraining implementation changed')
    metadata={'path':str(path),'sha256':digest,'step':payload.get('step'),'tokenizer_sha256':payload['identity']['tokenizer_sha256'],
              'manifest':str(original_manifest),'model_config':asdict(spec)}
    del payload
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Base changed during load')
    return model,spec,metadata


def run_lora(config,resume=None,stop_after_steps=None):
    validate(config);config=copy.deepcopy(config);train=config['train'];device=train['device'];output=Path(config['output_dir'])
    if resume is None and output.exists() and any(output.iterdir()):raise ValueError('Output nonempty; resume its own last or choose new run')
    if resume and Path(resume).resolve()!=(output/'last.pt').resolve():raise ValueError('Only this run last.pt may resume')
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps<1):raise ValueError('Pause step must be a positive absolute update')
    source=runtime_fingerprint()
    if config.get('runtime_sha256') and source!=config['runtime_sha256']:raise ValueError('LoRA source/dependency identity changed')
    if not config['fixture'] and runtime_identity()['full_sft_runtime_sha256']!=FROZEN_FULL_RUNTIME:raise ValueError('Frozen Full-SFT source changed')
    if device=='cuda':
        if not torch.cuda.is_available() or train['dtype']=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA/BF16 unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
    training=SFTCorpus(config['data_manifest'],config['train_split']);validation=SFTCorpus(config['data_manifest'],'validation')
    if training.fixture!=config['fixture']:raise ValueError('Fixture/production data mismatch')
    if config.get('data_manifest_sha256') and config['data_manifest_sha256']!=training.identity['manifest_sha256']:raise ValueError('SFT manifest changed')
    if config.get('data_identity') and config['data_identity']!=training.identity['data_identity']:raise ValueError('Training data identity changed')
    dev=development(config);evaluation_protocol=protocol(config)
    model,spec,base=load_base(config)
    if base['tokenizer_sha256']!=training.identity['tokenizer_sha256']:raise ValueError('Base/tokenizer mismatch')
    if max(len(row['input_ids'])-1 for corpus in (training,validation) for row in corpus.rows)>spec.max_seq_len:raise ValueError('Data exceeds Base context')
    if not config['fixture'] and model.parameter_count()!=303350784:raise ValueError('Wrong production Base size')
    random.seed(train['seed']);np.random.seed(train['seed']);torch.manual_seed(train['seed'])
    if device=='cuda':torch.cuda.manual_seed_all(train['seed'])
    model,adapter=inject(model,config['adapter']);base_weights_sha256=frozen_digest(model)
    model.to(device)
    torch.use_deterministic_algorithms(train['deterministic'])
    retention=BinaryCorpus(base['manifest'],'validation',train['retention_eval_seq_len']) if train['retention_eval_batches'] else None
    updates=update_plan(config,len(training));plan=plan_details(config,len(training))
    if len(updates)!=plan['max_steps']:raise ValueError('LoRA plan mismatch')
    environment={**_environment(device),'lora_runtime':runtime_identity()}
    identity_config={key:value for key,value in config.items() if key not in {'budget','output_dir','project_root'}}
    identity={'config':canonical_hash(identity_config),'base_checkpoint_sha256':base['sha256'],
        'base_frozen_weights_sha256':base_weights_sha256,'tokenizer_sha256':training.identity['tokenizer_sha256'],
        'train_data':training.identity,'validation_data':validation.identity,'development_data':dev.identity if dev else None,
        'evaluation_protocol_sha256':config.get('evaluation_protocol_sha256'),'source':source,'environment':environment,
        'adapter':adapter,'retention_data':retention.fingerprint if retention else None}
    previous=torch.load(resume,map_location='cpu',weights_only=False,mmap=True) if resume else None
    if previous:
        if previous.get('stage')!='sft' or previous.get('method')!='lora' or previous.get('checkpoint_kind')!='adapter_full':raise ValueError('Only complete Adapter last may resume')
        if previous['run_id']!=str(output.resolve()) or previous['identity']!=identity:raise ValueError('LoRA resume identity mismatch')
    parameters=[value for value in model.parameters() if value.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=train['learning_rate'],betas=tuple(train['betas']),weight_decay=train['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(step,plan['warmup_steps'],plan['max_steps'],train['min_lr_ratio']))
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and train['dtype']=='fp16')
    progress={'next_update':0,'examples_seen':0,'input_tokens_seen':0,'supervised_tokens_seen':0}
    step=0;best_loss=None;best_step=0;baseline=None;latest=None;latest_retention=None;latest_development=None
    if previous:
        load_adapter(model,previous['adapter_state']);optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler']);scaler.load_state_dict(previous['scaler'])
        step=previous['step'];progress=copy.deepcopy(previous['progress'])
        if progress['next_update']!=step or not 0<=step<=plan['stop_step']:raise ValueError('Resume cursor invalid')
        best_loss=previous['best_validation_loss'];best_step=previous['best_step'];baseline=previous['baseline_validation']
        latest=previous['latest_validation'];latest_retention=previous['latest_retention'];latest_development=previous.get('latest_development')
    output.mkdir(parents=True,exist_ok=True);run_id=str(output.resolve());guard=BudgetGuard(config['budget'],run_id,config['budget']['ledger_path'])
    write_json(output/'config.json',config);write_json(output/'environment.json',environment)
    write_json(output/'run_manifest.json',{'schema_version':1,'stage':'sft','method':'lora','run_id':run_id,
        'identity':identity,'base':base,'adapter':adapter,'plan':plan,'loss':'assistant/EOS sum normalized by update supervised tokens'})
    def payload(kind):
        value={'schema_version':1,'stage':'sft','method':'lora','checkpoint_kind':kind,'run_id':run_id,'config':config,
            'model_config':asdict(spec),'identity':identity,'base':base,'adapter':adapter,'adapter_state':adapter_state(model),
            'step':step,'progress':copy.deepcopy(progress),'best_validation_loss':best_loss,'best_step':best_step,
            'baseline_validation':baseline,'latest_validation':latest,'latest_retention':latest_retention,'latest_development':latest_development}
        if kind=='adapter_full':value.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=_rng(device))
        return value
    def space():
        if device=='cuda':
            incoming=adapter['trainable_parameters']*16+16*1024**2
            if shutil.disk_usage(output).free<config['min_free_bytes']+incoming:raise OSError('Adapter save would violate disk reserve')
    def save_last():space();save_checkpoint(output/'last.pt',payload('adapter_full'))
    def save_best():
        space();save_checkpoint(output/'best.pt',payload('adapter_only'))
        export_best(output,model,{'kind':'custom_lora_adapter','adapter':config['adapter'],'targets':adapter['targets'],
            'base_checkpoint':base['path'],'base_checkpoint_sha256':base['sha256'],'tokenizer_sha256':base['tokenizer_sha256'],
            'data_identity':training.identity['data_identity'],'runtime_sha256':source})
    stopped=False;old_handlers={};tracker=None;writer=None;status='failed';started=time.monotonic()
    initial_input=progress['input_tokens_seen'];initial_supervised=progress['supervised_tokens_seen']
    def request_stop(signum,frame):
        nonlocal stopped
        stopped=True
    try:
        if train['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(str(output/'tensorboard'),purge_step=step+1 if previous else None)
        tracker=start_tracker(train,output,Path(config['project_root']),job_type='sft-lora',
            run_config={'model':asdict(spec),'adapter':config['adapter'],'train':train,'plan':plan},resume=previous is not None)
        if previous:_restore_rng(previous['rng'],device)
        del previous
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM,signal.SIGINT):old_handlers[signum]=signal.signal(signum,request_stop)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            def record(value):
                value={'step':step,**value,'examples_seen':progress['examples_seen'],'input_tokens_seen':progress['input_tokens_seen'],
                    'supervised_tokens_seen':progress['supervised_tokens_seen'],'epochs_equivalent':progress['examples_seen']/len(training),
                    'elapsed_seconds_this_session':time.monotonic()-started,'budget':guard.update()}
                stream.write(__import__('json').dumps(value,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();tracker.log(value,step)
                if writer:
                    for key in ('train_loss','learning_rate','grad_norm_before_clip','input_tokens_per_second','supervised_tokens_per_second'):
                        if key in value:writer.add_scalar(key,value[key],step)
                    for key in ('validation','retention_validation','development_validation'):
                        if value.get(key):writer.add_scalar(key+'/macro_loss',value[key]['macro_loss'],step)
                _console_json(value)
            if baseline is None:
                latest=evaluate_sft_model(model,validation,config);latest_retention=evaluate_retention(model,retention,config)
                baseline={'sft':latest,'retention':latest_retention};best_loss=latest['macro_loss'];save_best()
                record({'event':'baseline','validation':latest,'retention_validation':latest_retention})
            status='completed';overflows=0
            while step<plan['stop_step']:
                if stopped:status='signal_stop';break
                if guard.should_stop():status='budget_or_time_limit';break
                if stop_after_steps is not None and step>=stop_after_steps:status='paused';break
                current=updates[step];rate=optimizer.param_groups[0]['lr']
                result=train_update(model,optimizer,scaler,training,current['microbatches'],config)
                assert_frozen(model)
                if not result['updated']:
                    overflows+=1
                    if overflows>32:raise FloatingPointError('Repeated FP16 overflow')
                    continue
                overflows=0;scheduler.step();step+=1
                progress['next_update']=step;progress['examples_seen']+=result['examples']
                progress['input_tokens_seen']+=result['input_tokens'];progress['supervised_tokens_seen']+=result['supervised_tokens']
                evaluated=step%train['eval_interval']==0 or step==plan['stop_step'] or step==stop_after_steps
                if evaluated:
                    latest=evaluate_sft_model(model,validation,config);latest_retention=evaluate_retention(model,retention,config)
                    if latest['macro_loss']<best_loss:best_loss=latest['macro_loss'];best_step=step;save_best()
                if step%train['log_interval']==0 or evaluated:
                    if device=='cuda':torch.cuda.synchronize()
                    seconds=max(time.monotonic()-started,1e-9)
                    value={'train_loss':result['loss'],'learning_rate':rate,'grad_norm_before_clip':result['grad_norm'],
                        'input_tokens_per_second':(progress['input_tokens_seen']-initial_input)/seconds,
                        'supervised_tokens_per_second':(progress['supervised_tokens_seen']-initial_supervised)/seconds,
                        'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
                    if evaluated:value.update(validation=latest,retention_validation=latest_retention)
                    record(value)
                if step%train['save_interval']==0 or step==plan['stop_step']:save_last()
            if frozen_digest(model)!=base_weights_sha256:raise ValueError('Frozen Base tensor checksum changed')
            if dev is not None:
                latest_development=evaluate_sft_model(model,dev,config);record({'event':'development','development_validation':latest_development})
            save_last()
    finally:
        failed=sys.exc_info()[0] is not None
        if tracker is not None:tracker.finish(exit_code=int(failed))
        if writer is not None:writer.close()
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        summary={'status':'failed' if failed else status,'step':step,'plan':plan,'progress':progress,
            'best_step':best_step,'best_validation_loss':best_loss,'latest_validation':latest,
            'latest_retention':latest_retention,'latest_development':latest_development,
            'adapter':adapter,'base_frozen_weights_sha256':base_weights_sha256,'base_frozen_unchanged':not failed,
            'budget':guard.update(),'last_checkpoint':str(output/'last.pt'),'best_checkpoint':str(output/'best.pt')}
        write_json(output/'summary.json',summary)
    return summary
