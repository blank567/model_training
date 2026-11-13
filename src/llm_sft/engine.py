"""回答 Token 加权的全参数 SFT，按完整参数更新保存与恢复。

数据模块已完成一次标签错位：[B,T] x 直接预测同形状 y；提示/PAD=-100。
同一次更新的各 microbatch 按总监督 Token 归一化，避免短回答被过度加权。
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict
from importlib import metadata
import errno
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import sys
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_lab.common.tracking import start_tracker
from llm_lab.config import ModelConfig
from llm_lab.engine.checkpoint import save_checkpoint
from llm_lab.engine.pretrain import lr_factor,optimizer_update,source_fingerprint
from llm_lab.models import Decoder
from .config import epoch_sample_counts,plan_details,runtime_fingerprint,validate_config
from .data import SFTCorpus,deterministic_epoch_batches


def _console_json(value):
    """控制台关闭不影响训练状态保存；JSONL、TensorBoard 和 W&B 继续工作。

    Ctrl+C 可能先结束前台 tee。若 stdout 的管道已断开，解除该控制台
    输出；只有 stderr 指向同一个已断开的管道时，才同步解除 stderr。
    """
    try:
        print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
    except OSError as error:
        if not isinstance(error,BrokenPipeError) and error.errno!=errno.EPIPE:
            raise
        try:
            descriptor=sys.stdout.fileno()
            pipe_stat=os.fstat(descriptor)
        except (AttributeError,OSError,ValueError):
            # 无文件描述符的嵌入式/测试输出流，不修改宿主进程的真实stdout。
            sys.stdout=io.StringIO()
            return
        descriptors=[descriptor]
        try:
            stderr_descriptor=sys.stderr.fileno()
            stderr_stat=os.fstat(stderr_descriptor)
            if (pipe_stat.st_dev,pipe_stat.st_ino)==(stderr_stat.st_dev,stderr_stat.st_ino):
                descriptors.append(stderr_descriptor)
        except (AttributeError,OSError,ValueError):
            pass
        sink=os.open(os.devnull,os.O_WRONLY)
        try:
            for target in set(descriptors):os.dup2(sink,target)
        finally:os.close(sink)


def autocast(config):
    train=config['train']
    if train['dtype']=='fp32':return contextlib.nullcontext()
    return torch.autocast(train['device'],dtype=torch.bfloat16 if train['dtype']=='bf16' else torch.float16)


def _rng(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if device=='cuda' else None}


def _restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if device=='cuda':torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _environment(device):
    packages={}
    for name in ('torch','numpy','tokenizers','tensorboard','wandb'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    return {'python':platform.python_version(),'packages':packages,'torch_cuda':torch.version.cuda,
            'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None}


def update_plan(config,examples):
    """逐轮洗牌；小数轮只取该轮排列前缀，末尾不足一个更新组仍保留。"""
    train=config['train'];result=[]
    for epoch,count in enumerate(epoch_sample_counts(examples,train['epochs'])):
        groups=deterministic_epoch_batches(examples,train['micro_batch_size'],train['gradient_accumulation_steps'],train['seed'],epoch)
        if count<examples:
            order=[index for group in groups for micro in group for index in micro][:count]
            micros=[order[i:i+train['micro_batch_size']] for i in range(0,count,train['micro_batch_size'])]
            groups=[micros[i:i+train['gradient_accumulation_steps']] for i in range(0,len(micros),train['gradient_accumulation_steps'])]
        result.extend({'epoch':epoch,'microbatches':group} for group in groups)
    return result


def train_update(model,optimizer,scaler,corpus,group,config):
    """一次optimizer更新；返回真实监督量，不用padding矩阵面积替代Token数。"""
    model.train();optimizer.zero_grad(set_to_none=True)
    denominator=sum(corpus.rows[index]['answer_token_count'] for micro in group for index in micro)
    if denominator<=0:raise ValueError('SFT update has no supervised tokens')
    loss_sum=0.;input_tokens=0;examples=0;actual=0
    for indices in group:
        batch=corpus.collate(indices,device=config['train']['device'])
        with autocast(config):
            logits=model(batch['x']).logits
            loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),batch['y'].reshape(-1),ignore_index=-100,reduction='sum')
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite SFT loss')
        scaler.scale(loss/denominator).backward()
        loss_sum+=float(loss.detach());input_tokens+=batch['input_tokens'];examples+=len(indices);actual+=batch['supervised_tokens']
    if actual!=denominator:raise ValueError('SFT supervised token accounting mismatch')
    updated,norm=optimizer_update(model,optimizer,scaler,config['train']['max_grad_norm'])
    return {'loss':loss_sum/denominator,'supervised_tokens':denominator,'input_tokens':input_tokens,
            'examples':examples,'grad_norm':norm,'updated':updated}


@torch.no_grad()
def evaluate_sft_model(model,corpus,config):
    was_training=model.training;model.eval();result={};all_loss=0.;all_tokens=0
    batch_size=config['train'].get('eval_micro_batch_size',8)
    try:
        for language in ('en','zh'):
            total=0.;tokens=0;indices=corpus.language_indices[language]
            if not indices:raise ValueError('Validation requires both languages')
            for offset in range(0,len(indices),batch_size):
                batch=corpus.collate(indices[offset:offset+batch_size],device=config['train']['device'])
                with autocast(config):
                    logits=model(batch['x']).logits
                    value=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),batch['y'].reshape(-1),ignore_index=-100,reduction='sum')
                if not torch.isfinite(value):raise FloatingPointError('Non-finite SFT validation loss')
                total+=float(value);tokens+=batch['supervised_tokens']
            average=total/tokens
            result[language]={'loss':average,'perplexity':math.exp(min(average,80.)),
                              'perplexity_capped':average>80.,'supervised_tokens':tokens,'examples':len(indices)}
            all_loss+=total;all_tokens+=tokens
        result['macro_loss']=(result['en']['loss']+result['zh']['loss'])/2
        result['token_weighted_loss']=all_loss/all_tokens
        return result
    finally:model.train(was_training)


@torch.no_grad()
def evaluate_retention(model,corpus,config):
    if corpus is None:return None
    train=config['train'];was_training=model.training;model.eval();result={}
    batches=train.get('retention_eval_batches',2);batch_size=train.get('eval_micro_batch_size',8)
    try:
        for number,language in enumerate(('en','zh')):
            sampler=np.random.default_rng(train['seed']+37000+number)
            total=0.;tokens=0
            for _ in range(batches):
                x,y=corpus.batch(language,batch_size,sampler)
                with autocast(config):loss=model(x.to(train['device']),y.to(train['device'])).loss
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite retention loss')
                total+=float(loss)*y.numel();tokens+=y.numel()
            value=total/tokens
            result[language]={'loss':value,'perplexity':math.exp(min(value,80.)),'tokens':tokens}
        result['macro_loss']=(result['en']['loss']+result['zh']['loss'])/2
        result['sampling']={'split':'validation','seq_len':corpus.seq_len,'batches':batches,'batch_size':batch_size,
                            'seed_base':train['seed']+37000}
        return result
    finally:model.train(was_training)


def _read_checkpoint(path):
    # 只读取本项目生成的可信checkpoint；mmap避免读取不使用的Base AdamW大张量。
    value=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if value.get('schema_version')!=1:raise ValueError('Unsupported checkpoint schema')
    return value


def _save_space(config,model,output):
    if config['train']['device']!='cuda':return
    # 下一次原子写完整恢复状态，旧last/best仍存在；水位之外预留新临时文件。
    tensor_bytes=sum(parameter.numel()*parameter.element_size() for parameter in model.parameters())
    incoming=3*tensor_bytes+16*1024**2
    if shutil.disk_usage(output).free<config['min_free_bytes']+incoming:
        raise OSError('SFT checkpoint would violate reserved disk space')


def run_sft(config:dict,resume=None,stop_after_steps=None):
    validate_config(config);config=copy.deepcopy(config);train=config['train'];device=train['device']
    output=Path(config['output_dir']);base_path=Path(config['base_checkpoint'])
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps<1):
        raise ValueError('stop_after_steps must be a positive absolute optimizer step')
    if resume is None and output.exists() and any(output.iterdir()):raise ValueError('SFT output is not empty')
    if resume is not None and Path(resume).resolve()!=(output/'last.pt').resolve():
        raise ValueError('Resume only accepts this run last.pt')
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if train['dtype']=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unsupported')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.backends.cuda.matmul.allow_tf32=False
    training=SFTCorpus(config['data_manifest'],config['train_split'])
    validation=SFTCorpus(config['data_manifest'],'validation')
    if training.fixture!=config['fixture'] or validation.fixture!=config['fixture']:
        raise ValueError('Fixture/production config and dataset mismatch')
    base_stat=base_path.stat();base_sha=sha256_file(base_path)
    if config.get('base_checkpoint_sha256') and config['base_checkpoint_sha256']!=base_sha:
        raise ValueError('Frozen Base checkpoint SHA mismatch')
    if config.get('data_manifest_sha256') and config['data_manifest_sha256']!=training.identity['manifest_sha256']:
        raise ValueError('Frozen SFT manifest SHA mismatch')
    if config.get('data_identity') and config['data_identity']!=training.identity['data_identity']:
        raise ValueError('Frozen SFT data identity mismatch')
    source=runtime_fingerprint()
    if config.get('runtime_sha256') and config['runtime_sha256']!=source:raise ValueError('SFT runtime source changed')
    base=_read_checkpoint(base_path)
    if base.get('stage') not in {None,'pretrain'} or 'model' not in base or 'model' not in base['config']:
        raise ValueError('SFT must initialize from the selected pretrained Base')
    if base['identity']['tokenizer_sha256']!=training.identity['tokenizer_sha256']:
        raise ValueError('Base/SFT tokenizer mismatch')
    if not config['fixture'] and base['identity'].get('source')!=source_fingerprint():
        raise ValueError('Pretraining model implementation changed')
    model_config=ModelConfig(**base['config']['model'])
    model_config.activation_checkpointing=train.get('activation_checkpointing',False)
    model_config.validate()
    actual_length=max(len(row['input_ids'])-1 for corpus in (training,validation) for row in corpus.rows)
    if model_config.vocab_size!=training.vocab_size or actual_length>model_config.max_seq_len:
        raise ValueError('SFT data incompatible with Base context/vocabulary')
    retention=None
    if train.get('retention_eval_batches',0)>0:
        from llm_lab.engine.corpus import BinaryCorpus
        old_manifest=Path(base['config']['data_manifest'])
        if not old_manifest.is_absolute():old_manifest=Path(base['config'].get('project_root',config['project_root']))/old_manifest
        if sha256_file(old_manifest)!=base['identity']['manifest_sha256']:raise ValueError('Base retention manifest changed')
        retention=BinaryCorpus(old_manifest,'validation',train.get('retention_eval_seq_len',512))
        if retention.seq_len>model_config.max_seq_len:raise ValueError('Retention length exceeds Base context')
    plan=plan_details(config,len(training));updates=update_plan(config,len(training))
    if len(updates)!=plan['max_steps']:raise ValueError('SFT schedule plan mismatch')
    environment=_environment(device)
    identity_config=copy.deepcopy(config)
    for name in ('budget','output_dir','project_root'):identity_config.pop(name,None)
    identity={'config':canonical_hash(identity_config),'base_checkpoint_sha256':base_sha,
              'manifest_sha256':training.identity['manifest_sha256'],'tokenizer_sha256':training.identity['tokenizer_sha256'],
              'data_identity':training.identity['data_identity'],'train_data':training.identity,'validation_data':validation.identity,
              'source':source,'environment':environment,'retention_data':retention.fingerprint if retention else None}
    previous=_read_checkpoint(Path(resume)) if resume else None
    if previous:
        if previous.get('stage')!='sft' or previous.get('checkpoint_kind')!='full':
            raise ValueError('Only a full SFT last.pt can resume; best.pt is inference-only')
        if previous.get('run_id')!=str(output.resolve()):
            raise ValueError('SFT resume checkpoint belongs to another run')
        if previous['identity']!=identity:raise ValueError('SFT resume identity mismatch: config/data/code/Base/environment changed')
    output.mkdir(parents=True,exist_ok=True)
    random.seed(train['seed']);np.random.seed(train['seed']);torch.manual_seed(train['seed'])
    if device=='cuda':torch.cuda.manual_seed_all(train['seed'])
    torch.use_deterministic_algorithms(train['deterministic'])
    if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=False
    run_id=str(output.resolve());guard=BudgetGuard(config['budget'],run_id,config['budget']['ledger_path'])
    _console_json({'phase':'loading_base','base_step':base.get('step'),'plan':plan})
    model=Decoder(model_config)
    model.load_state_dict(base['model'])
    base_metadata={'path':str(base_path),'sha256':base_sha,'step':base.get('step'),'tokens_seen':base.get('tokens_seen')}
    del base
    after=base_path.stat()
    if (base_stat.st_ino,base_stat.st_size,base_stat.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('Base checkpoint changed while loading')
    model.to(device)
    if not config['fixture'] and model.parameter_count()!=303350784:raise ValueError('Unexpected production parameter count')
    optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':train['weight_decay']},
                                {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}],
                               lr=train['learning_rate'],betas=tuple(train['betas']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(step,plan['warmup_steps'],plan['max_steps'],train['min_lr_ratio']))
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and train['dtype']=='fp16')
    progress={'next_update':0,'examples_seen':0,'input_tokens_seen':0,'supervised_tokens_seen':0}
    step=0;best_loss=None;best_step=0;baseline=None;latest=None;latest_retention=None
    if previous:
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler']);scaler.load_state_dict(previous['scaler'])
        step=previous['step'];progress=copy.deepcopy(previous['progress'])
        if progress['next_update']!=step or not 0<=step<=plan['stop_step']:raise ValueError('Invalid resume cursor')
        best_loss=previous['best_validation_loss'];best_step=previous['best_step'];baseline=previous['baseline_validation']
        latest=previous.get('latest_validation');latest_retention=previous.get('latest_retention')
    writer=None;tracker=None;status='failed';stop_requested=False;stop_signal=None;old_handlers={}
    started=time.monotonic();initial_tokens=progress['input_tokens_seen'];initial_supervised=progress['supervised_tokens_seen']
    write_json(output/'config.json',config);write_json(output/'environment.json',environment)
    write_json(output/'run_manifest.json',{'schema_version':1,'stage':'sft','method':'full','run_id':run_id,
                                         'identity':identity,'base':base_metadata,'parameters':model.parameter_count(),
                                         'plan':plan,'sampling':'deterministic shuffled epochs without replacement',
                                         'loss':'sum assistant/EOS cross entropy divided by update supervised tokens'})
    def payload(kind):
        common={'schema_version':1,'stage':'sft','checkpoint_kind':kind,'config':config,'model_config':asdict(model_config),
                'identity':identity,'run_id':run_id,'environment':environment,'base':base_metadata,'model':model.state_dict(),
                'step':step,'progress':copy.deepcopy(progress),'best_validation_loss':best_loss,'best_step':best_step,
                'baseline_validation':baseline,'latest_validation':latest,'latest_retention':latest_retention}
        if kind=='full':common.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=_rng(device))
        return common
    def persist_full():
        _save_space(config,model,output);save_checkpoint(output/'last.pt',payload('full'))
    def persist_best():
        _save_space(config,model,output);save_checkpoint(output/'best.pt',payload('model_only'))
    def request_stop(signum,frame):
        # 信号处理器只记状态，避免在stdout正在写入时重入输出或执行checkpoint I/O。
        nonlocal stop_requested,stop_signal
        stop_requested=True;stop_signal=signum
    try:
        if train['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(str(output/'tensorboard'),purge_step=step+1 if previous else None)
        tracker=start_tracker(train,output,Path(config['project_root']),job_type='sft-full',
                              run_config={'model':asdict(model_config),'train':train,'plan':plan},resume=previous is not None)
        if previous:_restore_rng(previous['rng'],device)
        del previous
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT,signal.SIGTERM):old_handlers[signum]=signal.signal(signum,request_stop)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as metrics:
            def record(value):
                value={'step':step,**value,'examples_seen':progress['examples_seen'],
                       'input_tokens_seen':progress['input_tokens_seen'],'supervised_tokens_seen':progress['supervised_tokens_seen'],
                       'epochs_equivalent':progress['examples_seen']/len(training),'elapsed_seconds_this_session':time.monotonic()-started,
                       'budget':guard.update()}
                metrics.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');metrics.flush();tracker.log(value,step)
                if writer:
                    for key in ('train_loss','learning_rate','grad_norm_before_clip','input_tokens_per_second','supervised_tokens_per_second'):
                        if key in value:writer.add_scalar(key,value[key],step)
                    if 'validation' in value:
                        for language in ('en','zh'):writer.add_scalar('validation/'+language+'_answer_loss',value['validation'][language]['loss'],step)
                    if value.get('retention_validation'):
                        for language in ('en','zh'):writer.add_scalar('retention/'+language+'_lm_loss',value['retention_validation'][language]['loss'],step)
                _console_json(value)
            if baseline is None:
                _console_json({'phase':'baseline_validation'})
                latest=evaluate_sft_model(model,validation,config);latest_retention=evaluate_retention(model,retention,config)
                baseline={'sft':latest,'retention':latest_retention};best_loss=latest['macro_loss'];best_step=0
                persist_best();record({'event':'baseline','validation':latest,'retention_validation':latest_retention})
            status='completed';overflows=0
            while step<plan['stop_step']:
                if stop_requested:
                    _console_json({'phase':'stop_requested','signal':stop_signal})
                    status='signal_stop';break
                if guard.should_stop():status='budget_or_time_limit';break
                if stop_after_steps is not None and step>=stop_after_steps:status='simulated_interruption';break
                current=updates[step];learning_rate=optimizer.param_groups[0]['lr']
                result=train_update(model,optimizer,scaler,training,current['microbatches'],config)
                if not result['updated']:
                    overflows+=1
                    if overflows>32:raise FloatingPointError('Repeated FP16 overflows')
                    continue
                overflows=0;scheduler.step();step+=1
                progress['next_update']=step;progress['examples_seen']+=result['examples']
                progress['input_tokens_seen']+=result['input_tokens'];progress['supervised_tokens_seen']+=result['supervised_tokens']
                evaluated=step%train['eval_interval']==0 or step==plan['stop_step']
                if evaluated:
                    latest=evaluate_sft_model(model,validation,config);latest_retention=evaluate_retention(model,retention,config)
                    if latest['macro_loss']<best_loss:
                        best_loss=latest['macro_loss'];best_step=step;persist_best()
                if step%train['log_interval']==0 or evaluated:
                    if device=='cuda':torch.cuda.synchronize()
                    seconds=max(time.monotonic()-started,1e-9)
                    value={'train_loss':result['loss'],'learning_rate':learning_rate,'grad_norm_before_clip':result['grad_norm'],
                           'input_tokens_per_second':(progress['input_tokens_seen']-initial_tokens)/seconds,
                           'supervised_tokens_per_second':(progress['supervised_tokens_seen']-initial_supervised)/seconds,
                           'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
                    if evaluated:value.update(validation=latest,retention_validation=latest_retention)
                    record(value)
                if step%train['save_interval']==0 or step==plan['stop_step']:persist_full()
            persist_full()
    finally:
        failed=sys.exc_info()[0] is not None
        if tracker is not None:tracker.finish(exit_code=int(failed))
        if writer is not None:writer.close()
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        budget=guard.update()
        summary={'status':'failed' if failed else status,'step':step,'plan':plan,'progress':progress,
                 'best_validation_loss':best_loss,'best_step':best_step,'baseline_validation':baseline,
                 'latest_validation':latest,'latest_retention':latest_retention,'budget':budget,
                 'parameters':model.parameter_count(),'last_checkpoint':str(output/'last.pt'),'best_checkpoint':str(output/'best.pt')}
        write_json(output/'summary.json',summary)
    return summary
