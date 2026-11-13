"""完整参数DPO/PPO/GRPO：同一SFT参考策略、单轮样本计划、边界保存与精确恢复。"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json,math,os,random,shutil,signal,sys,time
import numpy as np
import torch

from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_lab.common.budget import BudgetGuard
from llm_lab.common.tracking import start_tracker
from llm_lab.engine.checkpoint import save_checkpoint
from llm_lab.engine.pretrain import lr_factor
from llm_sft.data import SFTCorpus
from llm_sft.engine import _rng,_restore_rng,_environment
from llm_reading_independent.comparison import ComparisonRLCorpus
from .config import validate,runtime_identity,FROZEN_FULL
from .data import PreferenceCorpus,order_and_batches,validation_subset
from .generation import generate
from .scoring import token_scores,sequence_scores
from .objectives import dpo_loss,group_advantages,grpo_response_loss,generalized_advantages,ppo_token_loss
from .evaluation import evaluate


def _dpo_update(actor,reference,optimizer,rows,cfg):
    t=cfg['train'];device=t['device'];size=len(rows);micro=max(1,t['score_micro_batch_size']//2)
    optimizer.zero_grad(set_to_none=True);loss_sum=0.;margin_sum=0.
    for start in range(0,size,micro):
        chunk=rows[start:start+micro];p=[r['prefix_ids'] for r in chunk];a=[r['chosen_ids'] for r in chunk]+[r['rejected_ids'] for r in chunk]
        with torch.no_grad():ref=sequence_scores(reference,p+p,a,device)
        current=sequence_scores(actor,p+p,a,device);n=len(chunk)
        losses,margin=dpo_loss(current[:n],current[n:],ref[:n],ref[n:],t['beta'])
        (losses.sum()/size).backward();loss_sum+=float(losses.detach().sum());margin_sum+=float(margin.sum())
    norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),t['max_grad_norm'])
    if not torch.isfinite(norm):raise FloatingPointError('DPO gradient is not finite')
    optimizer.step()
    return {'loss':loss_sum/size,'dpo_margin':margin_sum/size,'grad_norm':float(norm)}


@torch.no_grad()
def _rollout(actor,reference,value_head,rows,cfg,score_reward):
    t=cfg['train'];device=t['device'];group=t['group_size'] if cfg['algorithm']=='grpo' else 1
    expanded=[row for row in rows for _ in range(group)];prompts=[r['prefix_ids'] for r in expanded]
    generated=generate(actor,NoneTokenizer(cfg),prompts,device,max_new_tokens=t['max_new_tokens'],sample=True)
    rewards=[score_reward(row,text) for row,text in zip(expanded,generated['texts'])]
    old=[];ref=[];values=[]
    for start in range(0,len(prompts),t['score_micro_batch_size']):
        p=prompts[start:start+t['score_micro_batch_size']];a=generated['answers'][start:start+t['score_micro_batch_size']]
        lp,v=token_scores(actor,p,a,device,value_head);rp,_=token_scores(reference,p,a,device)
        old.extend(x.detach() for x in lp);ref.extend(x.detach() for x in rp)
        if v is not None:values.extend(x.detach() for x in v)
    return {'rows':expanded,'prompts':prompts,'answers':generated['answers'],'rewards':rewards,
        'old':old,'reference':ref,'old_values':values,'eos_rate':sum(generated['eos'])/len(expanded),
        'generated_tokens':generated['generated_tokens']}


class NoneTokenizer:
    """每个run绑定同一分词器；不在采样时读取任何测试数据。"""
    def __init__(self,cfg):self.tokenizer=cfg['_tokenizer']
    def decode(self,*args,**kwargs):return self.tokenizer.decode(*args,**kwargs)


def _online_update(actor,value_head,optimizer,rollout,cfg,advantages=None,returns=None):
    t=cfg['train'];device=t['device'];count=len(rollout['answers']);micro=t['score_micro_batch_size']
    optimizer.zero_grad(set_to_none=True);loss_total=0.;kl_total=0.;clip_total=0.
    if cfg['algorithm']=='grpo':
        advantage,mixed=group_advantages(rollout['rewards'],t['group_size']);advantage=advantage.to(device)
    else:
        total_tokens=sum(len(a) for a in rollout['answers'])
    for start in range(0,count,micro):
        end=min(start+micro,count);p=rollout['prompts'][start:end];a=rollout['answers'][start:end]
        current,values=token_scores(actor,p,a,device,value_head)
        losses=[]
        for offset,lp in enumerate(current):
            i=start+offset;old=rollout['old'][i];ref=rollout['reference'][i]
            if cfg['algorithm']=='grpo':
                loss,kl,clipped=grpo_response_loss(lp,old,ref,advantage[i],t['clip_range'],t['kl_coefficient'])
                losses.append(loss/count);loss_total+=float(loss.detach())/count;kl_total+=float(kl)/count;clip_total+=float(clipped)/count
            else:
                per_token,policy,value=ppo_token_loss(lp,old,advantages[i],values[offset],rollout['old_values'][i],returns[i],
                    t['clip_range'],t['value_clip'],t['value_coefficient'])
                losses.append(per_token.sum()/total_tokens);loss_total+=float(per_token.detach().sum())/total_tokens
                kl_total+=float((old-ref).sum())/total_tokens;clip_total+=float(((lp-old).exp()-1).abs().gt(t['clip_range']).sum())/total_tokens
        sum(losses).backward()
    parameters=list(actor.parameters())+(list(value_head.parameters()) if value_head is not None else [])
    norm=torch.nn.utils.clip_grad_norm_(parameters,t['max_grad_norm'])
    if not torch.isfinite(norm):raise FloatingPointError('Online policy gradient is not finite')
    optimizer.step()
    return {'loss':loss_total,'kl':kl_total,'clip_fraction':clip_total,'grad_norm':float(norm),
        **({'mixed_reward_groups':int(mixed.sum()),'reward_groups':len(mixed)} if cfg['algorithm']=='grpo' else {})}


def _ppo_targets(rollout,cfg):
    t=cfg['train'];advantages=[];returns=[]
    for old,ref,value,reward in zip(rollout['old'],rollout['reference'],rollout['old_values'],rollout['rewards']):
        shaped=-t['kl_coefficient']*(old-ref);shaped=shaped.clone();shaped[-1]+=reward
        adv,target=generalized_advantages(shaped,value,t['gamma'],t['gae_lambda']);advantages.append(adv);returns.append(target)
    all_adv=torch.cat(advantages);mean=all_adv.mean();std=all_adv.std(unbiased=False)
    return [(adv-mean)/(std+1e-8) for adv in advantages],returns


def run(config,resume=None,stop_after_batches=None):
    validate(config);cfg=deepcopy(config);root=Path(cfg['project_root']);out=Path(cfg['output_dir']);t=cfg['train'];device=t['device']
    started=time.monotonic();runtime=runtime_identity(root)
    if not cfg['fixture'] and runtime['frozen_full_runtime']!=FROZEN_FULL:raise ValueError('Historical SFT runtime changed')
    if canonical_hash(runtime)!=cfg['runtime_sha256']:raise ValueError('Frozen posttraining runtime changed')
    if sha256_file(cfg['initial_checkpoint'])!=cfg['initial_checkpoint_sha256']:raise ValueError('Initial SFT weights changed')
    if sha256_file(cfg['arm_manifest'])!=cfg['arm_manifest_sha256']:raise ValueError('Frozen data arm changed')
    if resume is None and out.exists() and any(out.iterdir()):raise ValueError('Output is nonempty; use its own resume')
    if resume is not None and Path(resume).resolve()!=(out/'last.pt').resolve():raise ValueError('Resume must use this run last.pt')
    if device=='cuda':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.backends.cuda.matmul.allow_tf32=False
    random.seed(t['seed']);np.random.seed(t['seed']);torch.manual_seed(t['seed'])
    if device=='cuda':torch.cuda.manual_seed_all(t['seed'])
    torch.use_deterministic_algorithms(t['deterministic'])
    arm=ComparisonRLCorpus(cfg['arm_manifest'],allow_fixture=cfg['fixture'])
    if arm.source.fixture!=cfg['fixture']:raise ValueError('Fixture/production mismatch')
    sys.path.insert(0,str(root/'scripts'));from chat_sft import load_model
    actor,tokenizer,initial=load_model(cfg['initial_checkpoint'],device,threads=8 if device=='cuda' else 1)
    if not cfg['fixture'] and actor.parameter_count()!=303350784:raise ValueError('Wrong production model')
    actor.eval();actor.requires_grad_(True)
    # 参考模型总是从初始SFT复制；恢复策略权重发生在复制之后。
    reference=deepcopy(actor).eval().requires_grad_(False)
    value_head=None
    if cfg['algorithm']=='ppo':
        value_head=torch.nn.Linear(actor.config.dim,1).to(device);torch.nn.init.zeros_(value_head.weight);torch.nn.init.zeros_(value_head.bias)
    corpus=PreferenceCorpus(cfg['preference_manifest'],arm,tokenizer,cfg['initial_checkpoint_sha256']) if cfg['algorithm']=='dpo' else arm
    if len(corpus)!=cfg['train_examples']:raise ValueError('Frozen train count differs from accepted data')
    validation=SFTCorpus(cfg['validation_manifest'],'validation');dev=validation_subset(validation,t['eval_size'])
    batches=order_and_batches(corpus,t['prompt_batch_size'],t['seed']);passes=t['ppo_updates'] if cfg['algorithm']=='ppo' else 1
    total_updates=len(batches)*passes;warmup=math.ceil(total_updates*t['warmup_ratio'])
    plan={'epochs':1,'stop_epochs':None,'examples':len(corpus),'data_batches':len(batches),'optimizer_steps':total_updates,
          'ppo_updates_per_rollout':passes if cfg['algorithm']=='ppo' else None,'warmup_steps':warmup}
    identity={'config':canonical_hash(cfg),'runtime':canonical_hash(runtime),'initial_checkpoint_sha256':initial['checkpoint_sha256'],
        'arm_manifest_sha256':sha256_file(cfg['arm_manifest']),'tokenizer_sha256':initial['tokenizer_sha256'],
        'preference_manifest_sha256':sha256_file(cfg['preference_manifest']) if cfg.get('preference_manifest') else None,
        'validation_identity':validation.identity,'environment':_environment(device)}
    parameters=[{'params':actor.parameters(),'lr':t['learning_rate']}]
    if value_head is not None:parameters.append({'params':value_head.parameters(),'lr':t['critic_learning_rate']})
    optimizer=torch.optim.AdamW(parameters,betas=tuple(t['betas']),weight_decay=t['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(step,warmup,total_updates,t['min_lr_ratio']))
    cursor=0;optimizer_step=0;examples_seen=0;best=(-1.,-1.);best_step=0;latest=None;previous=None
    if resume is not None:
        previous=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
        if previous.get('identity')!=identity or previous.get('run_id')!=str(out.resolve()) or previous.get('checkpoint_kind')!='full':raise ValueError('Resume identity mismatch')
        actor.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);scheduler.load_state_dict(previous['scheduler'])
        if value_head is not None:value_head.load_state_dict(previous['value_head'])
        cursor=previous['data_batch_cursor'];optimizer_step=previous['optimizer_step'];examples_seen=previous['examples_seen']
        best=tuple(previous['best_score']);best_step=previous['best_step'];latest=previous['latest_validation']
        if not 0<=cursor<=len(batches) or optimizer_step!=cursor*passes:raise ValueError('Checkpoint cursor is outside original one-epoch plan')
    else:
        model_bytes=sum(p.numel()*p.element_size() for p in actor.parameters())
        if shutil.disk_usage(root).free<cfg['min_free_bytes']+7*model_bytes:raise OSError('Need reserved space plus best/last and atomic checkpoint temporary file')
    out.mkdir(parents=True,exist_ok=True);write_json(out/'config.json',cfg)
    write_json(out/'run_manifest.json',{'stage':'reading_rl','algorithm':cfg['algorithm'],'identity':identity,'runtime':runtime,'plan':plan,
        'initial_model':initial,'reference_policy':'same initial SFT; frozen','dropout':'disabled for all policy/ref scoring',
        'critic':'detached actor features plus independent trainable linear head' if value_head is not None else None,
        'test_used_for_selection':False,'data_regime':arm.manifest['condition']})
    guard=BudgetGuard(cfg['budget'],str(out.resolve()),cfg['budget']['ledger_path']);tracker=None;writer=None;status='failed';stop=False;handlers={}
    model_config=asdict(actor.config);cfg['_tokenizer']=tokenizer
    def stop_handler(signum,frame):
        nonlocal stop
        stop=True
    def persist(kind):
        model_bytes=sum(p.numel()*p.element_size() for p in actor.parameters());incoming=model_bytes*(3 if kind=='full' else 1)+16*2**20
        if shutil.disk_usage(out).free<cfg['min_free_bytes']+incoming:raise OSError('Checkpoint would violate disk reserve')
        payload={'schema_version':1,'stage':'reading_rl','algorithm':cfg['algorithm'],'checkpoint_kind':kind,
          'config':{k:v for k,v in cfg.items() if not k.startswith('_')},'identity':identity,'run_id':str(out.resolve()),
          'model_config':model_config,'model':actor.state_dict(),'data_batch_cursor':cursor,'optimizer_step':optimizer_step,
          'examples_seen':examples_seen,'best_score':best,'best_step':best_step,'latest_validation':latest,'plan':plan,
          'value_head':value_head.state_dict() if value_head is not None else None,
          'tokenizer':{'path':str(arm.source.tokenizer_path),'sha256':initial['tokenizer_sha256']}}
        if kind=='full':payload.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=_rng(device))
        save_checkpoint(out/('last.pt' if kind=='full' else 'best.pt'),payload)
    try:
        tracker=start_tracker(t,out,root,job_type=cfg['algorithm']+'-full',run_config={'algorithm':cfg['algorithm'],'train':t,'plan':plan,'data_arm':arm.manifest['condition']},resume=resume is not None)
        if t['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(str(out/'tensorboard'),purge_step=optimizer_step+1 if resume else None)
        if previous is not None:_restore_rng(previous['rng'],device)
        previous=None
        for sig in [signal.SIGINT,signal.SIGTERM]:handlers[sig]=signal.signal(sig,stop_handler)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        with (out/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            def record(values):
                row={'optimizer_step':optimizer_step,'data_batch_cursor':cursor,'examples_seen':examples_seen,
                     'epochs_equivalent':examples_seen/len(corpus),'elapsed_seconds':time.monotonic()-started,**values}
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();tracker.log(row,optimizer_step)
                print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
                if writer:
                    for key in ['loss','reward_mean','grad_norm','learning_rate']:
                        if key in row:writer.add_scalar(key,row[key],optimizer_step)
                    if 'validation' in row:writer.add_scalar('validation/exact_match',row['validation']['macro_exact_match'],optimizer_step)
            if latest is None:
                latest=evaluate(actor,tokenizer,validation,dev,device,t['eval_batch_size']);best=(latest['macro_exact_match'],latest['macro_f1']);persist('model_only')
                record({'event':'baseline','validation':latest})
            status='completed'
            while cursor<len(batches):
                if stop or (out/'STOP').exists():status='user_stop';break
                if stop_after_batches is not None and cursor>=stop_after_batches:status='bounded_smoke_stop';break
                if guard.should_stop():status='budget_or_time_limit';break
                rows=[corpus[i] for i in batches[cursor]];lr=optimizer.param_groups[0]['lr'];batch_started=time.monotonic()
                if cfg['algorithm']=='dpo':
                    stats=_dpo_update(actor,reference,optimizer,rows,cfg);scheduler.step();optimizer_step+=1
                else:
                    score=lambda row,text:__import__('llm_reading_data.verifier',fromlist=['reward']).reward(text,row['reference_answers'],row['language'])
                    rollout=_rollout(actor,reference,value_head,rows,cfg,score)
                    advantages,returns=_ppo_targets(rollout,cfg) if cfg['algorithm']=='ppo' else (None,None)
                    for _ in range(passes):
                        stats=_online_update(actor,value_head,optimizer,rollout,cfg,advantages,returns);scheduler.step();optimizer_step+=1
                    stats.update(reward_mean=sum(rollout['rewards'])/len(rollout['rewards']),eos_rate=rollout['eos_rate'],
                                 generated_tokens=rollout['generated_tokens'],rollout_responses=len(rollout['answers']))
                    del rollout,advantages,returns
                cursor+=1;examples_seen+=len(rows);evaluated=cursor%t['eval_interval']==0 or cursor==len(batches)
                if evaluated:
                    latest=evaluate(actor,tokenizer,validation,dev,device,t['eval_batch_size']);score=(latest['macro_exact_match'],latest['macro_f1'])
                    if score>best:best=score;best_step=optimizer_step;persist('model_only')
                if cursor==1 or cursor%t['log_interval']==0 or evaluated:
                    record({'event':'train',**stats,'learning_rate':lr,'batch_seconds':time.monotonic()-batch_started,
                            **({'validation':latest} if evaluated else {})})
                if cursor%t['save_interval']==0:persist('full')
            persist('full')
            summary={'status':status,'algorithm':cfg['algorithm'],'optimizer_step':optimizer_step,'data_batch_cursor':cursor,
                'examples_seen':examples_seen,'plan':plan,'best_score':best,'best_step':best_step,'latest_validation':latest,
                'budget':guard.update(),'elapsed_seconds':time.monotonic()-started,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None,
                'best_checkpoint':str(out/'best.pt'),'last_checkpoint':str(out/'last.pt')}
            write_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);return summary
    except BaseException as error:
        write_json(out/'failure.json',{'status':'failed','error_type':type(error).__name__,'message':str(error),
           'optimizer_step':optimizer_step,'data_batch_cursor':cursor,'resumable_checkpoint':str(out/'last.pt') if (out/'last.pt').exists() else None})
        status='failed';raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        guard.update()
        if writer:writer.close()
        if tracker:tracker.finish(0 if status in {'completed','user_stop','bounded_smoke_stop'} else 1)
