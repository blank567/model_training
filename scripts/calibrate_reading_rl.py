"""受限GPU数值/学习信号/三算法反向传播检查；不保存模型、不访问最终测试。"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_lab.common.budget import BudgetGuard
from llm_reading_independent.comparison import ComparisonRLCorpus
from llm_reading_data.verifier import reward
from llm_rl.config import DEFAULT_TRAIN,runtime_identity
from llm_rl.generation import prefill,generate,precision
from llm_rl.engine import _rollout,_online_update,_ppo_targets,_dpo_update
from train_sft import training_lock,gpu_idle
from chat_sft import load_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out=ROOT/'runs/assistant/benchmarks'/('rl_calibration_'+stamp);out.mkdir(parents=True,exist_ok=False)
    plan=read_json(a.plan);initial=plan['initial_checkpoint'];started=time.monotonic()
    guard=None
    try:
        with training_lock(ROOT):
            gpu_idle();assert sha256_file(initial['path'])==initial['sha256']
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
            guard=BudgetGuard({'hourly_rate_cny':3.03,'run_limit_cny':1,'project_limit_cny':500,
                'reserve_cny':100,'max_runtime_seconds':600},str(out),ROOT/'runs/budget_ledger.json')
            actor,tokenizer,_=load_model(initial['path'],'cuda',threads=8);actor.eval().requires_grad_(True)
            reference=deepcopy(actor).eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats()
            arms={k:ComparisonRLCorpus(a.plan.parent/v['manifest']) for k,v in plan['arms'].items()}
            rows=[arms['old_matched'][i] for i in [0,1]];prompts=[r['prefix_ids'] for r in rows]
            # BF16缓存与原始全序列前向对照；不同矩阵形状有舍入误差，记录完整误差与argmax。
            with torch.no_grad():
                state=prefill(actor,prompts,'cuda');errors=[];relative=[];argmax=[]
                for step in range(3):
                    for i,prompt in enumerate(prompts):
                        with precision('cuda'):expected=actor(torch.tensor([prompt],device='cuda')).logits[0,-1].float()
                        difference=(state.logits[i]-expected).abs();errors.append(float(difference.max()))
                        relative.append(float(difference.norm()/expected.norm().clamp_min(1e-6)))
                        argmax.append(bool(state.logits[i].argmax()==expected.argmax()))
                    chosen=state.logits.argmax(-1)
                    for i,token in enumerate(chosen.tolist()):prompts[i].append(token)
                    if step<2:state.advance(chosen,torch.ones(2,dtype=torch.bool,device='cuda'))
                del state
            assert max(relative)<.02 and all(argmax),'GPU cached logits diverged from original Decoder'
            report={'status':'running','initial_checkpoint_sha256':initial['sha256'],'runtime_sha256':canonical_hash(runtime_identity(ROOT)),
                'cache_check':{'max_absolute_error':max(errors),'max_relative_l2_error':max(relative),'all_argmax_equal':all(argmax)},
                'arms':{},'optimizer_smoke':{},'final_test_used':False,'output':str(out)}
            sampled={}
            for name,arm in arms.items():
                # 训练样本按语言固定分层抽样；32题×4个真实策略采样。
                selected=[]
                for language in ['zh','en']:
                    available=[i for i in range(len(arm)) if arm[i]['language']==language]
                    selected.extend(available[::max(1,len(available)//16)][:16])
                rewards=[];mixed=0;generated=[];records=[]
                for start in range(0,len(selected),8):
                    if guard.should_stop():raise TimeoutError('Calibration bounded budget/time reached')
                    batch=[arm[i] for i in selected[start:start+8]];expanded=[r for r in batch for _ in range(4)]
                    result=generate(actor,tokenizer,[r['prefix_ids'] for r in expanded],'cuda')
                    score=[reward(text,row['reference_answers'],row['language']) for row,text in zip(expanded,result['texts'])]
                    rewards.extend(score);mixed+=sum(min(score[i:i+4])<max(score[i:i+4]) for i in range(0,len(score),4))
                    generated.extend(result['eos'])
                    for row,answer,text,value in zip(expanded,result['answers'],result['texts'],score):
                        records.append({'id':row['id'],'language':row['language'],'completion':text,'reward':value})
                        if name=='old_matched':sampled.setdefault(row['id'],(row,[]))[1].append((answer,value))
                report['arms'][name]={'prompts':len(selected),'responses':len(rewards),'correct':sum(rewards),
                    'mixed_reward_groups':mixed,'eos_rate':sum(generated)/len(generated)}
                write_json(out/(name+'_samples.json'),records);write_json(out/'progress.json',report)
                print(json.dumps({'arm':name,**report['arms'][name]}),flush=True)
            assert all(v['correct']>0 and v['mixed_reward_groups']>0 for v in report['arms'].values()),'Insufficient real on-policy reward signal'
            base={'train':deepcopy(DEFAULT_TRAIN),'_tokenizer':tokenizer};base['train']['group_size']=1
            # 每算法均从同一SFT恢复；这些更新只用于GPU短测，随进程退出丢弃。
            train_rows=[value[0] for value in sampled.values()][:16]
            for algorithm in ['ppo','grpo','dpo']:
                if guard.should_stop():raise TimeoutError('Calibration bounded budget/time reached')
                actor.load_state_dict(reference.state_dict());actor.zero_grad(set_to_none=True)
                cfg=deepcopy(base);cfg['algorithm']=algorithm;cfg['train']['group_size']=4 if algorithm=='grpo' else 1
                value_head=torch.nn.Linear(actor.config.dim,1).cuda() if algorithm=='ppo' else None
                if value_head is not None:torch.nn.init.zeros_(value_head.weight);torch.nn.init.zeros_(value_head.bias)
                parameters=list(actor.parameters())+(list(value_head.parameters()) if value_head is not None else [])
                optimizer=torch.optim.AdamW(parameters,lr=1e-6,betas=(.9,.95));begin=time.monotonic()
                if algorithm=='dpo':
                    pairs=[]
                    for row,candidates in sampled.values():
                        bad=next((ids for ids,score in candidates if not score and len(ids)>1 and ids[-1]==2),None)
                        if bad:pairs.append({**row,'chosen_ids':tokenizer.encode(row['chosen'],add_special_tokens=False).ids+[2],'rejected_ids':bad})
                    assert pairs,'No actual sampled DPO negative for GPU smoke'
                    stats=_dpo_update(actor,reference,optimizer,pairs[:16],cfg)
                else:
                    roll=_rollout(actor,reference,value_head,train_rows,cfg,
                        lambda row,text:reward(text,row['reference_answers'],row['language']))
                    advantages,returns=_ppo_targets(roll,cfg) if algorithm=='ppo' else (None,None)
                    for _ in range(2 if algorithm=='ppo' else 1):stats=_online_update(actor,value_head,optimizer,roll,cfg,advantages,returns)
                    del roll,advantages,returns
                torch.cuda.synchronize();assert all(torch.isfinite(p).all() for p in parameters)
                report['optimizer_smoke'][algorithm]={'status':'passed','seconds':time.monotonic()-begin,**stats}
                print(json.dumps({'algorithm':algorithm,**report['optimizer_smoke'][algorithm]}),flush=True)
                del optimizer,value_head,parameters;actor.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            # 最长合法输入的16条微批反向传播，确认正式训练不会只在短样本上通过。
            longest=sorted((r for name in ['old_matched','new_matched'] for r in [arms[name][i] for i in range(len(arms[name]))]),
                key=lambda r:len(r['prefix_ids']),reverse=True)[:16]
            from llm_rl.scoring import token_scores
            actor.load_state_dict(reference.state_dict())
            scores,_=token_scores(actor,[r['prefix_ids'] for r in longest],[[3]*63+[2] for _ in longest],'cuda')
            (-torch.stack([v.mean() for v in scores]).mean()).backward();actor.zero_grad(set_to_none=True)
            # 上述合成回答仅用于最坏长度内存测试，不写入任何训练数据。
            torch.cuda.synchronize();assert sha256_file(initial['path'])==initial['sha256']
            report.update(status='passed',safe_score_micro_batch_size=16,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                elapsed_seconds=time.monotonic()-started,budget=guard.update())
            write_json(out/'calibration.json',report);print(json.dumps(report),flush=True)
    except BaseException as error:
        write_json(out/'failure.json',{'error':str(error),'type':type(error).__name__});traceback.print_exc();raise
    finally:
        if guard:guard.update()
if __name__=='__main__':main()
