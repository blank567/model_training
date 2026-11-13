"""同一SFT生成负回答，筛选错误片段或抽取格式负例；发布等量主对照，无参数更新。"""
from pathlib import Path
from collections import Counter
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from llm_lab.common.io import canonical_hash,read_json,read_jsonl,sha256_file,write_json,write_jsonl
from llm_lab.common.budget import BudgetGuard
from llm_reading_independent.comparison import ComparisonRLCorpus
from llm_reading_data.build import token_balanced
from llm_rl.generation import generate
from llm_rl.config import runtime_identity
from llm_rl.preference_rules import acceptable_negative,material_span
from llm_rl.data import PreferenceCorpus
from train_sft import training_lock,gpu_idle
from chat_sft import load_model


def generator_identity(initial):
    return {'initial_checkpoint_sha256':initial['sha256'],'runtime_sha256':canonical_hash(runtime_identity(ROOT)),
        'generator_sha256':sha256_file(__file__),'rules_sha256':sha256_file(ROOT/'src/llm_rl/preference_rules.py'),
        'temperature':1.,'top_p':1.,'candidate_limit_per_prompt':16,'candidates_per_round':4,
        'max_new_tokens':64,'prompt_batch_size':32,'seed':20261102,'updates':0}


def source_row(arm,index):
    row=arm[index];return {**row,'context':arm.source.rows[arm.indices[index]]['context']}


def produce(model,tokenizer,arm,work,identity,guard):
    work.mkdir(parents=True,exist_ok=True);state_file=work/'state.json';records=work/'candidates.jsonl'
    binding={**identity,'source_arm_identity':arm.manifest['data_identity']}
    state={'identity':binding,'cursor':0,'file_bytes':0,'pairs':0,'skipped':0}
    if state_file.exists():
        state=read_json(state_file)
        if state['identity']!=binding:raise ValueError('Preference resume recipe/source differs')
    if records.exists():
        with records.open('r+b') as stream:
            if stream.seek(0,2)<state['file_bytes']:raise ValueError('Missing committed candidate data')
            stream.truncate(state['file_bytes'])  # 仅移除中断后尚未提交的候选记录尾部。
    elif state['file_bytes']:raise ValueError('Committed candidates are missing')
    with records.open('ab') as stream:
        for start in range(state['cursor'],len(arm),32):
            if guard.should_stop():raise TimeoutError('Preference generation budget/time reached; resumable state retained')
            rows=[source_row(arm,i) for i in range(start,min(start+32,len(arm)))];pending=list(range(len(rows)));found={}
            for round_id in range(4):
                if not pending:break
                expanded=[rows[i] for i in pending for _ in range(4)]
                rng=torch.Generator(device='cuda').manual_seed(identity['seed']+start*17+round_id)
                result=generate(model,tokenizer,[r['prefix_ids'] for r in expanded],'cuda',generator=rng)
                for offset,i in enumerate(pending):
                    row=rows[i]
                    candidates=list(range(offset*4,offset*4+4))
                    candidates.sort(key=lambda j:not material_span(result['texts'][j],row))
                    for j in candidates:
                        ids,text=result['answers'][j],result['texts'][j]
                        if acceptable_negative(ids,text,row):
                            found[i]={'id':row['id'],'language':row['language'],'prefix_sha256':canonical_hash(row['prefix_ids']),
                                'chosen_ids':tokenizer.encode(row['chosen'],add_special_tokens=False).ids+[2],
                                'rejected_ids':ids,'rejected_text':text,'answer_token_count':len(tokenizer.encode(row['chosen'],add_special_tokens=False).ids)+1,
                                'negative_type':'wrong_material_span' if material_span(text,row) else 'not_a_material_span',
                                'generation':{'batch_start':start,'round':round_id,'candidate':j%4,'seed':identity['seed']+start*17+round_id}}
                            break
                pending=[i for i in pending if i not in found]
            for i in sorted(found):stream.write((json.dumps(found[i],ensure_ascii=False)+'\n').encode())
            stream.flush();os.fsync(stream.fileno());state.update(cursor=start+len(rows),file_bytes=stream.tell(),
                pairs=state['pairs']+len(found),skipped=state['skipped']+len(pending));write_json(state_file,state)
            if start==0 or state['cursor']%1024==0:print(json.dumps({'source':work.name,**{k:v for k,v in state.items() if k!='identity'}}),flush=True)
    return read_jsonl(records)


def publish(rows,arm,tokenizer,initial,identity,out):
    manifest=out/'manifest.json'
    if manifest.exists():
        checked=PreferenceCorpus(manifest,arm,tokenizer,initial['sha256']);return {'manifest':str(manifest),'examples':len(checked)}
    out.mkdir(parents=True,exist_ok=True);lookup={arm[i]['id']:i for i in range(len(arm))};seen=set()
    for row in rows:
        if row['id'] not in lookup or row['id'] in seen:raise ValueError('Preference question duplicated or outside arm')
        source=source_row(arm,lookup[row['id']]);gold=tokenizer.encode(source['chosen'],add_special_tokens=False).ids+[2]
        if row['prefix_sha256']!=canonical_hash(source['prefix_ids']) or row['chosen_ids']!=gold:raise ValueError('Gold/prompt mismatch')
        text=tokenizer.decode(row['rejected_ids'][:-1],skip_special_tokens=False)
        if not acceptable_negative(row['rejected_ids'],text,source):raise ValueError('Negative failed full publication recheck')
        seen.add(row['id'])
    write_jsonl(out/'pairs.jsonl',rows);digest=sha256_file(out/'pairs.jsonl')
    checks={k:True for k in ['original_human_gold','initial_sft_generated_negatives','negative_eos_and_token_checks',
        'negative_reward_zero','ambiguous_aliases_excluded','source_and_prefix_bound']}
    write_json(out/'acceptance.json',{'status':'passed','records_sha256':digest,'examples':len(rows),'checks':checks,
        'limitations':['Rule-derived preferences; rejected labels are not independently human judged.',
            'No semantic alias detector is complete; substrings, accents, F1 and edit similarity are conservatively filtered.']})
    write_json(manifest,{'schema_version':1,'status':'published','kind':'reading_policy_preferences',
        'initial_checkpoint_sha256':initial['sha256'],'source_manifest_sha256':arm.manifest['source_manifest_sha256'],
        'source_arm_identity':arm.manifest['data_identity'],'generator':identity,'examples':len(rows),
        'records':{'path':'pairs.jsonl','sha256':digest},'acceptance':{'path':'acceptance.json','sha256':sha256_file(out/'acceptance.json')},
        'languages':dict(Counter(r['language'] for r in rows)),
        'negative_types':dict(Counter(r.get('negative_type','fixture') for r in rows)),
        'human_preference_labels':False,'training_epochs':1})
    checked=PreferenceCorpus(manifest,arm,tokenizer,initial['sha256']);return {'manifest':str(manifest),'examples':len(checked)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    if not all(path.resolve().is_relative_to(ROOT) for path in [a.output,a.work]):raise ValueError('Output escapes project')
    plan=read_json(a.plan);initial=plan['initial_checkpoint'];identity=generator_identity(initial);a.work.mkdir(parents=True,exist_ok=True)
    guard=None
    try:
        with training_lock(ROOT):
            gpu_idle();assert sha256_file(initial['path'])==initial['sha256']
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.backends.cuda.matmul.allow_tf32=False
            torch.use_deterministic_algorithms(True);torch.manual_seed(42)
            guard=BudgetGuard({'hourly_rate_cny':3.03,'run_limit_cny':20,'project_limit_cny':500,'reserve_cny':100,
                'max_runtime_seconds':21600},str(a.work),ROOT/'runs/budget_ledger.json')
            actor,tokenizer,_=load_model(initial['path'],'cuda',threads=8);actor.eval().requires_grad_(False)
            arms={name:ComparisonRLCorpus(a.plan.parent/spec['manifest']) for name,spec in plan['arms'].items()}
            old=produce(actor,tokenizer,arms['old_full_50k'],a.work/'old50k',identity,guard)
            new=produce(actor,tokenizer,arms['new_matched'],a.work/'new25k',identity,guard)
            old_ids={arms['old_matched'][i]['id'] for i in range(len(arms['old_matched']))};matched=[r for r in old if r['id'] in old_ids]
            count=min(len(matched),len(new));selected=None
            # 不复制问题或伪造负例补齐数量；在质量过滤后保持两组样本数相同及回答Token中英均衡。
            for size in range(count,19999,-100):
                try:
                    left,_=token_balanced(matched,size,20261102);right,_=token_balanced(new,size,20261102);selected=(left,right);break
                except ValueError:continue
            if selected is None:raise ValueError('Fewer than 20000 balanced matched valid DPO pairs; manual review required')
            results={}
            for name,rows in [('old_matched',selected[0]),('new_matched',selected[1]),('old_full_50k',old)]:
                results[name]=publish(rows,arms[name],tokenizer,initial,identity,a.output/name)
            assert sha256_file(initial['path'])==initial['sha256']
            receipt={'status':'published','plan_sha256':sha256_file(a.plan),'initial_checkpoint_sha256':initial['sha256'],
                'arms':results,'matched_pair_count':len(selected[0]),'source_prompt_counts':{'old_full_50k':50000,'new_matched':25000},
                'budget':guard.update(),'final_test_used':False}
            write_json(a.output/'preference_index.json',receipt);print(json.dumps(receipt),flush=True)
    except BaseException as error:
        write_json(a.work/'failure.json',{'type':type(error).__name__,'error':str(error)});traceback.print_exc();raise
    finally:
        if guard:guard.update()
if __name__=='__main__':main()
