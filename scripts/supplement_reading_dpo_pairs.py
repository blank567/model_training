"""补采原旧匹配组尚缺负例的中文题，原筛选规则验收后发布三组偏好数据。"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,os,random,signal,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import canonical_hash,read_json,read_jsonl,sha256_file,write_json
from llm_reading_data.build import token_balanced
from llm_reading_independent.comparison import ComparisonRLCorpus
from llm_rl.preference_rules import acceptable_negative,material_span
from prepare_reading_dpo_pairs import generator_identity,source_row,publish


def combine_unique(base,additional):
    rows=list(base);seen={r['id'] for r in rows}
    if len(seen)!=len(rows):raise ValueError('Duplicate original preference ID')
    for row in additional:
        if row['id'] in seen:raise ValueError('Supplement would duplicate an existing question')
        seen.add(row['id']);rows.append(row)
    return rows


def select_matched(old,new,minimum=20000):
    maximum=min(len(old),len(new))
    if maximum<minimum:return None
    sizes=list(range(maximum,minimum-1,-100))
    if minimum not in sizes:sizes.append(minimum)
    for count in sizes:
        try:
            left,a=token_balanced(old,count,20261102);right,b=token_balanced(new,count,20261102)
            return left,right,{'examples_per_matched_arm':count,'old':a,'new':b}
        except ValueError:
            left,a=exchange_balanced(old,count);right,b=exchange_balanced(new,count)
            if left is not None and right is not None:
                return left,right,{'examples_per_matched_arm':count,'old':a,'new':b}
    return None


def exchange_balanced(rows,count):
    """在数学可行的语言数量中交换答案长度；只取唯一合格题，不使用模型评分选题。"""
    groups={k:[r for r in rows if r['language']==k] for k in ['zh','en']}
    if not all(groups.values()) or count>len(rows):return None,None
    sums={}
    for k in groups:
        random.Random('20261104-selection-'+k).shuffle(groups[k]);values=sorted(r['answer_token_count'] for r in groups[k]);prefix=[0]
        for value in values:prefix.append(prefix[-1]+value)
        sums[k]=prefix
    mean={k:sums[k][-1]/len(groups[k]) for k in groups};desired=round(count*mean['en']/(mean['zh']+mean['en']))
    possible=[]
    for nz in range(max(1,count-len(groups['en'])),min(count-1,len(groups['zh']))+1):
        ne=count-nz;z=sums['zh'];e=sums['en'];zlo=z[nz];zhi=z[-1]-z[len(z)-1-nz];elo=e[ne];ehi=e[-1]-e[len(e)-1-ne]
        if zlo/(zlo+ehi)<=.505 and zhi/(zhi+elo)>=.495:possible.append(nz)
    if not possible:return None,None
    nz=min(possible,key=lambda n:(abs(n-desired),n));counts={'zh':nz,'en':count-nz}
    chosen={k:groups[k][:counts[k]] for k in groups};unused={k:groups[k][counts[k]:] for k in groups}
    token={k:sum(r['answer_token_count'] for r in chosen[k]) for k in groups}
    fraction=lambda:token['zh']/sum(token.values());swaps=0
    high='zh' if fraction()>.5 else 'en';low='en' if high=='zh' else 'zh';moves={}
    # 降低偏多语言的Token，或提高偏少语言的Token，两种交换均保持题数与唯一性。
    moves[high]=list(zip(sorted(chosen[high],key=lambda r:-r['answer_token_count']),sorted(unused[high],key=lambda r:r['answer_token_count'])))
    moves[low]=list(zip(sorted(chosen[low],key=lambda r:r['answer_token_count']),sorted(unused[low],key=lambda r:-r['answer_token_count'])))
    position={k:0 for k in groups};selected={r['id']:r for values in chosen.values() for r in values}
    while abs(fraction()-.5)>.005:
        options=[]
        for k in [high,low]:
            if position[k]>=len(moves[k]):continue
            before,after=moves[k][position[k]];change=after['answer_token_count']-before['answer_token_count']
            if (k==high and change>=0) or (k==low and change<=0):continue
            trial={**token,k:token[k]+change};distance=abs(trial['zh']/sum(trial.values())-.5)
            if distance<abs(fraction()-.5):options.append((distance,k,before,after,change))
        if not options:return None,None
        _,k,before,after,change=min(options,key=lambda item:(item[0],item[1]));position[k]+=1
        selected.pop(before['id']);selected[after['id']]=after;token[k]+=change;swaps+=1
    result=list(selected.values());random.Random(20261104).shuffle(result)
    assert len(result)==count and len(selected)==count
    return result,{'zh_examples':nz,'en_examples':count-nz,'zh_answer_tokens':token['zh'],'en_answer_tokens':token['en'],
        'zh_answer_token_fraction':fraction(),'tolerance':.005,'answer_length_exchanges':swaps,
        'selection_note':'Answer-length conditioned unique subset; no evaluation or model accuracy used.'}


def unresolved_chinese(arm,accepted):
    selected=[i for i in range(len(arm)) if arm[i]['language']=='zh' and arm[i]['id'] not in accepted]
    return sorted(selected,key=lambda i:hashlib.sha256(('20261104-zh-supplement/'+arm[i]['id']).encode()).digest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);a=p.parse_args()
    q=read_json(a.state);plan=read_json(q['plan']);initial=plan['initial_checkpoint'];base=Path(q['preference_work'])
    work=base/'supplement_zh_v1';work.mkdir(exist_ok=True)
    old=read_jsonl(base/'old50k/candidates.jsonl');new=read_jsonl(base/'new25k/candidates.jsonl')
    identity=generator_identity(initial)
    arms={name:ComparisonRLCorpus(Path(q['plan']).parent/spec['manifest']) for name,spec in plan['arms'].items()}
    assert sha256_file(initial['path'])==initial['sha256']
    for label,name in [('old50k','old_full_50k'),('new25k','new_matched')]:
        s=read_json(base/label/'state.json')
        assert s['cursor']==len(arms[name]) and s['identity']=={**identity,'source_arm_identity':arms[name].manifest['data_identity']}
        assert (base/label/'candidates.jsonl').stat().st_size==s['file_bytes']
    original_ids={r['id'] for r in old};old_matched_ids={arms['old_matched'][i]['id'] for i in range(len(arms['old_matched']))}
    indices=unresolved_chinese(arms['old_matched'],original_ids)
    binding={'base_generator':identity,'supplement_script_sha256':sha256_file(__file__),
        'old_candidates_sha256':sha256_file(base/'old50k/candidates.jsonl'),'new_candidates_sha256':sha256_file(base/'new25k/candidates.jsonl'),
        'source_arm_identity':arms['old_matched'].manifest['data_identity'],'extra_candidate_limit':32,
        'temperature':1.,'max_new_tokens':64,'seed':20261104,'model_updates':0,'eligible_failed_chinese':len(indices)}
    statefile=work/'state.json';records=work/'pairs.jsonl'
    state={'identity':binding,'cursor':0,'file_bytes':0,'pairs':0,'sampled_responses':0}
    if statefile.exists():
        state=read_json(statefile);assert state['identity']==binding,'Supplement resume identity mismatch'
    if records.exists():
        with records.open('r+b') as stream:
            assert stream.seek(0,2)>=state['file_bytes'];stream.truncate(state['file_bytes'])
    added=read_jsonl(records) if records.exists() else []
    merged=combine_unique(old,added);choice=select_matched([r for r in merged if r['id'] in old_matched_ids],new)
    import torch
    from llm_lab.common.budget import BudgetGuard
    from llm_rl.generation import generate
    from train_sft import training_lock,gpu_idle
    from chat_sft import load_model
    guard=None;stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    try:
        with training_lock(ROOT):
            gpu_idle();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.backends.cuda.matmul.allow_tf32=False
            torch.use_deterministic_algorithms(True);torch.manual_seed(42)
            guard=BudgetGuard({'hourly_rate_cny':3.03,'run_limit_cny':6,'project_limit_cny':500,'reserve_cny':100,
                'max_runtime_seconds':3600},str(work),ROOT/'runs/budget_ledger.json')
            actor,tokenizer,_=load_model(initial['path'],'cuda',threads=8);actor.eval().requires_grad_(False)
            with records.open('ab') as stream:
                for start in range(state['cursor'],len(indices),32):
                    if choice is not None:break
                    if stopping:raise InterruptedError('User stopped supplement; committed journal retained')
                    if guard.should_stop():raise TimeoutError('Bounded supplement budget/time reached')
                    batch=[source_row(arms['old_matched'],i) for i in indices[start:start+32]];pending=list(range(len(batch)));found={}
                    for round_id in range(8):
                        if not pending:break
                        expanded=[batch[i] for i in pending for _ in range(4)];seed=20261104+start*19+round_id
                        rng=torch.Generator(device='cuda').manual_seed(seed)
                        result=generate(actor,tokenizer,[r['prefix_ids'] for r in expanded],'cuda',generator=rng)
                        state['sampled_responses']+=len(expanded)
                        for offset,i in enumerate(pending):
                            row=batch[i];candidates=list(range(offset*4,offset*4+4))
                            candidates.sort(key=lambda j:not material_span(result['texts'][j],row))
                            for j in candidates:
                                ids,text=result['answers'][j],result['texts'][j]
                                if acceptable_negative(ids,text,row):
                                    chosen=tokenizer.encode(row['chosen'],add_special_tokens=False).ids+[2]
                                    found[i]={'id':row['id'],'language':'zh','prefix_sha256':canonical_hash(row['prefix_ids']),
                                        'chosen_ids':chosen,'rejected_ids':ids,'rejected_text':text,'answer_token_count':len(chosen),
                                        'negative_type':'wrong_material_span' if material_span(text,row) else 'not_a_material_span',
                                        'generation':{'phase':'zh_supplement_v1','batch_start':start,'round':round_id,'candidate':j%4,'seed':seed}}
                                    break
                        pending=[i for i in pending if i not in found]
                    additions=[found[i] for i in sorted(found)]
                    for row in additions:stream.write((json.dumps(row,ensure_ascii=False)+'\n').encode())
                    stream.flush();os.fsync(stream.fileno());added.extend(additions)
                    state.update(cursor=start+len(batch),file_bytes=stream.tell(),pairs=len(added));write_json(statefile,state)
                    merged=combine_unique(old,added);choice=select_matched([r for r in merged if r['id'] in old_matched_ids],new)
                    progress={'status':'supplementing' if choice is None else 'balanced_selection_ready',
                        'processed_failed_chinese':state['cursor'],'supplemental_pairs':len(added),'eligible_failed_chinese':len(indices),
                        'selected_count':choice[2]['examples_per_matched_arm'] if choice else None,'budget':guard.update()}
                    write_json(work/'progress.json',progress);print(json.dumps(progress),flush=True)
            if choice is None:raise ValueError('Still insufficient balanced 20000 pairs after bounded supplement; review required')
            del actor;torch.cuda.empty_cache()
            left,right,balance=choice;out=Path(q['preference_output']);published={}
            for name,rows in [('old_matched',left),('new_matched',right),('old_full_50k',merged)]:
                published[name]=publish(rows,arms[name],tokenizer,initial,binding,out/name)
                assert published[name]['examples']==len(rows),'Previously published partial arm differs from current selection'
            assert sha256_file(initial['path'])==initial['sha256']
            receipt={'status':'published','plan_sha256':sha256_file(q['plan']),'initial_checkpoint_sha256':initial['sha256'],
                'arms':published,'matched_pair_count':len(left),'source_prompt_counts':{'old_full_50k':50000,'new_matched':25000},
                'supplement':{'path':str(work),'pairs':len(added),'balance':balance,'recipe':binding},
                'budget':guard.update(),'final_test_used':False}
            write_json(out/'preference_index.json',receipt);write_json(work/'completed.json',receipt);print(json.dumps(receipt),flush=True)
    except BaseException as error:
        write_json(work/'failure.json',{'type':type(error).__name__,'error':str(error)});raise
    finally:
        if guard:guard.update()
if __name__=='__main__':main()
