"""同一模型、同128条输入的推理批量短测；不计算任务分数、不更新参数。"""
from pathlib import Path
import sys,time,json,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import torch
from llm_lab.common.io import read_json,write_json
from evaluate_posttrain_comprehensive import get_corpus,load_model,cached_greedy,state_digest
import train_sft as locking
work=ROOT/'runs/assistant/benchmarks/posttrain_eval_batch_20261104';work.mkdir(exist_ok=False)
plan=read_json(ROOT/'runs/assistant/inference/posttrain_comprehensive_20261104_v2/plan.json')
corpus=get_corpus(plan['suites']['instruction_test_1000'])
item=next(m for m in plan['models'] if m['key']=='sft100k_full')
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
order=sorted(corpus.rows,key=lambda r:r['prompt_token_count'],reverse=True)[:128]
prefixes=[r['input_ids'][:r['prompt_token_count']] for r in order]
with locking.training_lock(ROOT):
    locking.gpu_idle();model,_=load_model(item,plan['suites']['reading_test_2000']['manifest']);model.requires_grad_(False)
    before=state_digest(model.state_dict());records=[]
    cached_greedy(model,prefixes[:2],2,'cuda',(0,1))
    for batch_size in [16,128]:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic();out=[]
        try:
            for start in range(0,128,batch_size):out.extend(cached_greedy(model,prefixes[start:start+batch_size],32,'cuda',(0,1)))
            torch.cuda.synchronize();elapsed=time.monotonic()-started
            records.append({'batch_size':batch_size,'seconds':elapsed,'examples':len(out),'new_tokens':sum(len(r['generated_ids']) for r in out),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'status':'completed',
                'generated_ids':[r['generated_ids'] for r in out]})
        except torch.cuda.OutOfMemoryError:
            records.append({'batch_size':batch_size,'status':'out_of_memory'});torch.cuda.empty_cache()
    assert before==state_digest(model.state_dict())
    if all(r['status']=='completed' for r in records):
        agreement=sum(a==b for a,b in zip(records[0]['generated_ids'],records[1]['generated_ids']))
        result={'status':'completed','records':records,'speedup':records[0]['seconds']/records[1]['seconds'],
                'first32_exact_sequence_matches':agreement,'weights_unchanged':True,'final_task_scores_not_used':True}
    else:result={'status':'benchmark_failed','records':records,'weights_unchanged':True}
    write_json(work/'benchmark.json',result)
    print(json.dumps({**result,'records':[{k:v for k,v in r.items() if k!='generated_ids'} for r in records]}))
