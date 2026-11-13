"""读取补充计划，等待原GPU锁释放后检查十份阅读策略的通用回答建模保持。"""
from pathlib import Path
import argparse,fcntl,gc,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from llm_lab.common.io import read_json,write_json,write_jsonl,canonical_hash,sha256_file
from llm_lab.common.budget import BudgetGuard
from evaluate_posttrain_comprehensive import checkpoint_signature,get_corpus,load_model,state_digest
import evaluate_sft_models as metrics
import train_sft as locking


def run(path):
    plan=read_json(path);out=Path(plan['output_dir']);source=read_json(plan['generation_plan'])
    if sha256_file(plan['generation_plan'])!=plan['generation_plan_sha256']:raise ValueError('Main plan changed')
    if sha256_file(__file__)!=plan['evaluator_sha256']:raise ValueError('Supplemental evaluator changed')
    if canonical_hash({k:v for k,v in plan.items() if k!='plan_identity'})!=plan['plan_identity']:raise ValueError('Supplemental plan changed')
    models={m['key']:m for m in source['models']};spec=source['suites']['instruction_test_1000']
    write_json(out/'state.json',{'status':'waiting_for_generation_evaluation_lock','model_count':len(plan['models'])})
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    with (ROOT/'runs/.gpu_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if read_json(Path(source['output_dir'])/'state.json')['status']!='completed':raise RuntimeError('Generation evaluation did not complete')
        # GPU锁释放与前一Python进程退出之间可有短暂CUDA上下文清理时间。
        for attempt in range(30):
            try:locking.gpu_idle();break
            except RuntimeError:
                if attempt==29:raise
                time.sleep(1)
        corpus=get_corpus(spec);cache={};results=[]
        guard=BudgetGuard(plan['budget'],str(out.resolve()),ROOT/'runs/budget_ledger.json')
        try:
            for key in plan['models']:
                dest=out/key;dest.mkdir(exist_ok=True);item=models[key]
                signature,metadata=checkpoint_signature(item,corpus.identity['tokenizer_sha256'])
                write_json(out/'state.json',{'status':'evaluating_native_answer_loss','model':key})
                if signature in cache:
                    old=cache[signature];result={**read_json(old/'metrics.json'),'key':key,'model':item,
                        'metadata':metadata,'identical_tensor_result_from':str(old),'actual_gpu_forward':False}
                    (dest/'test_rows.jsonl').write_bytes((old/'test_rows.jsonl').read_bytes())
                else:
                    model,_=load_model(item,source['suites']['reading_test_2000']['manifest']);model.requires_grad_(False)
                    before=state_digest(model.state_dict())
                    values,rows=metrics.evaluate_loss(model,corpus,{'train':{'device':'cuda','dtype':'bf16','eval_micro_batch_size':8}},guard)
                    if len(rows)!=1000 or len({r['id'] for r in rows})!=1000:raise ValueError('Incomplete retention test')
                    if before!=state_digest(model.state_dict()):raise ValueError('Read-only weights changed')
                    write_jsonl(dest/'test_rows.jsonl',rows)
                    result={'key':key,'model':item,'metadata':metadata,'test':values,'test_identity':corpus.identity,
                            'examples':1000,'generation_performed':False,'weights_unchanged':True,'actual_gpu_forward':True}
                    del model;gc.collect();torch.cuda.empty_cache();cache[signature]=dest
                write_json(dest/'metrics.json',result);results.append(result)
                print(json.dumps({'key':key,'macro_ce':result['test']['macro_loss'],'examples':1000}),flush=True)
                if guard.should_stop():raise TimeoutError('Supplemental evaluation budget reached')
            write_json(out/'comparison.json',{'status':'completed','plan_identity':plan['plan_identity'],'results':results,
                'budget':guard.update(),'purpose':'cross-domain answer likelihood; not generated semantic accuracy'})
            write_json(out/'state.json',{'status':'completed','models':len(results),'examples_each':1000})
        except BaseException as error:
            write_json(out/'failure.json',{'type':type(error).__name__,'error':str(error)});traceback.print_exc();raise
        finally:guard.update()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);run(p.parse_args().plan)
