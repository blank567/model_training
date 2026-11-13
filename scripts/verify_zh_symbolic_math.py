"""对公开中文符号数学题独立重算；只输出验证成功的整条最终解答。"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_math_public.symbolic import verify,math_spans


def worker(job):
    index,prompt,response,identity=job
    result,reason=verify(prompt,response)
    return index,prompt,response,identity,result,reason


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--catalog',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();started=time.monotonic()
    spec=next(s for s in read_json(args.catalog) if s['source']==args.source and s['path']=='train.jsonl')
    if sha256_file(args.input)!=spec['sha256']:raise ValueError('Pinned public source changed')
    if args.output.exists():raise ValueError('Never overwrite a previous verification result')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    counts=Counter();tasks=Counter();unique=set()
    def jobs():
        for index,line in enumerate(args.input.open(encoding='utf-8')):
            raw=json.loads(line);counts['source_rows']+=1
            prompt=raw.get('prompt',raw.get('query'));response=raw.get('response')
            if not isinstance(prompt,str) or not isinstance(response,str):counts['invalid_text']+=1;continue
            if not any(word in prompt for word in ['因式分解','因子分解','展开','化简','简化','合并同类项','方程','求解','定积分']):
                counts['outside_supported_verifier_scope']+=1;continue
            if len(math_spans(prompt))!=1:counts['not_single_self_contained_expression']+=1;continue
            yield index,prompt,response,{'source':args.source,'revision':spec['revision'],'source_file_sha256':spec['sha256'],
                'source_row_sha256':canonical_hash(raw),'source_row':index,'source_id':raw.get('id'),
                'view':'complete response field; separate reasoning field is not the SFT target'}
    with args.output.open('x',encoding='utf-8') as writer,ProcessPoolExecutor(max_workers=args.workers) as pool:
        for processed,item in enumerate(pool.map(worker,jobs(),chunksize=100),1):
            index,prompt,response,identity,result,reason=item
            if reason:counts[reason]+=1;continue
            counts['verified_rows']+=1
            if result['task_sha256'] in unique:counts['duplicate_canonical_task']+=1;continue
            unique.add(result['task_sha256']);tasks[result['task']]+=1
            record={'language':'zh','question':prompt,'response':response,'verification':result,'provenance':identity}
            writer.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            if len(unique)%1000==0:
                writer.flush();print(json.dumps({'verified_unique':len(unique),'processed_candidates':processed}),flush=True)
    report={'status':'verified_candidate_pool','source':args.source,'counts':dict(counts),'unique_verified_rows':len(unique),
            'tasks':dict(tasks),'output':str(args.output),'sha256':sha256_file(args.output),
            'elapsed_seconds':time.monotonic()-started,'final_answer_verification':'independent bounded polynomial CAS',
            'full_natural_language_proof_certified':False,'training_started':False}
    write_json(args.report,report);print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
