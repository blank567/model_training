"""独立验收公开数学数据，--publish 后才允许用于正式训练。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llm_math_public.audit import audit

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--publish',action='store_true')
    args=ap.parse_args();report=audit(args.manifest,args.cache,args.publish)
    print(json.dumps({k:v for k,v in report.items() if k!='splits'},ensure_ascii=False),flush=True)
