import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from llm_math_public.vetted_audit import audit

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='独立核对有标准答案的数学候选；发布必须绑定抽检回执')
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--publish',action='store_true');ap.add_argument('--review',type=Path)
    a=ap.parse_args();r=audit(a.manifest,a.cache,a.publish,a.review)
    print(json.dumps({k:v for k,v in r.items() if k!='splits'},ensure_ascii=False),flush=True)
