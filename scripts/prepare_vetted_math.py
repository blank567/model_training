"""准备有来源标准答案的广泛数学候选；独立验收之前禁止训练。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from llm_math_public.vetted import build

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    args=ap.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    build(ROOT,args.cache,args.tokenizer,args.output,args.work)
