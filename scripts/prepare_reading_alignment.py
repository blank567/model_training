"""准备50,000条中英监督Token均衡的人工标注阅读理解数据。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from llm_reading_data.build import build

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    a=ap.parse_args();build(ROOT,a.cache,a.tokenizer,a.output,a.work)
