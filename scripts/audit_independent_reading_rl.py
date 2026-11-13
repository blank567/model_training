"""验收不复用SFT材料的阅读RL候选；不执行任何模型训练。"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llm_reading_independent.audit import audit
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--source-lock',type=Path,required=True)
    p.add_argument('--publish',action='store_true');p.add_argument('--review',type=Path);a=p.parse_args()
    result=audit(a.manifest,a.cache,a.source_lock,a.publish,a.review)
    print(json.dumps({k:v for k,v in result.items() if k!='splits'},ensure_ascii=False),flush=True)
