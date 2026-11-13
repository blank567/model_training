import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llm_reading_data.audit import audit

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='独立验收人工阅读理解数据；完整核对官方标注、简体答案、Token和文章隔离')
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--publish',action='store_true');ap.add_argument('--review',type=Path)
    a=ap.parse_args();r=audit(a.manifest,a.cache,a.publish,a.review)
    print(json.dumps({k:v for k,v in r.items() if k not in {'splits','normalizer_resources'}},ensure_ascii=False),flush=True)
