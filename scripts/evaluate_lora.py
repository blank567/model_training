"""对已锁定配置执行Full-vs-LoRA的1000题最终评测。"""
import argparse,json
from pathlib import Path
from llm_lora.evaluate import final_evaluation

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selection',type=Path,required=True)
    print(json.dumps(final_evaluation(parser.parse_args().selection),ensure_ascii=False),flush=True)
