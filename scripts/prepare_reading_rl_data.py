"""发布PPO/GRPO只读训练视图并明确DPO偏好对的待办状态；不执行GPU生成。"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llm_align.reading_rl_data import prepare

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=prepare(args.source,args.output)
    print(json.dumps({'status':'rl_input_view_ready','manifest':str(args.output),'splits':manifest['splits'],
        'dpo':manifest['dpo'],'sft_training_prompts_reused':True,'gpu_training_started':False},ensure_ascii=False))
