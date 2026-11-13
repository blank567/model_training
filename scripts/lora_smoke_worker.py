"""受限GPU正确性子进程；输出限定在助手smoke目录。"""
import argparse,json
from pathlib import Path
from llm_lora.config import read_config
from llm_lora.engine import run_lora
import train_sft as locking

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--until-step',required=True,type=int);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    cfg=read_config(args.config);output=Path(cfg['output_dir'])
    if not output.resolve().is_relative_to(Path(cfg['project_root'])/'runs/assistant/smoke'):
        raise ValueError('GPU smoke outputs must stay under assistant/smoke')
    with locking.training_lock(cfg['project_root']):
        locking.gpu_idle();locking.check_cuda(cfg)
        result=run_lora(cfg,resume=output/'last.pt' if args.resume else None,stop_after_steps=args.until_step)
    print(json.dumps(result,ensure_ascii=False),flush=True)
