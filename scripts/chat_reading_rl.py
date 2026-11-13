"""已训练DPO/PPO/GRPO模型的单次生成；不读取最终测试题。"""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llm_rl.inference import load_model
from llm_rl.generation import generate
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--tokenizer');p.add_argument('--max-new-tokens',type=int,default=64)
    a=p.parse_args();model,tokenizer,_=load_model(a.checkpoint,a.device,a.tokenizer)
    prefix=tokenizer.encode('User:\n'+a.prompt+'\n\nAssistant:\n',add_special_tokens=False).ids
    print(generate(model,tokenizer,[prefix],a.device,a.max_new_tokens,sample=False)['texts'][0])
