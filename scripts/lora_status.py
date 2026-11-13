"""只读LoRA实时进度；区分运行中的恢复进程和旧summary快照。"""
import argparse,json
from pathlib import Path
from llm_lab.common.io import read_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]

def read(path):return read_json(path) if path.is_file() else None

def snapshot(config_path):
    path=Path(config_path).resolve()
    if not path.is_relative_to(ROOT/'runs/configs'):raise ValueError('Select a frozen project LoRA config')
    config=read_json(path);identity=read_json(path.with_suffix('.identity.json'))
    if sha256_file(path)!=identity['config_sha256']:raise ValueError('Frozen config changed')
    output=Path(config['output_dir']);state=read(path.with_suffix('.state.json'));summary=read(output/'summary.json')
    alive=False
    if state and type(state.get('pid')) is int:
        process=Path('/proc')/str(state['pid'])/'cmdline'
        if process.is_file():
            args=process.read_bytes().decode(errors='replace').split('\0')
            alive=str(path) in args and any(Path(arg).name=='train_lora.py' for arg in args)
    metrics=[]
    if (output/'metrics.jsonl').is_file():
        for line in (output/'metrics.jsonl').read_text().splitlines():
            try:metrics.append(json.loads(line))
            except ValueError:pass
    training=[row for row in metrics if 'train_loss' in row]
    validation=[row for row in metrics if row.get('validation')]
    tracking=read(output/'wandb_run.json') or {}
    return {'name':path.stem,'status':'running' if alive else (summary or {}).get('status','not_started'),
        'pid':state.get('pid') if alive else None,'step':training[-1]['step'] if training else 0,
        'final_step':identity['plan']['stop_step'],'train_loss':training[-1]['train_loss'] if training else None,
        'latest_validation':validation[-1]['validation'] if validation else None,'wandb':tracking.get('url'),
        'best_exists':(output/'best.pt').is_file(),'last_exists':(output/'last.pt').is_file(),'output_dir':str(output)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config');args=parser.parse_args()
    if args.config:result=snapshot(args.config)
    else:
        launch=read_json(ROOT/'runs/assistant/system/latest_lora_experiments.json')
        result={'controller':read(Path(launch['state_file'])),'runs':[snapshot(item['config']) for item in launch['candidates']]}
    print(json.dumps(result,ensure_ascii=False,indent=2))
