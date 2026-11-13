"""只读查看单轮 rank/dropout 队列和实际训练进度。"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def snapshot(item):
    path = Path(item['config'])
    cfg = read(path)
    output = Path(cfg['output_dir'])
    state = read(path.with_suffix('.state.json'))
    summary = read(output / 'summary.json')
    pid = state.get('pid')
    command = Path('/proc') / str(pid) / 'cmdline'
    alive = False
    if command.is_file():
        args = command.read_bytes().decode(errors='replace').split('\0')
        alive = str(path) in args and any(Path(arg).name == 'train_lora_one_epoch.py' for arg in args)
    latest = {}
    metrics = output / 'metrics.jsonl'
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if 'train_loss' in row:
                latest = row
    return {'key': item['key'], 'status': 'running' if alive else summary.get('status', 'not_started'),
            'pid': pid if alive else None, 'step': latest.get('step', summary.get('step', 0)),
            'target_steps': read(path.with_suffix('.identity.json')).get('plan', {}).get('stop_step'),
            'adapter': cfg['adapter'], 'epochs': cfg['train']['epochs'],
            'development_loss': (summary.get('latest_development') or {}).get('macro_loss'),
            'wandb': read(output / 'wandb_run.json').get('url') or read(output / 'cloud_sync.json').get('url'),
            'config': str(path)}


if __name__ == '__main__':
    launch = read(ROOT / 'runs/assistant/system/latest_lora_ablation.json')
    if not launch:
        raise SystemExit('No single-epoch LoRA ablation has been launched')
    state = read(Path(launch['state_file']))
    items = launch['rank_candidates'] + list(state.get('prepared_dropouts', {}).values())
    print(json.dumps({'controller': {key: value for key, value in state.items()
                                    if key not in {'completed', 'prepared_dropouts'}},
                      'runs': [snapshot(item) for item in items]}, ensure_ascii=False, indent=2))
