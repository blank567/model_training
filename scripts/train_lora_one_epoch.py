"""单轮 LoRA 实验入口：复用冻结引擎，独立约束轮数并在名称中标注 dropout。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from llm_lab.common.io import read_json, sha256_file
from llm_lora.config import read_config
import train_lora as original


ENTRY = Path(__file__).resolve()
BASE_NAME_FOR = original.name_for


def validate_one_epoch(config):
    train = config['train']
    if isinstance(train['epochs'], bool) or train['epochs'] != 1 or train.get('stop_epochs') is not None:
        raise ValueError('Only epochs=1 and stop_epochs=null are allowed')
    if config['experiment'].get('one_epoch_entry_sha256') != sha256_file(ENTRY):
        raise ValueError('One-epoch entry source identity changed')


def name_for(config, stamp):
    name = BASE_NAME_FOR(config, stamp)
    scope = config['adapter']['target_scope']
    dropout = original.legacy._number_token(config['adapter']['dropout'])
    return name.replace(f'_{scope}_lr', f'_{scope}_ld{dropout}_lr', 1)


@contextmanager
def named_experiment():
    # 只在本进程调用期间扩展新实验的命名，不修改旧入口文件或历史身份。
    before = original.name_for
    original.name_for = name_for
    try:
        yield
    finally:
        original.name_for = before


def prepare(template, stamp=None):
    validate_one_epoch(read_config(template))
    with named_experiment():
        return original.prepare(template, stamp)


def frozen_config(path):
    with named_experiment():
        config = original.frozen_config(path)
    validate_one_epoch(config)
    return config


def execute(action, path):
    frozen_config(path)
    with named_experiment():
        return original.execute(action, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    item = commands.add_parser('prepare')
    item.add_argument('--template', required=True)
    item.add_argument('--stamp')
    for action in ('check', 'start', 'resume', 'status'):
        commands.add_parser(action).add_argument('--config', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.template, args.stamp)
    elif args.action == 'status':
        config = frozen_config(args.config)
        path = Path(config['output_dir']) / 'summary.json'
        result = read_json(path) if path.is_file() else {'status': 'not_started'}
    else:
        result = execute(args.action, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
