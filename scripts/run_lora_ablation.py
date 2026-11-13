"""串行完成单轮 rank 对照，再在胜出 rank 上研究 dropout；不使用测试集选型。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from llm_lab.common.io import read_json, sha256_file, write_json
from llm_lora.config import runtime_fingerprint
from llm_sft.config import plan_details
import train_lora_one_epoch as entry


def completed_result(summary, expected_steps):
    plan = summary.get('plan', {})
    if (summary.get('status') != 'completed' or summary.get('step') != expected_steps
            or plan.get('max_steps') != expected_steps or plan.get('stop_step') != expected_steps
            or plan.get('epochs') != 1 or plan.get('stop_epochs') is not None
            or summary.get('progress', {}).get('examples_seen') != plan.get('train_examples')):
        raise ValueError('Candidate must complete exactly one epoch before selection')
    development = summary.get('latest_development') or {}
    loss = development.get('macro_loss')
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise ValueError('Finite development loss is required; never select from a test set')
    return {'development_loss': loss, 'step': summary['step'],
            'primary_validation_loss': summary['latest_validation']['macro_loss'],
            'retention': summary.get('latest_retention'), 'best_step': summary['best_step'],
            'best_validation_loss': summary['best_validation_loss'],
            'adapter': summary['adapter'], 'budget': summary.get('budget'),
            'best_checkpoint': summary['best_checkpoint'], 'last_checkpoint': summary['last_checkpoint']}


def select(results, expected_count):
    if len(results) != expected_count or len({row['key'] for row in results}) != expected_count:
        raise ValueError('All requested candidates must finish before selection')
    for row in results:
        loss = row['development_loss']
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError('Invalid development loss')
    return min(results, key=lambda row: (row['development_loss'], row['key']))


def drive(rank_items, run_candidate, prepare_dropout, rank_selected):
    """先完成两组 rank，随后才为胜出 rank 创建两组独立 dropout 训练。"""
    ranks = [run_candidate(item, 'rank') for item in rank_items]
    winner = select(ranks, 2)
    rank_selected(winner, ranks)
    dropouts = [winner]
    for value in (0.05, 0.1):
        item = prepare_dropout(winner, value)
        dropouts.append(run_candidate(item, 'dropout'))
    return {'rank_results': ranks, 'selected_rank': winner['adapter']['rank'],
            'dropout_results': dropouts, 'selected': select(dropouts, 3)}


def validate_rank_configs(configs):
    if len(configs) != 2 or {cfg['adapter']['rank'] for cfg in configs} != {8, 32}:
        raise ValueError('Exactly ranks 8 and 32 are requested')
    shared = []
    for cfg in configs:
        entry.validate_one_epoch(cfg)
        adapter = cfg['adapter']
        if (adapter['alpha'] != 2 * adapter['rank'] or adapter['dropout'] != 0
                or adapter['target_scope'] != 'all' or cfg['train']['learning_rate'] != 2e-4):
            raise ValueError('Rank controls differ from the requested fixed learning rate and scale')
        value = deepcopy(cfg)
        for key in ('adapter', 'experiment', 'output_dir'):
            value.pop(key, None)
        value['train'].pop('wandb_run_name', None)
        shared.append(value)
    if shared[0] != shared[1]:
        raise ValueError('Rank candidates differ in another control variable')


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    root = Path(plan['project_root']).resolve()
    directory = Path(plan['controller_dir']).resolve()
    if not directory.is_relative_to(root / 'runs/assistant/system'):
        raise ValueError('Controller records must remain in runs/assistant/system')
    if (sha256_file(Path(__file__)) != plan['controller_sha256']
            or sha256_file(entry.ENTRY) != plan['entry_sha256']
            or runtime_fingerprint() != plan['runtime_sha256']):
        raise ValueError('Frozen runtime or ablation entry changed')
    configs = [entry.frozen_config(item['config']) for item in plan['rank_candidates']]
    validate_rank_configs(configs)
    if plan.get('dropout_values') != [0.05, 0.1]:
        raise ValueError('Unexpected dropout search scope')
    state_path = directory / 'state.json'
    previous = read_json(state_path) if state_path.is_file() else {}
    state = {'status': 'starting', 'controller_pid': os.getpid(), 'current_pid': None,
             'completed': [], 'prepared_dropouts': previous.get('prepared_dropouts', {}),
             'cost_before_cny': previous.get('cost_before_cny')}
    active = None
    stopped = False
    handlers = {}

    def save():
        state['updated_unix_seconds'] = time.time()
        write_json(state_path, state)

    def total_cost():
        path = root / 'runs/budget_ledger.json'
        ledger = read_json(path) if path.is_file() else {'runs': {}}
        return sum(float(row['estimated_compute_cny']) for row in ledger['runs'].values())

    if state['cost_before_cny'] is None:
        state['cost_before_cny'] = total_cost()

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if active is not None:
            try:
                active.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    def child(action, item, timeout):
        nonlocal active
        if stopped:
            raise RuntimeError('Pipeline stopped; no subsequent candidate will start')
        if total_cost() - state['cost_before_cny'] >= plan['pipeline_compute_cap_cny']:
            raise RuntimeError('Pipeline compute cap reached')
        command = [str(root / '.venv/bin/python'), '-u', str(entry.ENTRY), action, '--config', item['config']]
        Path(item['log']).parent.mkdir(parents=True, exist_ok=True)
        with Path(item['log']).open('ab') as stream:
            active = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state['current_pid'] = active.pid
            save()
            deadline = time.monotonic() + timeout
            while True:
                try:
                    code = active.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    capped = total_cost() - state['cost_before_cny'] >= plan['pipeline_compute_cap_cny']
                    if not stopped and not capped and time.monotonic() < deadline:
                        save()
                        continue
                    active.send_signal(signal.SIGTERM)
                    try:
                        active.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        active.kill()
                        active.wait(timeout=30)
                    code = 124
                    break
        active = None
        state['current_pid'] = None
        save()
        if code != 0 or stopped:
            raise RuntimeError(f'{item["key"]}: {action} interrupted or failed (exit {code})')

    def candidate(item, phase):
        if stopped:
            raise RuntimeError('Pipeline stopped')
        cfg = entry.frozen_config(item['config'])
        details = plan_details(cfg)
        if details['max_steps'] != details['stop_step']:
            raise ValueError('Partial multi-epoch schedule is forbidden')
        output = Path(cfg['output_dir'])
        summary_path = output / 'summary.json'
        state.update(status=phase, current_candidate=item['key'])
        save()
        summary = read_json(summary_path) if summary_path.is_file() else {}
        if summary.get('status') != 'completed':
            child('check', item, 180)
            action = 'resume' if (output / 'last.pt').is_file() else 'start'
            child(action, item, cfg['budget']['max_runtime_seconds'] + 600)
            summary = read_json(summary_path)
        result = {**item, **completed_result(summary, details['stop_step'])}
        result['adapter'] = {**result['adapter'], **cfg['adapter']}
        if not all(Path(result[key]).is_file() for key in ('best_checkpoint', 'last_checkpoint')):
            raise ValueError('Completed candidate checkpoints are missing')
        state['completed'].append(result)
        save()
        return result

    def rank_selected(winner, results):
        selection = {'selection_metric': 'development macro loss at completed one epoch',
                     'historical_r16_excluded_due_to_different_schedule': True,
                     'test_used_for_selection': False, 'rank_results': results, 'selected': winner}
        write_json(directory / 'rank_selection.json', selection)
        state['selected_rank'] = winner['adapter']['rank']
        save()

    def prepare_dropout(winner, dropout):
        key = f'r{winner["adapter"]["rank"]}_ld{dropout:g}'
        if key in state['prepared_dropouts']:
            return state['prepared_dropouts'][key]
        cfg = deepcopy(entry.frozen_config(winner['config']))
        cfg['adapter']['dropout'] = dropout
        cfg['train']['wandb_run_name'] = None
        cfg['experiment'].update(phase='lora_dropout_one_epoch', dropout=dropout,
                                 rank_selection=str(directory / 'rank_selection.json'))
        template = directory / 'templates' / (key + '.json')
        write_json(template, cfg)
        ready = entry.prepare(template, plan['stamp'])
        item = {'key': key, 'config': ready['RUN_CONFIG'], 'output_dir': ready['RUN_DIR'],
                'log': ready['RUN_LOG'], 'env': ready['env'], 'prepared': ready}
        state['prepared_dropouts'][key] = item
        save()
        return item

    import fcntl
    with (directory / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, stop)
        try:
            save()
            result = drive(plan['rank_candidates'], candidate, prepare_dropout, rank_selected)
            comparison = {'status': 'completed', 'epochs': 1, 'stop_epochs': None,
                          'learning_rate': 2e-4, 'test_used_for_selection': False,
                          'selection_metric': 'development macro loss at completed one epoch', **result}
            write_json(directory / 'comparison.json', comparison)
            state.update(status='completed', comparison=str(directory / 'comparison.json'),
                         selected=result['selected']['key'])
        except BaseException as error:
            state.update(status='stopped' if stopped else 'failed', error_type=type(error).__name__, reason=str(error))
            raise
        finally:
            save()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan), ensure_ascii=False), flush=True)
