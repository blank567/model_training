"""补传当前已结束的LoRA离线记录，并逐条读回云端更新步数验证完整性。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
ENTITY = 'mblankm-peking-university'
PROJECT = 'bilingual-training-lab'


def read(path):
    return json.loads(Path(path).read_text())


def verify_steps(source_steps, cloud_steps, expected_final):
    if not source_steps or max(source_steps) != expected_final:
        raise ValueError('Local history does not reach the completed training step')
    if list(source_steps) != list(cloud_steps):
        raise ValueError('Cloud history is incomplete or differs from the original optimizer-step sequence')
    return {'source_history_records': len(source_steps), 'cloud_history_records': len(cloud_steps),
            'cloud_optimizer_step': max(cloud_steps), 'history_sequence_verified': True}


def sync_current():
    import wandb
    from llm_lab.common.io import sha256_file, write_json
    from llm_lora.config import runtime_fingerprint
    launch = read(ROOT / 'runs/assistant/system/latest_lora_ablation.json')
    state = read(launch['state_file'])
    if state.get('status') != 'completed':
        raise ValueError('This command only syncs the completed current-ablation pipeline')
    candidates = launch['rank_candidates'] + list(state.get('prepared_dropouts', {}).values())
    if len(candidates) != 4 or len({item['config'] for item in candidates}) != 4:
        raise ValueError('Expected exactly the four current rank/dropout experiments')
    proxy = os.environ.get('WANDB_HTTPS_PROXY')
    if not proxy:
        raise ValueError('Source scripts/env_remote.sh after installing the W&B network profile')
    # 仅当前补传进程使用通用代理，不改变任何冻结实验的wandb_mode或身份。
    os.environ['HTTPS_PROXY'] = proxy
    os.environ['HTTP_PROXY'] = proxy
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    work = ROOT / 'runs/assistant/system' / ('wandb_backfill_' + stamp)
    work.mkdir(exist_ok=False)
    before = runtime_fingerprint()
    report = {'status': 'syncing', 'records': [], 'work_dir': str(work), 'training_started': False}
    write_json(work / 'report.json', report)
    for item in candidates:
        path = Path(item['config']).resolve()
        if not path.is_relative_to(ROOT / 'runs/configs'):
            raise ValueError('Unexpected frozen config path')
        identity = read(path.with_suffix('.identity.json'))
        if sha256_file(path) != identity['config_sha256']:
            raise ValueError('Frozen config changed before backfill')
        config = read(path)
        output = Path(config['output_dir'])
        summary = read(output / 'summary.json')
        tracking = read(output / 'wandb_run.json')
        if summary['status'] != 'completed' or summary['step'] != identity['plan']['stop_step']:
            raise ValueError('Only completely finished runs can be backfilled')
        if tracking['identity']['mode'] != 'offline':
            raise ValueError('Expected the existing offline records')
        folders = sorted((output / 'wandb').glob('offline-run-*'))
        if not folders or any(not p.resolve().is_relative_to(output) or p.is_symlink() for p in folders):
            raise ValueError('No safe offline run folders')
        print(json.dumps({'phase': 'syncing', 'key': item['key'], 'run_id': tracking['id']}, ensure_ascii=False), flush=True)
        log = work / (item['key'] + '.log')
        with log.open('wb') as stream:
            command = ['bash', str(ROOT / 'scripts/wandb_online.sh'), 'sync', '--yes', '--include-synced',
                       '--entity', ENTITY, '--project', PROJECT, *map(str, folders)]
            child = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        if child.returncode != 0:
            raise RuntimeError('W&B CLI failed; see ' + str(log))
        local_steps = [int(json.loads(line)['step']) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        verified = None
        last_error = None
        for attempt in range(5):
            try:
                remote = wandb.Api(timeout=30).run(f'{ENTITY}/{PROJECT}/{tracking["id"]}')
                cloud_steps = [int(row['optimizer_step']) for row in remote.scan_history(keys=['optimizer_step'], page_size=1000)]
                verified = verify_steps(local_steps, cloud_steps, summary['step'])
                if dict(remote.summary).get('optimizer_step') != summary['step']:
                    raise ValueError('Cloud summary does not reach the final step')
                break
            except Exception as error:
                verified = None
                last_error = error
                if attempt < 4: time.sleep(2)
        if verified is None:
            raise RuntimeError('Upload exit code alone is insufficient; cloud verification failed') from last_error
        if sha256_file(path) != identity['config_sha256']:
            raise ValueError('Frozen config changed while uploading')
        record = {'key': item['key'], 'run_id': tracking['id'], 'url': remote.url, 'cloud_state': remote.state,
                  'original_tracking_mode': 'offline', 'config_sha256_unchanged': True,
                  'log': str(log), **verified}
        write_json(output / 'cloud_sync.json', record)
        report['records'].append(record)
        write_json(work / 'report.json', report)
        print(json.dumps({'phase': 'cloud_verified', **record}, ensure_ascii=False), flush=True)
    if runtime_fingerprint() != before:
        raise ValueError('Frozen numerical runtime changed')
    report.update(status='completed', runtime_unchanged=True, runtime_sha256=before)
    write_json(work / 'report.json', report)
    write_json(ROOT / 'runs/assistant/system/latest_wandb_backfill.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(sync_current(), ensure_ascii=False, indent=2), flush=True)
