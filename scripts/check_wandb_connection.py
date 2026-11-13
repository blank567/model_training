"""验证服务器W&B认证，可选写入一次CPU连通性探针并读回云端结果。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import netrc
import os
from pathlib import Path
from urllib.parse import urlsplit


def credential(base_url):
    value = os.environ.get('WANDB_API_KEY')
    if value:
        return value
    try:
        account = netrc.netrc(os.environ.get('NETRC')).authenticators(urlsplit(base_url).hostname)
        return account[2] if account else None
    except (OSError, netrc.NetrcParseError):
        return None


def check(write_test=False):
    import requests
    base_url = os.environ.get('WANDB_BASE_URL', 'https://api.wandb.ai').rstrip('/')
    if base_url not in {'https://api.wandb.ai', 'https://api.forge.coreweave.com'}:
        raise ValueError('Only the configured official W&B cloud API is allowed by this diagnostic')
    key = credential(base_url)
    if not key:
        raise RuntimeError('No server-side W&B credentials found; log in in the remote terminal')
    proxy = os.environ.get('WANDB_HTTPS_PROXY')
    session = requests.Session()
    session.trust_env = False
    response = session.post(base_url + '/graphql', auth=('api', key),
                            json={'query': 'query { viewer { id } }'},
                            proxies={'https': proxy} if proxy else None, timeout=(5, 20))
    if response.status_code != 200:
        raise RuntimeError(f'W&B authentication endpoint returned HTTP {response.status_code}')
    payload = response.json()
    if not (payload.get('data') or {}).get('viewer'):
        raise RuntimeError('W&B did not validate the current server-side credential')
    result = {'status': 'authenticated', 'http_status': 200, 'credential_verified': True,
              'proxy_configured': bool(proxy), 'api_key_exported': False, 'gpu_training_started': False}
    if write_test:
        import wandb
        root = Path(__file__).resolve().parents[1]
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        output = root / 'runs/assistant/system' / ('wandb_online_check_' + stamp)
        output.mkdir(parents=True, exist_ok=False)
        run = wandb.init(project='bilingual-training-lab', entity='mblankm-peking-university',
                         name='assistant_wandb_connectivity_' + stamp, job_type='connectivity-check',
                         tags=['assistant', 'network-check'], mode='online', dir=str(output), save_code=False,
                         config={'purpose': 'W&B network repair verification', 'model_training': False},
                         settings=wandb.Settings(disable_code=True, disable_git=True, console='off',
                                                x_disable_stats=True, init_timeout=45, finish_timeout=20))
        run.define_metric('optimizer_step')
        run.define_metric('*', step_metric='optimizer_step')
        run.log({'optimizer_step': 0, 'connectivity_probe': 0.})
        run.log({'optimizer_step': 1, 'connectivity_probe': 1.})
        identifier, url = run.id, run.url
        run.finish()
        # Public API也使用同一代理；仅影响当前诊断进程，不修改全局下载环境。
        if proxy:
            os.environ['HTTPS_PROXY'] = proxy
            os.environ['HTTP_PROXY'] = proxy
        remote = wandb.Api(timeout=30).run('mblankm-peking-university/bilingual-training-lab/' + identifier)
        summary = dict(remote.summary)
        if summary.get('connectivity_probe') != 1 or summary.get('optimizer_step') != 1:
            raise RuntimeError('Online probe was not confirmed by reading its cloud summary')
        result.update(status='online_write_and_read_verified', url=url, run_id=identifier,
                      cloud_optimizer_step=summary['optimizer_step'], output_dir=str(output))
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.write_test), ensure_ascii=False, indent=2))
