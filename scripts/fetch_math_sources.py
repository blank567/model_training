"""下载固定提交的公开数学题原始数据；训练与官方测试来源分别登记。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def download(url, destination, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError('Refusing a symlink download target')
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'bilingual-training-lab-math-data/1'})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(24 * 1024**2 + 1)
            if len(data) > 24 * 1024**2 or expected_size is not None and len(data) != expected_size:
                raise ValueError('Public source size does not match the pinned repository tree')
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError('Existing source differs; do not overwrite a previous snapshot')
            if not destination.exists():
                temporary = destination.with_name(destination.name + '.part')
                temporary.write_bytes(data)
                temporary.replace(destination)
            return {'url': url, 'path': destination.as_posix(), 'size_bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()}
        except (OSError, ValueError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f'Could not fetch pinned source: {destination.name}') from error


def fetch(root, discovery):
    root = Path(root).resolve()
    discovered = json.loads(Path(discovery).read_text(encoding='utf-8'))
    sources = []
    jobs = []
    for source in discovered:
        if source['repo'] not in {'openai/grade-school-math', 'math-eval/TAL-SCQ5K'}:
            raise ValueError('Unexpected public repository')
        folder = root / source['repo'].replace('/', '__')
        for spec in source['files']:
            name = spec['path']
            if 'socratic' in name or 'example_model' in name:
                continue
            url = f'https://raw.githubusercontent.com/{source["repo"]}/{source["revision"]}/{name}'
            destination = folder / name
            if not destination.resolve().is_relative_to(root):
                raise ValueError('Source path escaped the dedicated cache')
            jobs.append((source, spec, url, destination))
    def task(job):
        source, spec, url, destination = job
        result = download(url, destination, spec['size'])
        result.update(repository=source['repo'], revision=source['revision'], upstream_path=spec['path'],
                      role='external_test_only' if 'test' in spec['path'] else
                           'training_reserve_only' if 'train' in spec['path'] else 'documentation')
        result['path'] = destination.relative_to(root).as_posix()
        return result
    with ThreadPoolExecutor(max_workers=3) as executor:
        sources = list(executor.map(task, jobs))
    # TAL 原仓库没有单列 LICENSE，另保留发布者的 HF 数据卡与其精确内容摘要。
    card = download('https://huggingface.co/datasets/math-eval/TAL-SCQ5K/raw/main/README.md', root / 'tal_hf_dataset_card.md')
    card['path'] = 'tal_hf_dataset_card.md'
    lock = {'schema_version': 1, 'kind': 'pinned_math_public_sources', 'sources': sources,
            'tal_dataset_card': card,
            'licenses': {'openai/grade-school-math': 'MIT; see cached LICENSE',
                         'math-eval/TAL-SCQ5K': 'MIT as declared in cached publisher dataset card; original GitHub repository has no separate LICENSE'},
            'training_is_automatic': False}
    lock_path = root / 'sources.lock.json'
    data = json.dumps(lock, ensure_ascii=False, indent=2) + '\n'
    if lock_path.exists() and lock_path.read_text(encoding='utf-8') != data:
        raise ValueError('Source lock already exists with a different identity')
    lock_path.write_text(data, encoding='utf-8', newline='\n')
    return {'source_lock': str(lock_path), 'files': len(sources) + 1,
            'total_bytes': sum(item['size_bytes'] for item in sources) + card['size_bytes']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--discovery', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.root, args.discovery), ensure_ascii=False, indent=2))
