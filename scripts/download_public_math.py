"""下载固定官方版本的数学训练候选；校验文件对象，不执行数据中的代码。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


SOURCES = [
    {'source': 'meta-math/MetaMathQA', 'revision': 'aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18',
     'path': 'MetaMathQA-395K.json', 'bytes': 395626321,
     'sha256': 'fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725',
     'language': 'en', 'license': 'MIT as declared by publisher', 'role': 'training_candidates'},
    {'source': 'meta-math/MetaMathQA', 'revision': 'aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18',
     'path': 'README.md', 'bytes': 4445, 'git_blob': '606bfd3edf6fe30bbc0c9b335435f4d9048d3a53', 'role': 'documentation'},
    {'source': 'THUDM/MathGLM', 'revision': '658d1e2d28a8666d54b7797fbc4484d95fc24e4f',
     'path': 'MathGLM_MWP/dataset/data.jsonl', 'bytes': 45400015,
     'git_blob': 'c83bf2334ab08e04c0cb144d6eca6ca764421e08', 'language': 'zh',
     'license': 'No separate LICENSE in pinned repository tree; preserve original provenance',
     'role': 'training_candidates_require_split_audit'},
    {'source': 'THUDM/MathGLM', 'revision': '658d1e2d28a8666d54b7797fbc4484d95fc24e4f',
     'path': 'MathGLM_MWP/load_dataset.py', 'bytes': 3213,
     'git_blob': 'f75fd12356bf57d7677003381ff45d687d955bb1', 'role': 'split_documentation_not_executable'},
    {'source': 'THUDM/MathGLM', 'revision': '658d1e2d28a8666d54b7797fbc4484d95fc24e4f',
     'path': 'README.md', 'bytes': 5865, 'git_blob': '65d3cc1e6219e503f61f6f4f521273674cc5471e', 'role': 'documentation'},
]

BROAD_SOURCES = [
    {'source': 'TIGER-Lab/MathInstruct', 'revision': 'b4fdc323a7be1379c9c7c0b67b1de72dfee2111a',
     'path': 'MathInstruct.json', 'bytes': 212488891,
     'sha256': '15b8be9824c31d619078f43baf67f7243c53f475ae4de5f8346b6d3ea9c269dd',
     'language': 'en', 'license': 'Per-subset licenses in publisher README', 'role': 'broad_math_training_candidates'},
    {'source': 'TIGER-Lab/MathInstruct', 'revision': 'b4fdc323a7be1379c9c7c0b67b1de72dfee2111a',
     'path': 'README.md', 'bytes': 2716, 'git_blob': '41aa2f31b98faeda95ccf1a76e5216b2f9e4c258', 'role': 'documentation'},
    {'source': 'ALmonster/MathInstruct-Chinese', 'revision': '4831ef47453915b7750672d4f0e6c0b5096d26f9',
     'path': 'mathinstruct-zh.parquet', 'bytes': 93189722,
     'sha256': '997403a204bfecf1a7aab333c5359067680ae58a38363ae9f566ac0d1cde93cd',
     'language': 'zh', 'license': 'Per-subset upstream licenses; public LLM translation, not human annotation',
     'role': 'broad_math_training_candidates'},
    {'source': 'ALmonster/MathInstruct-Chinese', 'revision': '4831ef47453915b7750672d4f0e6c0b5096d26f9',
     'path': 'README.md', 'bytes': 814, 'git_blob': '2a67f9fc2d4ee1d7b72cb77bd1545c0d2c7a216a', 'role': 'documentation'},
]


def digest_file(path):
    sha, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            sha.update(block)
            blob.update(block)
    return sha.hexdigest(), blob.hexdigest()


def fetch(spec, cache, proxy):
    folder = cache / spec['source'].replace('/', '__')
    destination = folder / spec['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError('Symlink source target refused')
    host = 'https://raw.githubusercontent.com' if spec['source'] == 'THUDM/MathGLM' else 'https://huggingface.co/datasets'
    branch = '/' if spec['source'] == 'THUDM/MathGLM' else '/resolve/'
    url = f"{host}/{spec['source']}{branch}{spec['revision']}/{spec['path']}"
    started = time.monotonic()
    if not destination.exists():
        partial = destination.with_name(destination.name + '.part')
        command = ['curl.exe' if __import__('os').name == 'nt' else 'curl', '--fail', '--location',
                   '--silent', '--show-error', '--retry', '2', '--retry-all-errors', '--connect-timeout', '15',
                   '--max-time', '900', '--output', str(partial), url]
        if proxy:
            command += ['--proxy', proxy]
        subprocess.run(command, check=True, timeout=2800)
        path = partial
    else:
        path = destination
    if path.stat().st_size != spec['bytes']:
        raise ValueError('Pinned size mismatch: ' + spec['path'])
    sha, blob = digest_file(path)
    if spec.get('sha256', sha) != sha or spec.get('git_blob', blob) != blob:
        raise ValueError('Pinned content checksum mismatch: ' + spec['path'])
    if path != destination:
        path.replace(destination)
    result = {**spec, 'local_path': destination.relative_to(cache).as_posix(), 'url': url,
              'sha256': sha, 'download_seconds': time.monotonic() - started}
    print(json.dumps({'downloaded': result['local_path'], 'bytes': spec['bytes'], 'sha256': sha}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--proxy')
    ap.add_argument('--profile', choices=['broad', 'reserve'], default='broad')
    ap.add_argument('--catalog', type=Path, help='Explicit pinned source catalog; never execute upstream dataset scripts')
    args = ap.parse_args()
    cache = args.cache.resolve()
    if __import__('os').name == 'nt' and cache.drive.upper() != 'E:':
        raise ValueError('All local data must remain on E:')
    cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache).free < 2 * 2**30:
        raise RuntimeError('Reserve at least 2 GiB before source downloads')
    results = []
    selected = json.loads(args.catalog.read_text(encoding='utf-8')) if args.catalog else (BROAD_SOURCES if args.profile=='broad' else SOURCES)
    allowed={'meta-math/MetaMathQA','THUDM/MathGLM','TIGER-Lab/MathInstruct','ALmonster/MathInstruct-Chinese',
             'project-numina/NuminaMath-1.5','ecnu-icalk/cmm-math','ALmonster/MATH-Hard-Chinese',
             'Mxode/Math-Chinese-DeepSeek-R1-10K','Mxode/CMID-Chinese_Math_Instruct_Dataset',
             'a-m-team/AM-DeepSeek-R1-Distilled-1.4M','opencsg/Fineweb-Edu-Chinese-V3'}
    for spec in selected:
        relative=Path(spec['path'])
        if spec['source'] not in allowed or relative.is_absolute() or '..' in relative.parts or not 0<spec['bytes']<600*2**20:
            raise ValueError('Unexpected public math source/path/size')
        if not __import__('re').fullmatch('[0-9a-f]{40}',spec['revision']):raise ValueError('A fixed commit is required')
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(fetch, spec, cache, args.proxy) for spec in selected]
        for job in as_completed(pending):
            results.append(job.result())
    receipt = {'schema_version': 1, 'kind': 'pinned_public_math_training_candidates',
               'sources': sorted(results, key=lambda row: row['local_path']),
               'training_started': False, 'passed_curated_dataset_acceptance': False,
               'note': 'Public publisher-provided sources, not a guarantee of correct reasoning or answers'}
    target = cache/'sources.lock.json'
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if [(r['local_path'], r['sha256']) for r in old['sources']] != [(r['local_path'], r['sha256']) for r in receipt['sources']]:
            raise ValueError('Existing source lock differs')
    else:
        target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'downloaded_verified', 'files': len(results),
                      'bytes': sum(row['bytes'] for row in results), 'source_lock': str(target)}), flush=True)


if __name__ == '__main__':
    main()
