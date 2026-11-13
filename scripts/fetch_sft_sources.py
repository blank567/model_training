"""固定版本 SFT 文件下载；仅写专属缓存，保留可恢复 partial 和磁盘水位。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.request

from llm_lab.data.acquire_stream import _network_openers, _resolve_target


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(root: Path, source: dict, reserve: int) -> Path:
    """URL、大小、摘要均来自冻结配置；签名重定向地址不进入日志。"""
    destination=(root/source['cache_path']).resolve()
    cache_root=(root/'.cache/data/sft_300m').resolve()
    if not destination.is_relative_to(cache_root) or destination.is_symlink():
        raise ValueError('SFT cache path escapes the private cache directory')
    if not re.fullmatch('[a-f0-9]{64}',source['sha256']):
        raise ValueError('Expected a full source SHA256')
    if not source['url'].startswith(('https://huggingface.co/datasets/','https://raw.githubusercontent.com/')):
        raise ValueError('Unsupported public SFT source host')
    destination.parent.mkdir(parents=True,exist_ok=True)
    size=int(source['size_bytes'])
    if destination.exists():
        if destination.stat().st_size!=size or sha256(destination)!=source['sha256']:
            raise ValueError('Existing SFT source identity mismatch: '+source['name'])
        return destination
    partial=destination.with_name(destination.name+'.part')
    if partial.is_symlink():
        raise ValueError('Symlink partial file is not allowed')
    metadata,direct=_network_openers()
    last_error=None
    for attempt in range(3):
        received=partial.stat().st_size if partial.exists() else 0
        if received>size:
            raise ValueError('SFT partial file exceeds expected size')
        if shutil.disk_usage(destination.parent).free < reserve+size-received:
            raise OSError('SFT download would violate disk reserve')
        try:
            if received<size:
                target=_resolve_target(metadata,source['url'],source['sha256'],size)
                headers={'User-Agent':'llm-lab-sft-data/1'}
                if received:
                    headers['Range']=f'bytes={received}-'
                with direct.open(urllib.request.Request(target,headers=headers),timeout=45) as response:
                    if received and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {received}-')):
                        raise ValueError('SFT server refused exact Range resume')
                    with partial.open('ab' if received else 'wb') as stream:
                        while True:
                            chunk=response.read(min(1024*1024,size-received+1))
                            if not chunk:
                                break
                            received+=len(chunk)
                            if received>size:
                                raise ValueError('SFT source exceeded pinned size')
                            stream.write(chunk)
                        stream.flush();os.fsync(stream.fileno())
            if partial.stat().st_size!=size or sha256(partial)!=source['sha256']:
                raise ValueError('SFT source size/SHA verification failed')
            os.replace(partial,destination)
            print(json.dumps({'phase':'source_verified','source':source['name'],'bytes':size}),flush=True)
            return destination
        except (OSError,ValueError,urllib.error.URLError) as exc:
            last_error=exc
            print(json.dumps({'phase':'source_retry','source':source['name'],'attempt':attempt+1,'error_type':type(exc).__name__}),flush=True)
            time.sleep(1+attempt)
    raise RuntimeError('Unable to verify SFT source '+source['name']+' ('+type(last_error).__name__+')')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    settings=json.loads(args.config.read_text(encoding='utf-8'))
    root=(args.config.resolve().parent/settings['project_root']).resolve()
    for source in settings['sources']:
        fetch_file(root,source,int(settings['min_free_bytes']))
