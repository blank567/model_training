"""固定 HF 语料分片下载：TLS 验证、官方 SHA、Range 恢复和磁盘余量。"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import pathlib
import re
import shutil
import time
import urllib.error
import urllib.request

from llm_lab.data.acquire_stream import _network_openers, _resolve_target, _https_url


def fetch_shard(source, file_spec, settings, root):
    root = pathlib.Path(root).resolve()
    cache = (root / settings.get('cache_dir', '.cache/data/expansion_2b')).resolve()
    if not cache.is_relative_to(root) or cache.is_symlink():
        raise ValueError('Download cache must remain under project root')
    cache.mkdir(parents=True, exist_ok=True)
    digest = file_spec['sha256']
    size = int(file_spec.get('size_bytes', file_spec.get('size', 0)))
    revision, repository = source['revision'], source['repository']
    relative = file_spec['path']
    if not re.fullmatch(r'[0-9a-f]{64}', digest) or size <= 0:
        raise ValueError('Invalid pinned shard identity')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Full commit SHA is required')
    if repository not in {'HuggingFaceFW/fineweb-2','HuggingFaceFW/fineweb-edu'}:
        raise ValueError('Unapproved dataset repository')
    if relative.startswith('/') or '..' in pathlib.PurePosixPath(relative).parts:
        raise ValueError('Invalid repository-relative file path')
    destination = cache / (digest+'.parquet')
    partial = destination.with_suffix('.part')
    for p in (destination,partial):
        if p.is_symlink() or (p.exists() and not p.is_file()):
            raise ValueError('Expected regular cache file')
    reserve = int(settings.get('min_free_bytes',12*1024**3))
    def room(incoming=0):
        if shutil.disk_usage(cache).free < reserve+incoming:
            raise OSError('Disk reserve reached; safe resumable stop')
    def digest_file(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024**2),b''): h.update(chunk)
        return h.hexdigest()
    output=(root/settings['output_dir']).resolve()
    if not output.is_relative_to(root): raise ValueError('Unsafe progress directory')
    output.mkdir(parents=True, exist_ok=True)
    def progress(received,phase,attempt=0):
        record={'phase':phase,'repository':repository,'revision':revision,
                'shard':relative,'sha256':digest,'received_bytes':received,
                'size_bytes':size,'attempt':attempt,'time_unix':time.time(),
                'free_bytes':shutil.disk_usage(cache).free}
        temp=output/'download_progress.json.tmp'
        temp.write_text(json.dumps(record),encoding='utf-8')
        os.replace(temp,output/'download_progress.json')
    if destination.exists():
        if destination.stat().st_size != size or digest_file(destination) != digest:
            raise ValueError('Existing shard size/SHA mismatch')
        progress(size,'verified_cached')
        return destination
    metadata,direct = _network_openers()
    url=f'https://huggingface.co/datasets/{repository}/resolve/{revision}/{relative}'
    for attempt in range(1,int(settings.get('download_attempts',5))+1):
        received=partial.stat().st_size if partial.exists() else 0
        if received > size: raise ValueError('Partial exceeds pinned size')
        room(size-received)
        hasher=hashlib.sha256()
        if received:
            with partial.open('rb') as f:
                for chunk in iter(lambda:f.read(4*1024**2),b''):hasher.update(chunk)
        try:
            progress(received,'resolving',attempt)
            if received < size:
                target=_resolve_target(metadata,url,digest,size)
                headers={'User-Agent':'llm-lab-corpus/1'}
                if received:headers['Range']=f'bytes={received}-'
                with direct.open(urllib.request.Request(target,headers=headers),timeout=60) as response:
                    _https_url(response.geturl())
                    if received:
                        if response.status!=206 or not re.fullmatch(rf'bytes {received}-\d+/{size}',response.headers.get('Content-Range','')):
                            raise ValueError('Server rejected exact Range; partial retained')
                    elif response.status not in {200,206}:
                        raise ValueError('Unexpected download status')
                    with partial.open('ab' if partial.exists() else 'xb') as f:
                        last=time.monotonic()
                        while True:
                            chunk=response.read(min(1024**2,size-received+1))
                            if not chunk:break
                            if received+len(chunk)>size:raise ValueError('Response exceeds pinned file size')
                            room(len(chunk)); f.write(chunk); hasher.update(chunk);received+=len(chunk)
                            if time.monotonic()-last>=3:
                                progress(received,'downloading',attempt);last=time.monotonic()
                        f.flush();os.fsync(f.fileno())
            if received!=size or hasher.hexdigest()!=digest:
                raise ValueError('Downloaded size/SHA mismatch; partial retained for audit')
            os.replace(partial,destination);progress(size,'verified',attempt)
            return destination
        except (urllib.error.URLError,TimeoutError,ConnectionError,OSError,http.client.HTTPException) as exc:
            progress(received,'retry_'+type(exc).__name__,attempt)
            if attempt>=int(settings.get('download_attempts',5)):
                raise RuntimeError('Dataset download failed; resumable partial preserved') from None
            time.sleep(min(5*attempt,20))
    raise AssertionError('unreachable')
