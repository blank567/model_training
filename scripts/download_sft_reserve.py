"""限量下载固定版本SFT候选原始文件；不发布训练集，不触碰训练源码和权重。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time
import urllib.request

from llm_lab.data.acquire_stream import _network_openers, _resolve_target

GIB = 1024**3


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def private_path(path, parent):
    """所有写入限制在本次候选储备内部，拒绝符号链接和路径穿越。"""
    path, parent = Path(path), Path(parent).resolve()
    if not path.is_absolute():
        path = parent/path
    for current in (path,*path.parents):
        if current.is_symlink():
            raise ValueError('Reserve path cannot traverse a symlink')
    target = path.resolve()
    if target == parent or not target.is_relative_to(parent):
        raise ValueError('Reserve path escapes its dedicated directory')
    return target


def validate(config, root):
    directory = private_path(Path(root)/config['reserve_dir'],Path(root)/'artifacts/posttrain')
    if config.get('kind') != 'sft_candidate_raw_reserve' or config.get('training_ready') is not False:
        raise ValueError('Only unapproved raw candidate reserves can be downloaded')
    if not 24*GIB <= config['min_free_bytes'] <= 64*GIB:
        raise ValueError('Concurrent training needs at least 24GiB free')
    if not 0 < config['max_total_source_bytes'] <= 2*GIB or not 0 < config['bytes_per_second'] <= 8*1024**2:
        raise ValueError('Reserve size or download rate exceeds this task budget')
    if not 0 < config['max_runtime_seconds'] <= 1800:
        raise ValueError('Reserve runtime is limited to 30 minutes')
    if not config['files'] or sum(row['size_bytes'] for row in config['files']) > config['max_total_source_bytes']:
        raise ValueError('Planned source bytes exceed reserve size budget')
    seen = set()
    for row in config['files']:
        if not re.fullmatch(r'[a-f0-9]{40}',row['revision']) or not re.fullmatch(r'[a-f0-9]{64}',row['sha256']):
            raise ValueError('Full immutable revision and SHA256 are required')
        if type(row['size_bytes']) is not int or row['size_bytes'] <= 0:
            raise ValueError('Source size must be positive')
        if row['upstream_split'] not in {'train_or_unsplit_source','unsplit_source'}:
            raise ValueError('Only upstream training or unsplit sources are allowed')
        if any(word in row['file'].casefold() for word in ('test-','validation-','/test/','/validation/')):
            raise ValueError('Do not download public evaluation files into training reserves')
        allowed = (f"https://huggingface.co/datasets/{row['repository']}/resolve/{row['revision']}/",
                   f"https://raw.githubusercontent.com/{row['repository']}/{row['revision']}/")
        if not row['url'].startswith(allowed) or not row['url'].endswith('/'+row['file']):
            raise ValueError('Source URL does not identify its pinned official file')
        path = private_path(row['destination'],directory)
        if not path.is_relative_to(directory/'raw') or path in seen:
            raise ValueError('Duplicate or non-raw source destination')
        seen.add(path)
    return directory


def fetch(row, directory, reserve, rate, progress, check_stop):
    """按官方HEAD解析下载目标，支持精确Range续传；SHA通过才提交原始文件。"""
    destination = private_path(row['destination'],directory)
    destination.parent.mkdir(parents=True,exist_ok=True)
    size = row['size_bytes']
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != size or digest(destination) != row['sha256']:
            raise ValueError('Existing source content differs from pinned identity')
        return destination
    partial = destination.with_name(destination.name+'.part')
    private_path(partial,directory)
    metadata, direct = _network_openers()
    if row['url'].startswith('https://raw.githubusercontent.com/'):
        return fetch_github_ranges(row,destination,partial,metadata,reserve,rate,progress,check_stop)
    for attempt in range(3):
        check_stop()
        received = partial.stat().st_size if partial.exists() else 0
        if received > size:
            raise ValueError('Partial exceeds pinned source size')
        if shutil.disk_usage(destination.parent).free < reserve+size-received:
            raise OSError('Reserve download would use space reserved for training')
        try:
            if received < size:
                target = _resolve_target(metadata,row['url'],row['sha256'],size)
                headers = {'User-Agent':'llm-lab-sft-reserve/1'}
                if received:
                    headers['Range'] = f'bytes={received}-'
                with direct.open(urllib.request.Request(target,headers=headers),timeout=45) as response:
                    if received and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {received}-')):
                        raise ValueError('Exact Range resume was not honored')
                    started, initial, reported = time.monotonic(), received, received
                    with partial.open('ab' if received else 'wb') as stream:
                        while True:
                            check_stop()
                            chunk = response.read(min(1024**2,size-received+1))
                            if not chunk:
                                break
                            if received+len(chunk) > size:
                                raise ValueError('Remote body exceeds pinned size')
                            if shutil.disk_usage(destination.parent).free < reserve+size-received:
                                raise OSError('Training disk reserve reached during download')
                            stream.write(chunk);received += len(chunk)
                            delay = (received-initial)/rate-(time.monotonic()-started)
                            if delay > 0:
                                time.sleep(delay)
                            if received-reported >= 16*1024**2 or received == size:
                                progress(row,received);reported = received
                        stream.flush();os.fsync(stream.fileno())
            if partial.stat().st_size != size or digest(partial) != row['sha256']:
                raise ValueError('Downloaded size or SHA256 verification failed')
            partial.replace(destination)
            return destination
        except InterruptedError:
            raise
        except OSError as error:
            # 不打印签名地址、代理或凭据，完整partial保留供同一配置续传。
            print(json.dumps({'event':'download_retry','source':row['source'],'file':row['file'],
                              'attempt':attempt+1,'error_type':type(error).__name__}),flush=True)
            if attempt == 2:
                raise RuntimeError('Source download failed after three attempts') from None
            time.sleep(1+attempt)
    raise RuntimeError('Source download did not finish')


def fetch_github_ranges(row,destination,partial,opener,reserve,rate,progress,check_stop):
    """GitHub原始大文件按4MiB区间获取，避免平台上的HEAD或整文件请求挂起。"""
    size = row['size_bytes']
    received = partial.stat().st_size if partial.exists() else 0
    if received > size:
        raise ValueError('Partial exceeds pinned source size')
    while received < size:
        check_stop()
        if shutil.disk_usage(destination.parent).free < reserve+size-received:
            raise OSError('Reserve download would use space reserved for training')
        end = min(received+4*1024**2,size)-1
        expected = end-received+1
        for attempt in range(3):
            check_stop()
            try:
                began = time.monotonic()
                request = urllib.request.Request(row['url'],headers={
                    'User-Agent':'llm-lab-sft-reserve/1','Range':f'bytes={received}-{end}'})
                with opener.open(request,timeout=20) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {received}-{end}/{size}':
                        raise ValueError('GitHub did not honor the exact bounded Range')
                    body = response.read(expected+1)
                    if len(body) != expected:
                        raise ValueError('GitHub Range body length differs')
                with partial.open('ab' if received else 'wb') as stream:
                    stream.write(body);stream.flush();os.fsync(stream.fileno())
                received += expected
                progress(row,received)
                delay = expected/rate-(time.monotonic()-began)
                if delay > 0:
                    time.sleep(delay)
                break
            except InterruptedError:
                raise
            except OSError as error:
                if attempt == 2:
                    raise RuntimeError('GitHub bounded Range request failed') from None
                time.sleep(1+attempt)
    if partial.stat().st_size != size or digest(partial) != row['sha256']:
        raise ValueError('Downloaded size or SHA256 verification failed')
    partial.replace(destination)
    return destination


def inventory(config, directory):
    """仅做实际行数和来源/评分字段统计，不把结构检查当作语义验收。"""
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    arrow.set_cpu_count(1);arrow.set_io_thread_count(1)
    output = []
    for row in config['files']:
        path = directory/row['destination']
        info = {key:row[key] for key in ('source','file','sha256','size_bytes','language','format')}
        if path.suffix == '.parquet':
            file = parquet.ParquetFile(path)
            info.update(rows=file.metadata.num_rows,columns=file.schema_arrow.names)
            wanted = {'smoltalk_parquet':['source'], 'opencsg_parquet':['score','classify','difficulty'],
                      'oasst2_parquet':['lang','role','review_result','deleted']}[row['format']]
            columns = [key for key in wanted if key in file.schema_arrow.names]
            counts = {key:Counter() for key in columns}
            reviewed_pairs = Counter()
            score_four = 0
            for batch in file.iter_batches(batch_size=4096,columns=columns,use_threads=False):
                for record in batch.to_pylist():
                    for key in columns:
                        counts[key][str(record.get(key))] += 1
                    if row['format'] == 'opencsg_parquet' and isinstance(record.get('score'),(int,float)) and record['score'] >= 4:
                        score_four += 1
                    if row['format'] == 'oasst2_parquet' and record.get('lang') in {'en','zh'} and record.get('role') == 'assistant' and record.get('review_result') is True and not record.get('deleted'):
                        reviewed_pairs[record['lang']] += 1
            info['metadata_distributions'] = {key:dict(value) for key,value in counts.items()}
            if row['format'] == 'smoltalk_parquet':
                info['allowed_family_rows_before_content_checks'] = sum(counts['source'][name] for name in row['selection_policy']['source_allowlist'])
            if row['format'] == 'opencsg_parquet':
                info['instruction_score_ge4_rows_before_answer_checks'] = score_four
            if row['format'] == 'oasst2_parquet':
                info['reviewed_assistant_messages_before_parent_linking'] = dict(reviewed_pairs)
        elif row['format'] == 'alpaca_json':
            with path.open(encoding='utf-8') as stream:
                info['rows'] = len(json.load(stream))
        elif row['format'] == 'dolly_jsonl':
            counts = Counter();total = 0
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line);total += 1;counts[record['category']] += 1
            info.update(rows=total,metadata_distributions={'category':dict(counts)})
        else:
            raise ValueError('Unsupported candidate source format')
        output.append(info)
    return {'files':output,'total_raw_rows_or_messages':sum(row['rows'] for row in output),
            'note':'Mixed units: OASST rows are messages. Raw rows are not accepted unique training examples.',
            'training_ready':False,'semantic_gold':False,'current_sft_dataset_changed':False}


def run(config_path, work_dir):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    root = (config_path.parent/config['project_root']).resolve()
    directory = validate(config,root)
    work_dir = private_path(work_dir,root/'runs/assistant/data_prepare')
    directory.mkdir(parents=True,exist_ok=True);work_dir.mkdir(parents=True,exist_ok=True)
    # 数据下载和统计仅占两个CPU核；不实例化模型，不获取项目GPU训练锁。
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if hasattr(os,'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[-2:]))
    try:
        os.nice(10)
    except (AttributeError,OSError):
        pass
    state = {'status':'downloading','pid':os.getpid(),'cpu_only':True,'files_completed':[],
             'files_planned':len(config['files']),'bytes_planned':sum(row['size_bytes'] for row in config['files']),
             'training_ready':False,'current_sft_dataset_changed':False}
    started = time.monotonic();stopped = False
    def stop(signum,frame):
        nonlocal stopped
        stopped = True
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,stop)
    def check_stop():
        if stopped or time.monotonic()-started > config['max_runtime_seconds']:
            raise InterruptedError('Download stopped at its bounded runtime or by signal')
    def progress(row,received):
        state.update(current_source=row['source'],current_file=row['file'],current_received_bytes=received,
                     elapsed_seconds=time.monotonic()-started,free_gib=shutil.disk_usage(root).free/GIB)
        save(work_dir/'state.json',state)
    try:
        for identifier,spec in config['source_documents'].items():
            target = private_path('source_cards/'+identifier+'/README.md',directory)
            target.parent.mkdir(parents=True,exist_ok=True)
            data = spec['readme'].encode('utf-8')
            if hashlib.sha256(data).hexdigest() != spec['readme_sha256']:
                raise ValueError('Pinned source card SHA mismatch')
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Refusing to replace a different source card')
            target.write_bytes(data)
            if 'data_license' in spec:
                license_data = spec['data_license'].encode('utf-8')
                if hashlib.sha256(license_data).hexdigest() != spec['data_license_sha256']:
                    raise ValueError('Pinned data license SHA mismatch')
                target.with_name('DATA_LICENSE.txt').write_bytes(license_data)
        save(directory/'reserve_config.json',config)
        save(work_dir/'state.json',state)
        for row in config['files']:
            progress(row,0)
            path = fetch(row,directory,config['min_free_bytes'],config['bytes_per_second'],progress,check_stop)
            state['files_completed'].append({'source':row['source'],'file':row['file'],'path':str(path),
                                            'size_bytes':row['size_bytes'],'sha256':row['sha256']})
            save(work_dir/'state.json',state)
            print(json.dumps({'event':'source_verified','source':row['source'],'file':row['file'],
                              'files_completed':len(state['files_completed'])}),flush=True)
        state['status'] = 'counting_source_metadata';save(work_dir/'state.json',state)
        report = inventory(config,directory)
        save(directory/'inventory.json',report)
        state.update(status='completed',elapsed_seconds=time.monotonic()-started,
                     free_gib=shutil.disk_usage(root).free/GIB,inventory=str(directory/'inventory.json'))
        save(directory/'manifest.json',{'kind':'sft_candidate_raw_reserve','training_ready':False,
             'current_sft_dataset_changed':False,'files':state['files_completed'],
             'future_release_requirements':config['future_release_requirements'],
             'inventory_sha256':digest(directory/'inventory.json'),'config_sha256':digest(config_path)})
    except BaseException as error:
        state.update(status='stopped' if isinstance(error,InterruptedError) else 'failed',error_type=type(error).__name__)
        raise
    finally:
        save(work_dir/'state.json',state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--work-dir',required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config,arguments.work_dir),ensure_ascii=False,indent=2),flush=True)
