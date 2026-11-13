"""验证下载中断、损坏与空间不足不会发布坏文件或覆盖现有训练数据。"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/download_sft_reserve.py'
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).with_name('download_sft_reserve.py')
SPEC = importlib.util.spec_from_file_location('sft_reserve_download',SCRIPT)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


@pytest.fixture
def source(tmp_path,monkeypatch):
    body = b'complete immutable training source'
    row = {'source':'example','repository':'author/data','revision':'a'*40,'file':'data/train-00.parquet',
           'url':'https://huggingface.co/datasets/author/data/resolve/'+'a'*40+'/data/train-00.parquet',
           'size_bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),
           'destination':'raw/example/train-00.parquet','upstream_split':'train_or_unsplit_source'}
    monkeypatch.setattr(download.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*download.GIB))
    return row,body


def open_source(monkeypatch,body,*,status=200,headers=None):
    import io
    requests = []
    class Response(io.BytesIO):
        def __init__(self):
            super().__init__(body);self.status=status;self.headers=headers or {}
    class Opener:
        def open(self,request,**kwargs):
            requests.append(request);return Response()
    monkeypatch.setattr(download,'_network_openers',lambda:(None,Opener()))
    monkeypatch.setattr(download,'_resolve_target',lambda *args:'https://trusted-cas.example/file')
    return requests


def config(row):
    return {'kind':'sft_candidate_raw_reserve','training_ready':False,
            'reserve_dir':'artifacts/posttrain/reserve','min_free_bytes':24*download.GIB,
            'max_total_source_bytes':download.GIB,'bytes_per_second':8*1024**2,
            'max_runtime_seconds':1800,'files':[row]}


def test_complete_source_is_committed_only_after_sha_verification(tmp_path,monkeypatch,source):
    row,body=source;requests=open_source(monkeypatch,body)
    path=download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert path.read_bytes()==body and len(requests)==1
    assert not path.with_name(path.name+'.part').exists()


def test_exact_range_resumes_previous_partial(tmp_path,monkeypatch,source):
    row,body=source;path=tmp_path/row['destination'];path.parent.mkdir(parents=True)
    path.with_name(path.name+'.part').write_bytes(body[:7])
    requests=open_source(monkeypatch,body[7:],status=206,headers={'Content-Range':f'bytes 7-{len(body)-1}/{len(body)}'})
    result=download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert result.read_bytes()==body and requests[0].get_header('Range')=='bytes=7-'


def test_server_cannot_silently_replace_a_resumed_download(tmp_path,monkeypatch,source):
    row,body=source;path=tmp_path/row['destination'];path.parent.mkdir(parents=True)
    part=path.with_name(path.name+'.part');part.write_bytes(body[:7]);open_source(monkeypatch,body)
    with pytest.raises(ValueError,match='Range'):
        download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert not path.exists() and part.read_bytes()==body[:7]


def test_corrupt_body_is_not_published(tmp_path,monkeypatch,source):
    row,body=source;open_source(monkeypatch,b'x'*len(body))
    with pytest.raises(ValueError,match='SHA256'):
        download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert not (tmp_path/row['destination']).exists()


def test_oversized_response_is_not_committed(tmp_path,monkeypatch,source):
    row,body=source;open_source(monkeypatch,body+b'excess')
    with pytest.raises(ValueError,match='exceeds'):
        download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert not (tmp_path/row['destination']).exists()


def test_insufficient_disk_never_starts_a_transfer(tmp_path,monkeypatch,source):
    row,body=source;requests=open_source(monkeypatch,body)
    monkeypatch.setattr(download.shutil,'disk_usage',lambda path:SimpleNamespace(free=24*download.GIB))
    with pytest.raises(OSError,match='space reserved'):
        download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert not requests and not (tmp_path/row['destination']).exists()


def test_mismatched_existing_source_and_unrelated_checkpoint_are_preserved(tmp_path,monkeypatch,source):
    row,body=source;open_source(monkeypatch,body)
    existing=tmp_path/row['destination'];existing.parent.mkdir(parents=True);existing.write_bytes(b'old-content')
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'user model')
    with pytest.raises(ValueError,match='Existing source'):
        download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert existing.read_bytes()==b'old-content' and checkpoint.read_bytes()==b'user model'


def test_public_test_and_validation_cannot_enter_reserve(tmp_path,source):
    row,_=source
    for filename in ('data/test-00.parquet','data/validation-00.parquet'):
        other=deepcopy(row);other['file']=filename
        with pytest.raises(ValueError,match='evaluation'):
            download.validate(config(other),tmp_path)


def test_path_traversal_is_rejected_before_any_write(tmp_path,source):
    row,_=source;row['destination']='../../best.pt'
    with pytest.raises(ValueError,match='escapes'):
        download.validate(config(row),tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_reserve_directory_resolves_once_relative_to_project(tmp_path,source):
    row,_=source
    assert download.validate(config(row),tmp_path)==tmp_path/'artifacts/posttrain/reserve'


def test_signal_stop_keeps_partial_and_never_commits(tmp_path,monkeypatch,source):
    row,body=source;open_source(monkeypatch,body)
    checks=0
    def stop():
        nonlocal checks
        checks+=1
        if checks==2:raise InterruptedError('requested')
    with pytest.raises(InterruptedError):
        download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,stop)
    assert not (tmp_path/row['destination']).exists()


def test_github_uses_bounded_get_without_head_and_verifies_sha(tmp_path,monkeypatch,source):
    row,body=source
    row['url']='https://raw.githubusercontent.com/author/data/'+'a'*40+'/data/train-00.parquet'
    requests=open_source(monkeypatch,body,status=206,headers={'Content-Range':f'bytes 0-{len(body)-1}/{len(body)}'})
    direct=download._network_openers()[1]
    monkeypatch.setattr(download,'_network_openers',lambda:(direct,None))
    monkeypatch.setattr(download,'_resolve_target',lambda *args:pytest.fail('GitHub HEAD is not needed'))
    path=download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert path.read_bytes()==body and requests[0].get_header('Range')==f'bytes=0-{len(body)-1}'


def test_github_rejects_wrong_content_range_before_committing(tmp_path,monkeypatch,source):
    row,body=source
    row['url']='https://raw.githubusercontent.com/author/data/'+'a'*40+'/data/train-00.parquet'
    open_source(monkeypatch,body,status=206,headers={'Content-Range':f'bytes 1-{len(body)}/{len(body)}'})
    direct=download._network_openers()[1];monkeypatch.setattr(download,'_network_openers',lambda:(direct,None))
    with pytest.raises(ValueError,match='bounded Range'):
        download.fetch(row,tmp_path,24*download.GIB,10**12,lambda *args:None,lambda:None)
    assert not (tmp_path/row['destination']).exists()
