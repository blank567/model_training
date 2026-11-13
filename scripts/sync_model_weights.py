"""经SSH将正式实验权重备份到原E盘目录，支持断点续传、双端SHA256和冲突保留。"""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
import argparse,hashlib,json,os,shlex,shutil,subprocess,threading,time

ROOT=Path(__file__).resolve().parents[1]
SSH=r'C:\Program Files\OpenSSH\ssh.exe'
REMOTE_ROOT='/root/autodl-tmp/model_training'
REMOTE=r'''
from pathlib import Path
import hashlib,json,os,sys
os.nice(10)
r=json.load(sys.stdin);root=Path('/root/autodl-tmp/model_training');relative=Path(r['path']);p=root/relative
assert not relative.is_absolute() and '..' not in relative.parts and p.resolve().is_relative_to(root/'runs')
assert p.suffix in {'.pt','.safetensors','.bin'} and not p.is_symlink()
with p.open('rb') as f:
    s=os.fstat(f.fileno());meta={'path':r['path'],'size':s.st_size,'mtime_ns':s.st_mtime_ns,'dev':s.st_dev,'ino':s.st_ino}
    if r.get('source') is not None and r['source']!=meta:
        print(json.dumps({'error':'source_changed','source':meta}),file=sys.stderr,flush=True);raise SystemExit(43)
    offset=r['offset'];end=min(s.st_size,r.get('limit') or s.st_size)
    assert 0<=offset<=end
    print(json.dumps({'source':meta,'offset':offset,'end':end}),file=sys.stderr,flush=True)
    digest=hashlib.sha256();left=offset
    while left:
        block=f.read(min(left,4*2**20));assert block;digest.update(block);left-=len(block)
    left=end-offset
    while left:
        block=f.read(min(left,4*2**20));assert block;digest.update(block);sys.stdout.buffer.write(block);left-=len(block)
    sys.stdout.buffer.flush();after=os.fstat(f.fileno())
    assert (s.st_size,s.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source modified while reading'
print(json.dumps({'sha256':digest.hexdigest(),'bytes':end,'complete':end==s.st_size}),file=sys.stderr,flush=True)
'''


def remote(program):
    script='cd '+shlex.quote(REMOTE_ROOT)+' && source scripts/env_remote.sh && '+shlex.join(['.venv/bin/python','-c',program])
    return [SSH,'-F',str(ROOT/'tmp/ssh/config'),'-o','ConnectTimeout=15','-T','autodl-control',shlex.join(['bash','-c',script])]


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def safe_target(relative):
    p=PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative or p.parts[0]!='runs':raise ValueError('Unsafe artifact path')
    target=ROOT.joinpath(*p.parts)
    if not target.resolve().is_relative_to(ROOT/'runs') or target.is_symlink():raise ValueError('Local destination escapes runs')
    return target


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
    return h.hexdigest()


def promote(part,target,work):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        backup=work/'previous_local'/target.relative_to(ROOT)
        if backup.exists():backup=backup.with_name(backup.name+'.'+str(time.time_ns()))
        backup.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(target,backup)
        except OSError:shutil.copy2(target,backup)
    os.replace(part,target)


class Transfer:
    def __init__(self,work,files):
        self.work=work;self.files=files;self.lock=threading.Lock();self.start=time.time();self.stop=threading.Event()
        self.records={};self.active={};self.downloaded=0;self.errors=[]
        receipt=work/'weights_verified.jsonl'
        if receipt.exists():
            for line in receipt.read_text(encoding='utf-8').splitlines():
                r=json.loads(line);self.records[r['path']]=r
        self.total=sum(r['bytes'] for r in files)

    def status(self,phase='downloading'):
        with self.lock:
            done=[r for r in self.files if r['path'] in self.records]
            verified=sum(r['bytes'] for r in done);elapsed=max(.001,time.time()-self.start);rate=self.downloaded/elapsed
            return {'phase':phase,'total_files':len(self.files),'verified_files':len(done),'total_bytes':self.total,
                'verified_bytes':verified,'received_this_session_bytes':self.downloaded,'bytes_per_second':rate,
                'active':dict(self.active),'errors':list(self.errors),'updated_unix':time.time(),
                'elapsed_seconds':elapsed}

    def monitor(self):
        while not self.stop.wait(2):save(self.work/'weights_status.json',self.status())

    def pull(self,spec,limit=None):
        name=spec['path'];target=safe_target(name);key=hashlib.sha256(name.encode()).hexdigest()
        partial=self.work/'partial'/(key+'.part');metadata=partial.with_suffix('.json');partial.parent.mkdir(parents=True,exist_ok=True)
        previous=self.records.get(name)
        if limit is None and previous and target.exists() and target.stat().st_size==previous['bytes']:
            if digest_file(target)==previous['sha256']:return previous
        for attempt in range(4):
            offset=partial.stat().st_size if partial.exists() and metadata.exists() else 0
            source=json.loads(metadata.read_text()) if offset else None
            process=subprocess.Popen(remote(REMOTE),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            request={'path':name,'offset':offset,'source':source,'limit':limit}
            process.stdin.write(json.dumps(request).encode());process.stdin.close()
            line=process.stderr.readline();header=json.loads(line) if line else {}
            if header.get('error')=='source_changed':
                process.wait()
                if partial.exists():partial.rename(partial.with_name(partial.name+'.stale.'+str(time.time_ns())))
                metadata.unlink(missing_ok=True);continue
            if 'source' not in header:
                error=process.stderr.read().decode(errors='replace');process.wait();raise RuntimeError('SSH source header unavailable: '+error[-300:])
            save(metadata,header['source']);end=header['end'];h=hashlib.sha256()
            if offset:
                with partial.open('rb') as f:
                    for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
            with self.lock:self.active[name]={'received_bytes':offset,'total_bytes':header['source']['size']}
            with partial.open('ab' if offset else 'wb') as f:
                while True:
                    block=process.stdout.read(2**20)
                    if not block:break
                    f.write(block);h.update(block)
                    with self.lock:self.downloaded+=len(block);self.active[name]['received_bytes']+=len(block)
                f.flush();os.fsync(f.fileno())
            trailer=process.stderr.read().decode(errors='replace');code=process.wait()
            if code:
                if attempt==3:raise RuntimeError('SSH transfer interrupted: '+trailer[-300:])
                time.sleep(2);continue
            result=json.loads(trailer.strip().splitlines()[-1])
            if partial.stat().st_size!=end or h.hexdigest()!=result['sha256']:raise RuntimeError('SHA256 or size mismatch: '+name)
            record={'path':name,'bytes':end,'sha256':result['sha256'],'source':header['source'],'verified_at_unix':time.time(),
                'complete':result['complete']}
            if not result['complete']:return record
            if target.exists() and digest_file(target)==record['sha256']:partial.unlink()
            else:promote(partial,target,self.work)
            os.utime(target,ns=(header['source']['mtime_ns'],header['source']['mtime_ns']))
            with self.lock:
                self.records[name]=record;self.active.pop(name,None)
                with (self.work/'weights_verified.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps({'verified':name,'bytes':end}),flush=True);return record
        raise RuntimeError('Repeated source changes: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['probe','sync']);p.add_argument('--inventory',type=Path,default=ROOT/'tmp/model_sync_inventory_20261103.json')
    p.add_argument('--work',type=Path);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    latest=ROOT/'runs/assistant/system/latest_model_sync.json'
    if a.work:work=a.work.resolve()
    elif a.action=='sync' and latest.exists():work=Path(json.loads(latest.read_text())['work'])
    else:work=ROOT/'runs/assistant/system'/('model_sync_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    if not work.is_relative_to(ROOT/'runs/assistant/system'):raise ValueError('Work must stay in assistant system records')
    work.mkdir(parents=True,exist_ok=True);save(latest,{'work':str(work)})
    inventory=json.loads(a.inventory.read_text(encoding='utf-8'));save(work/'inventory.json',inventory)
    files=[f for r in inventory['formal_runs'] for f in r['weights'] if not r.get('fixture')]
    files.sort(key=lambda f:(0 if 'pretrain_300m_3b_' in f['path'] and f['path'].endswith('/best.pt') else 1 if not f['path'].endswith('/last.pt') else 2,f['bytes'],f['path']))
    if shutil.disk_usage(ROOT).free<sum(r['bytes'] for r in files)+8*2**30:raise OSError('Local space insufficient for full backup')
    transfer=Transfer(work,files);thread=threading.Thread(target=transfer.monitor,daemon=True);thread.start()
    try:
        if a.action=='probe':
            before=time.monotonic();r=transfer.pull(files[0],limit=128*2**20);seconds=time.monotonic()-before
            speed=r['bytes']/seconds;result={'work':str(work),'probe_bytes':r['bytes'],'seconds':seconds,'bytes_per_second':speed,
                'estimated_weight_download_hours':transfer.total/speed/3600,'partial_reused_by_full_sync':True}
            save(work/'probe.json',result);print(json.dumps(result),flush=True)
        else:
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                tasks={pool.submit(transfer.pull,f):f for f in files}
                for future in as_completed(tasks):
                    try:future.result()
                    except Exception as error:
                        with transfer.lock:transfer.errors.append({'path':tasks[future]['path'],'error':str(error)})
            state=transfer.status('completed' if not transfer.errors else 'completed_with_errors');save(work/'weights_status.json',state)
            save(work/'weights_result.json',state);print(json.dumps(state),flush=True)
    finally:transfer.stop.set();thread.join(timeout=3)
if __name__=='__main__':main()
