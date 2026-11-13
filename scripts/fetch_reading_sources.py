"""下载官方固定版本阅读理解数据；原始标注文件只读，不运行上游代码。"""
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from urllib.parse import quote


def digests(path):
    sha,git=hashlib.sha256(),hashlib.sha1()
    git.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):sha.update(block);git.update(block)
    return sha.hexdigest(),git.hexdigest()


def fetch(spec,cache,proxy):
    allowed={'ymcui/cmrc2018','DRCKnowledgeTeam/DRCD','rajpurkar/squad'}
    if spec['source'] not in allowed:raise ValueError('Unexpected source')
    rel=Path(spec['path']);target=cache/spec['source'].replace('/','__')/rel
    if rel.is_absolute() or '..' in rel.parts or not target.resolve().is_relative_to(cache) or target.is_symlink():raise ValueError('Unsafe source path')
    host='https://raw.githubusercontent.com' if spec['provider']=='github' else 'https://huggingface.co/datasets'
    separator='/' if spec['provider']=='github' else '/resolve/'
    url=host+'/'+spec['source']+separator+spec['revision']+'/'+quote(spec['path'])
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.part')
    if not target.exists():
        command=['curl.exe' if os.name=='nt' else 'curl','--fail','--location','--silent','--show-error',
                 '--retry','2','--retry-all-errors','--connect-timeout','15','--max-time','180','--output',str(temporary),url]
        if proxy:command+=['--proxy',proxy]
        subprocess.run(command,check=True,timeout=600)
        path=temporary
    else:path=target
    if path.stat().st_size!=spec['bytes']:raise ValueError('Source size changed')
    sha,git=digests(path)
    if spec.get('sha256',sha)!=sha or spec.get('git_blob',git)!=git:raise ValueError('Pinned source digest mismatch')
    if path!=target:path.replace(target)
    result={**spec,'sha256':sha,'url':url,'local_path':target.relative_to(cache).as_posix()}
    print(json.dumps({'downloaded':result['local_path'],'bytes':spec['bytes']}),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--catalog',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--proxy');args=ap.parse_args()
    cache=args.cache.resolve()
    if os.name=='nt' and cache.drive.upper()!='E:':raise ValueError('Local data must stay on E:')
    cache.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(lambda spec:fetch(spec,cache,args.proxy),json.loads(args.catalog.read_text(encoding='utf-8'))))
    payload={'schema_version':1,'kind':'pinned_official_extractive_qa_sources','sources':results,'training_started':False}
    text=json.dumps(payload,ensure_ascii=False,indent=2)+'\n';target=cache/'sources.lock.json'
    if target.exists() and target.read_text(encoding='utf-8')!=text:raise ValueError('Never replace a source lock')
    if not target.exists():target.write_text(text,encoding='utf-8')
    print(json.dumps({'status':'verified','files':len(results),'bytes':sum(r['bytes'] for r in results),'cache':str(cache)}),flush=True)


if __name__=='__main__':main()
