"""在E盘校验并读取本项目测试包，避免套用Linux文件权限或解包到目录外。"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tarfile


def extract(archive,receipt,output,kind='results'):
    archive=Path(archive);receipt=json.loads(Path(receipt).read_text(encoding='utf-8-sig'));output=Path(output).resolve()
    if output.drive.upper()!='E:':raise ValueError('Test artifacts must remain on E:')
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=receipt['sha256']:raise ValueError('Archive SHA256 mismatch')
    output.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as stream:
        index='report_sources/file_manifest.json' if kind=='sources' else 'transfer_manifest.json'
        manifest=json.load(stream.extractfile(index))
        if kind=='sources':expected={'report_sources/'+r['relative']:r for r in manifest}
        else:expected={r['path']:r for r in manifest['files']}
        seen=set()
        for member in stream.getmembers():
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or any(':' in p or '\\' in p for p in path.parts):raise ValueError('Unsafe archive member')
            if member.isdir():continue
            if not member.isfile():raise ValueError('Unsafe archive link')
            relative=path.relative_to('report_sources') if kind=='sources' else path
            target=(output/Path(*relative.parts)).resolve()
            if not target.is_relative_to(output):raise ValueError('Archive path escaped')
            blob=stream.extractfile(member).read()
            if member.name!=index:
                check=expected[member.name]
                if hashlib.sha256(blob).hexdigest()!=check['sha256'] or len(blob)!=check['bytes']:raise ValueError('Member changed')
                seen.add(member.name)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
        if seen!=set(expected):raise ValueError('Missing members')
    print(json.dumps({'output':str(output),'verified_files':len(seen),'mode':'sources' if kind=='sources' else manifest['mode']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--receipt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['results','sources'],default='results')
    a=p.parse_args();extract(a.archive,a.receipt,a.output,a.kind)
