"""交付只包含评测数据、报告及工具快照，明确排除模型和W&B备份。"""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main(work):
    work=Path(work).resolve();root=Path(__file__).resolve().parents[1]
    assert read(work/'delivery_verification.json')['status']=='verified'
    data=work/'results_local';sources=work/'sources_local';snapshots=data/'source_snapshots';snapshots.mkdir(exist_ok=True)
    plan=read(sources/'plan.json');cross_plan=read(data/'cross_domain_loss/plan.json')
    assert sha(snapshots/'reading_evaluator.py')==plan['reused_suites']['reading_test_2000']['source_evaluator_sha256']
    for filename,target,expected in [('evaluate_posttrain_comprehensive.py','instruction_evaluator.py',read(data/'comparison.json')['source_sha256']),
                                     ('evaluate_posttrain_retention.py','cross_domain_evaluator.py',cross_plan['evaluator_sha256'])]:
        source=root/'scripts'/filename;assert sha(source)==expected;shutil.copyfile(source,snapshots/target)
    (snapshots/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in snapshots.glob('*.py')},ensure_ascii=False,indent=2),encoding='utf-8')
    tools=work/'reproduce_tools';tools.mkdir(exist_ok=True)
    for filename in ['build_posttrain_test_report.py','verify_posttrain_test_report.py','extract_test_report_bundle.py','package_posttrain_test_delivery.py']:
        shutil.copyfile(root/'scripts'/filename,tools/filename)
    archive=work/'后训练测试报告与数据_20261104.zip'
    if archive.exists():raise FileExistsError('Do not overwrite an existing delivery archive')
    files=[work/'README.md',work/'delivery_verification.json']
    for folder in ['report','results_local','sources_local','reproduce_tools']:
        files.extend(p for p in (work/folder).rglob('*') if p.is_file())
    assert not any(p.suffix.lower() in {'.pt','.pth','.bin','.wandb'} or p.name in {'.netrc','authorized_keys','network.env'} for p in files)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for p in sorted(files):stream.write(p,p.relative_to(work).as_posix())
    with zipfile.ZipFile(archive) as stream:assert stream.testzip() is None
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files),
             'no_model_weights_or_wandb_exports':True,'zip_crc_checked':True}
    (work/'delivery_archive.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',required=True,type=Path);main(p.parse_args().work)
