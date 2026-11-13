"""CPU 数据构建监督器：真实依赖测试、构建、独立验收、发布记录。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root):
    return hashlib.sha256('\n'.join(path.relative_to(root).as_posix()+':'+digest(path)
        for path in sorted((root/'src/llm_lab').rglob('*.py'))).encode()).hexdigest()


def save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def main(config_path):
    settings=json.loads(config_path.read_text(encoding='utf-8'))
    root=(config_path.resolve().parent/settings['project_root']).resolve()
    work=root/settings['work_dir'];work.mkdir(parents=True,exist_ok=True)
    if hasattr(os,'sched_getaffinity'):
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[-2:]))
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                                    'TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1',
                                    'PYTHONPATH':str(root/'src'),'TMPDIR':str(root/'tmp')})
    started=time.monotonic();before=source_identity(root)
    state={'status':'starting','cpu_only':True,'supervisor_pid':os.getpid(),
           'source_tree_sha256_before':before,'config':str(config_path),'new_model_training_started':False}
    snapshot=work/'runtime_snapshot';snapshot.mkdir(exist_ok=True)
    files=['prepare_sft_data.py','fetch_sft_sources.py','sft_data_quality.py','audit_sft_data.py','run_sft_data_build.py']
    expected={name:digest(root/'scripts'/name) for name in files}
    for name in files:
        target=snapshot/name
        if target.exists() and digest(target)!=expected[name]:
            raise ValueError('Refusing to replace an immutable SFT runtime snapshot')
        shutil.copy2(root/'scripts'/name,target)
    shutil.copy2(config_path,snapshot/'config.json')
    save(work/'job_state.json',state)
    def run(arguments,timeout):
        subprocess.run([sys.executable,*arguments],cwd=root,env=env,check=True,timeout=timeout)
    try:
        state['status']='testing';save(work/'job_state.json',state)
        run(['-m','pytest','tests/test_sft_data_quality.py','tests/test_sft_data_build.py','tests/test_sft_data_audit.py',
             '-q','--basetemp',str(work/'fixture-tests')],120)
        state['status']='building';save(work/'job_state.json',state)
        run(['-u','scripts/prepare_sft_data.py','--config',str(config_path)],4800)
        output=root/settings['output_dir']
        state['status']='auditing';save(work/'job_state.json',state)
        run(['-u','scripts/audit_sft_data.py','--data-dir',str(output),'--output',str(output/'acceptance.json')],600)
        acceptance=json.loads((output/'acceptance.json').read_text(encoding='utf-8'))
        if acceptance['status']!='passed' or not acceptance['structurally_ready']:
            raise ValueError('SFT independent acceptance did not pass')
        if source_identity(root)!=before or any(digest(root/'scripts'/name)!=expected[name] for name in files):
            raise ValueError('Source or SFT runtime changed during data preparation')
        manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if manifest['data_identity']!=acceptance['data_identity']:
            raise ValueError('Audit report identity mismatch')
        manifest['acceptance']={'status':'passed','report':'acceptance.json','sha256':digest(output/'acceptance.json'),
                                'scope':'structural, masking, exact grouping, fixed-point normalization and source hashes',
                                'semantic_gold':False}
        save(output/'manifest.json',manifest)
        state.update(status='completed',output=str(output),data_identity=manifest['data_identity'],
                     elapsed_seconds=time.monotonic()-started,source_tree_sha256_after=source_identity(root),
                     data_manifest_sha256=digest(output/'manifest.json'),acceptance=acceptance,
                     disk_free_gib=shutil.disk_usage(root).free/1024**3)
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-started)
        save(work/'job_state.json',state)
        raise
    save(work/'job_state.json',state)
    print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    main(parser.parse_args().config)
