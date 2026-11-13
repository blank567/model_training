"""单个扩充实验的后台监督：使用原训练入口，成功后才做冻结最终测试。"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from llm_lab.common.io import read_json, sha256_file, write_json
from llm_sft.config import runtime_fingerprint
import train_sft as launcher


def scaled_template(source, manifest, selected, comparison_file):
    config=copy.deepcopy(source)
    for key in ('data_manifest_sha256','data_identity'):config.pop(key,None)
    config['data_manifest']=str(manifest);config['output_dir']=str(Path(config['project_root'])/'runs/sft_100k_unprepared')
    config['train']['wandb_run_name']=None
    config['experiment']={'phase':'test_selected_configuration_scaled_to_100k','train_examples':100000,
        'source_config':selected['source_config'],'source_checkpoint':selected['checkpoint'],
        'source_checkpoint_sha256':selected['checkpoint_sha256'],'selection_report':str(comparison_file),
        'initialization':'same pretraining Base; fresh AdamW/scheduler/RNG',
        'primary_test_used_for_configuration_selection':True,'auxiliary_test_reserved':True,
        'note':'Only published training data version changes; original training hyperparameters and Base are retained.'}
    return config


def training_complete(code, summary, expected_steps):
    return code==0 and summary.get('status')=='completed' and summary.get('step')==expected_steps


def run(handoff_path):
    handoff=read_json(handoff_path);root=Path(handoff['project_root']);directory=Path(handoff['controller_dir'])
    if not directory.resolve().is_relative_to(root/'runs/assistant/system'):raise ValueError('Wrong controller directory')
    config_path=Path(handoff['RUN_CONFIG']);config=launcher.frozen_config(config_path)
    if runtime_fingerprint(root)!=handoff['runtime_sha256']:raise ValueError('Frozen source changed')
    state={'status':'starting','controller_pid':os.getpid(),'RUN_CONFIG':str(config_path),
        'RUN_DIR':handoff['RUN_DIR'],'RUN_LOG':handoff['RUN_LOG'],'current_pid':None,'step':0}
    active=None;stopped=False;old_handlers={}
    def save():state['updated_unix_seconds']=time.time();write_json(directory/'state.json',state)
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        if active is not None:
            try:active.send_signal(signal.SIGTERM)
            except ProcessLookupError:pass
    def child(command,log,timeout,phase):
        nonlocal active
        if stopped:return -signal.SIGTERM
        path=Path(log);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('ab') as stream:
            active=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status=phase,current_pid=active.pid);save()
            if stopped:
                try:active.send_signal(signal.SIGTERM)
                except ProcessLookupError:pass
            deadline=time.monotonic()+timeout
            while True:
                try:code=active.wait(timeout=min(15,max(.01,deadline-time.monotonic())));break
                except subprocess.TimeoutExpired:
                    if time.monotonic()<deadline:save();continue
                    active.send_signal(signal.SIGTERM)
                    try:active.wait(timeout=120)
                    except subprocess.TimeoutExpired:active.kill();active.wait(timeout=30)
                    state['timeout_phase']=phase;code=124;break
        active=None;state['current_pid']=None;save();return code
    python=str(root/'.venv/bin/python');launcher_file=str(root/'scripts/train_sft.py')
    for signum in (signal.SIGTERM,signal.SIGINT):old_handlers[signum]=signal.signal(signum,stop)
    try:
        save()
        checked=child([python,'-u',launcher_file,'check','--config',str(config_path)],
            directory/'preflight.log',180,'checking')
        if checked!=0 or stopped:state.update(status='stopped' if stopped else 'failed_preflight');return state
        code=child([python,'-u',launcher_file,'start','--config',str(config_path)],
            handoff['RUN_LOG'],config['budget']['max_runtime_seconds']+600,'training')
        summary_path=Path(handoff['RUN_DIR'])/'summary.json';summary=read_json(summary_path) if summary_path.is_file() else {}
        state.update(step=summary.get('step',0),training_summary=summary)
        if stopped:state['status']='stopped';return state
        if not training_complete(code,summary,handoff['plan']['stop_step']):
            state.update(status='training_incomplete',returncode=code);return state
        final_code=child([python,'-u',str(Path(handoff['runtime_snapshot'])/'evaluate_scaled_sft.py'),
            '--handoff',str(Path(handoff_path).resolve())],directory/'final_evaluation.log',2100,'final_evaluation')
        if stopped:state['status']='stopped'
        elif final_code:state.update(status='final_evaluation_failed',returncode=final_code)
        else:
            report=read_json(Path(handoff['final_evaluation_dir'])/'comparison.json')
            assert report['status']=='completed'
            state.update(status='completed',final_comparison=str(Path(handoff['final_evaluation_dir'])/'comparison.json'))
        return state
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__);raise
    finally:
        save()
        for signum,handler in old_handlers.items():signal.signal(signum,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--handoff',type=Path,required=True)
    result=run(parser.parse_args().handoff);print(json.dumps(result,ensure_ascii=False),flush=True)
    raise SystemExit(0 if result['status']=='completed' else 2)
