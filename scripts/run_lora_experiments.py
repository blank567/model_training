"""三组学习率等预算筛选、胜出配置继续及全量最终测试的后台监督。"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import signal,subprocess,time

from llm_lab.common.io import read_json,sha256_file,write_json
from llm_lora.config import read_config,runtime_fingerprint
from llm_sft.config import plan_details
import train_lora


def screen_result(summary,expected):
    if summary.get('status')!='paused' or summary.get('step')!=expected or not summary.get('latest_development'):
        raise ValueError('Candidate did not finish equal development screen budget')
    value=summary['latest_development']['macro_loss']
    import math
    if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Invalid development loss')
    return value


def select_candidate(results):
    if len(results)!=3 or len({row['key'] for row in results})!=3:raise ValueError('All three candidates must complete before selection')
    return min(results,key=lambda row:(row['development_loss'],row['key']))


def run(plan_path):
    plan_path=Path(plan_path);plan=read_json(plan_path);root=Path(plan['project_root']);directory=Path(plan['controller_dir'])
    if not directory.resolve().is_relative_to(root/'runs/assistant/system'):raise ValueError('Invalid controller directory')
    if runtime_fingerprint()!=plan['runtime_sha256']:raise ValueError('LoRA source identity changed')
    configurations=[train_lora.frozen_config(item['config']) for item in plan['candidates']]
    state={'status':'starting','controller_pid':os.getpid(),'current_pid':None,'completed':[]};active=None;stopped=False;handlers={}
    def save():state['updated_unix_seconds']=time.time();write_json(directory/'state.json',state)
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        if active is not None:
            try:active.send_signal(signal.SIGTERM)
            except ProcessLookupError:pass
    ledger_path=root/'runs/budget_ledger.json'
    def total_cost():
        ledger=read_json(ledger_path) if ledger_path.exists() else {'runs':{}}
        return sum(float(row['estimated_compute_cny']) for row in ledger['runs'].values())
    cost_before=total_cost()
    def child(arguments,log,timeout,phase):
        nonlocal active
        if stopped:return -signal.SIGTERM
        if total_cost()-cost_before>=plan['pipeline_compute_cap_cny']:raise RuntimeError('LoRA pipeline compute cap reached')
        with Path(log).open('ab') as stream:
            active=subprocess.Popen([str(root/'.venv/bin/python'),'-u',*arguments],cwd=root,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
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
                    code=124;break
        active=None;state['current_pid']=None;save();return code
    for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,stop)
    try:
        save();results=[]
        for item,config in zip(plan['candidates'],configurations):
            if stopped:state['status']='stopped';return state
            state['current_candidate']=item['key'];save()
            command=[str(root/'scripts/train_lora.py')]
            log=Path(item['log']);log.parent.mkdir(parents=True,exist_ok=True)
            check=child(command+['check','--config',item['config']],log,180,'preflight')
            if check or stopped:state.update(status='stopped' if stopped else 'failed_preflight',returncode=check);return state
            code=child(command+['probe','--config',item['config'],'--steps',str(plan['screen_step'])],log,
                config['budget']['max_runtime_seconds']+600,'screening')
            if code or stopped:state.update(status='stopped' if stopped else 'screen_failed',returncode=code);return state
            summary=read_json(Path(config['output_dir'])/'summary.json');value=screen_result(summary,plan['screen_step'])
            result={**item,'development_loss':value,'primary_validation_loss':summary['latest_validation']['macro_loss'],
                    'retention':summary['latest_retention'],'step':summary['step']}
            results.append(result);state['completed']=results;save()
        chosen=select_candidate(results);state['selected']=chosen['key'];save()
        config=train_lora.frozen_config(chosen['config']);selection={'status':'configuration_locked_before_final_test',
            'config':chosen['config'],'config_sha256':sha256_file(chosen['config']),'screen_results':results,
            'selection_metric':'auxiliary development macro loss at identical screen step','test_not_read_for_selection':True,
            'full_checkpoint':plan['full_checkpoint'],'full_checkpoint_sha256':plan['full_checkpoint_sha256'],
            'final_output_dir':plan['final_output_dir'],'final_step':plan['final_step']}
        selection_path=directory/'selection.json';write_json(selection_path,selection)
        code=child([str(root/'scripts/train_lora.py'),'resume','--config',chosen['config']],Path(chosen['log']),
            config['budget']['max_runtime_seconds']+600,'continuing_selected')
        if code or stopped:state.update(status='stopped' if stopped else 'continued_training_failed',returncode=code);return state
        summary=read_json(Path(config['output_dir'])/'summary.json')
        if summary['status']!='completed' or summary['step']!=plan['final_step']:raise ValueError('Winner did not complete planned updates')
        code=child([str(root/'scripts/evaluate_lora.py'),'--selection',str(selection_path)],directory/'final_evaluation.log',
            5700,'final_test')
        if code or stopped:state.update(status='stopped' if stopped else 'final_test_failed',returncode=code);return state
        final=read_json(Path(plan['final_output_dir'])/'comparison.json')
        if final['status']!='completed' or final['generated_examples_per_model']!=1000:raise ValueError('Final test incomplete')
        state.update(status='completed',final_comparison=str(Path(plan['final_output_dir'])/'comparison.json'))
        return state
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,reason=str(error));raise
    finally:
        save()
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);args=parser.parse_args()
    result=run(args.plan);print(json.dumps(result,ensure_ascii=False),flush=True)
    raise SystemExit(0 if result['status']=='completed' else 2)
