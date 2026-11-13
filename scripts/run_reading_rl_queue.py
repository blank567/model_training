"""九组独立RL顺序队列；等待扩容、绑定进程身份，失败即停，不删除任何权重。"""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from train_reading_rl import prepare,checked


def process_alive(pid,ticks):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split()
        return fields[2]!='Z' and fields[21]==str(ticks)
    except FileNotFoundError:return False


def case_finished(spec):
    out=Path(spec['output']);summary=out/'summary.json'
    if (out/'failure.json').exists():raise RuntimeError('Training failed: '+str(out/'failure.json'))
    if not summary.exists():return False
    result=read_json(summary)
    if result['status']!='completed':raise RuntimeError('Training stopped before one epoch: '+result['status'])
    if result['examples_seen']!=result['plan']['examples'] or result['data_batch_cursor']!=result['plan']['data_batches']:
        raise RuntimeError('Completed summary does not match one-epoch plan')
    if result['optimizer_step']!=result['plan']['optimizer_steps']:raise RuntimeError('Optimizer plan incomplete')
    if not all(Path(result[k]).is_file() for k in ['best_checkpoint','last_checkpoint']):raise RuntimeError('Required model output is missing')
    return True


def free_required(kind):
    return 8*2**30+(7*303350784*4 if kind=='train' else 2*2**30)


def launch(command,log):
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21]
    return {'pid':process.pid,'process_start_ticks':ticks,'command':command}


def run(path):
    state=read_json(path);control=path.parent;started=time.monotonic();children=[]
    if state['queue_code_sha256']!=sha256_file(__file__):raise ValueError('Queue control code changed')
    if sha256_file(state['plan'])!=state['plan_sha256']:raise ValueError('Frozen comparison plan changed')
    with (control/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(control/'daemon.json',{'pid':os.getpid(),'process_start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]})
        try:
            for index,job in enumerate(state['jobs']):
                state['current_job']=index;state['updated_unix']=time.time();write_json(path,state)
                if job.get('status')=='completed':continue
                if (control/'STOP').exists():state['status']='stopped_before_next_job';write_json(path,state);return
                if job['kind']=='train' and 'config' not in job:
                    preferences=read_json(Path(state['preference_output'])/'preference_index.json')
                    if preferences.get('status')!='published' or preferences['plan_sha256']!=state['plan_sha256']:
                        raise ValueError('DPO preference publication is not ready for this plan')
                    job.update(prepare(state['plan'],job['arm'],job['algorithm'],
                        preferences=preferences['arms'][job['arm']]['manifest'],calibration=state['calibration'],micro=16))
                    write_json(path,state)
                if job['kind']=='train':checked(job['config'])
                if job.get('status')!='running':
                    waiting=time.monotonic()
                    while shutil.disk_usage(ROOT).free<free_required(job['kind']):
                        state.update(status='waiting_for_disk',disk_free_gib=shutil.disk_usage(ROOT).free/2**30,
                            disk_required_gib=free_required(job['kind'])/2**30);write_json(path,state)
                        if (control/'STOP').exists():state['status']='stopped_before_next_job';write_json(path,state);return
                        if time.monotonic()-waiting>=3600:
                            state['status']='waiting_for_disk_queue_exited';write_json(path,state);return
                        time.sleep(20)
                    if job['kind']=='train':command=[sys.executable,'-u','scripts/train_reading_rl.py','start','--config',job['config']]
                    else:command=[sys.executable,'-u','scripts/prepare_reading_dpo_pairs.py','--plan',state['plan'],
                        '--output',state['preference_output'],'--work',state['preference_work']]
                    job.update(launch(command,job['log']),status='running',started_unix=time.time());write_json(path,state)
                state['status']='running';write_json(path,state)
                while process_alive(job['pid'],job['process_start_ticks']):
                    time.sleep(10)
                if job['kind']=='train':
                    if not case_finished(job):raise RuntimeError('Process exited without a completed summary: '+job['log'])
                else:
                    receipt=Path(state['preference_output'])/'preference_index.json'
                    if not receipt.exists() or read_json(receipt).get('status')!='published':raise RuntimeError('Preference generation failed: '+job['log'])
                job['status']='completed';job['finished_unix']=time.time();write_json(path,state)
                print(json.dumps({'completed_job':index,'algorithm':job.get('algorithm'),'arm':job.get('arm')}),flush=True)
            state['status']='all_nine_completed';write_json(path,state)
        except BaseException as error:
            state.update(status='failed_requires_review',error_type=type(error).__name__,error=str(error));write_json(path,state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);a=p.parse_args()
    if not a.state.resolve().is_relative_to(ROOT/'runs/assistant/system'):raise ValueError('Queue state must stay in assistant system records')
    run(a.state)
if __name__=='__main__':main()
