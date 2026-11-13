"""自研300M的DPO/PPO/GRPO独立入口：prepare/check/start/resume。"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from copy import deepcopy
import argparse,json,shlex,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_reading_independent.comparison import ComparisonRLCorpus
from llm_rl.config import DEFAULT_TRAIN,read_config,validate,runtime_identity
from llm_rl.data import PreferenceCorpus
from train_sft import training_lock,gpu_idle,require_linux


def prepare(plan_path,arm_name,algorithm,preferences=None,calibration=None,stamp=None,micro=16):
    from tokenizers import Tokenizer
    plan_path=Path(plan_path);plan=read_json(plan_path);root=ROOT;spec=plan['arms'][arm_name]
    if not plan.get('launch_enabled'):raise ValueError('Training is paused in this plan')
    arm_path=plan_path.parent/spec['manifest']
    if sha256_file(arm_path)!=spec['sha256']:raise ValueError('Frozen comparison arm changed')
    arm=ComparisonRLCorpus(arm_path);tokenizer=Tokenizer.from_file(str(arm.source.tokenizer_path))
    initial=plan['initial_checkpoint']
    if sha256_file(initial['path'])!=initial['sha256']:raise ValueError('Initial SFT checkpoint changed')
    if algorithm=='dpo':
        if preferences is None:raise ValueError('DPO preference publication is still pending')
        corpus=PreferenceCorpus(preferences,arm,tokenizer,initial['sha256'])
    else:corpus=arm
    train=deepcopy(DEFAULT_TRAIN);train.update(score_micro_batch_size=micro,group_size=4 if algorithm=='grpo' else 1,
        ppo_updates=plan['shared_settings']['ppo_optimization_passes_per_rollout'])
    if algorithm=='dpo':train['learning_rate']=5e-6
    stamp=stamp or datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    label={'old_matched':'old25k','new_matched':'new25k','old_full_50k':'old50k'}[arm_name]
    name=f'{algorithm}_300m_{label}_n{len(corpus)}_pb16_mb{micro}_lr{train["learning_rate"]:.0e}_'+(
        f'g{train["group_size"]}_' if algorithm=='grpo' else f'k{train["ppo_updates"]}_' if algorithm=='ppo' else 'beta0p1_')+f'ep1_seed42_{stamp}'
    name=name.replace('e-0','e-');train['wandb_run_name']=name
    cfg={'schema_version':1,'stage':'reading_rl','algorithm':algorithm,'project_root':str(root),
       'initial_checkpoint':initial['path'],'initial_checkpoint_sha256':initial['sha256'],
       'arm_manifest':str(arm_path.resolve()),'arm_manifest_sha256':sha256_file(arm_path),
       'validation_manifest':str((root/'artifacts/posttrain/reading_qa_50k_zh_en_20261101_v3/manifest.json').resolve()),
       'output_dir':str(root/'runs'/name),'preference_manifest':str(Path(preferences).resolve()) if preferences else None,
       'train_examples':len(corpus),'fixture':False,'min_free_bytes':8*2**30,'train':train,
       'budget':{'hourly_rate_cny':3.03,'run_limit_cny':12,'project_limit_cny':500,'reserve_cny':100,
                 'max_runtime_seconds':14400,'ledger_path':str(root/'runs/budget_ledger.json')},
       'runtime_sha256':canonical_hash(runtime_identity(root)),
       'experiment':{'plan':str(plan_path.resolve()),'plan_sha256':sha256_file(plan_path),'arm':arm_name,
                     'source_prompt_examples':len(arm),'test_used_for_selection':False},
       'calibration_report':str(Path(calibration).resolve()) if calibration else None,
       'calibration_report_sha256':sha256_file(calibration) if calibration else None}
    validate(cfg);target=root/'runs/configs'/(name+'.json');log=root/'runs/logs'/(name+'.log')
    if target.exists() or Path(cfg['output_dir']).exists():raise ValueError('Run identity already exists')
    write_json(target,cfg);write_json(target.with_suffix('.identity.json'),{'config_sha256':sha256_file(target),'runtime':runtime_identity(root)})
    target.with_suffix('.env').write_text('\n'.join(k+'='+shlex.quote(str(v)) for k,v in {
         'RUN_CONFIG':target,'RUN_LOG':log,'RUN_DIR':cfg['output_dir']}.items())+'\n',encoding='utf-8')
    return {'config':str(target),'output':cfg['output_dir'],'log':str(log),'env':str(target.with_suffix('.env'))}


def checked(path):
    path=Path(path);receipt=read_json(path.with_suffix('.identity.json'))
    if sha256_file(path)!=receipt['config_sha256']:raise ValueError('Frozen training config changed')
    cfg=read_config(path)
    if runtime_identity(cfg['project_root'])!=receipt['runtime']:raise ValueError('Frozen runtime changed')
    if sha256_file(cfg['initial_checkpoint'])!=cfg['initial_checkpoint_sha256']:raise ValueError('Initial weights changed')
    if sha256_file(cfg['arm_manifest'])!=cfg['arm_manifest_sha256']:raise ValueError('Frozen data arm changed')
    if sha256_file(cfg['experiment']['plan'])!=cfg['experiment']['plan_sha256']:raise ValueError('Comparison plan changed')
    if not cfg['fixture']:
        if not cfg.get('calibration_report'):raise ValueError('GPU numerical/development calibration required')
        calibration=read_json(cfg['calibration_report'])
        if sha256_file(cfg['calibration_report'])!=cfg['calibration_report_sha256']:raise ValueError('Frozen calibration changed')
        if calibration.get('runtime_sha256')!=cfg['runtime_sha256']:raise ValueError('Calibration used a different numerical runtime')
        if calibration.get('status')!='passed' or calibration['initial_checkpoint_sha256']!=cfg['initial_checkpoint_sha256']:
            raise ValueError('Invalid initial-policy calibration')
        if cfg['algorithm'] in {'ppo','grpo'}:
            signals=calibration['arms'][cfg['experiment']['arm']]
            if signals['correct']==0 or cfg['algorithm']=='grpo' and signals['mixed_reward_groups']==0:
                raise ValueError('No verified learning signal for this arm/algorithm')
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','check','start','resume'])
    p.add_argument('--plan',type=Path);p.add_argument('--arm',choices=['old_matched','new_matched','old_full_50k'])
    p.add_argument('--algorithm',choices=['dpo','ppo','grpo']);p.add_argument('--preferences',type=Path)
    p.add_argument('--calibration',type=Path);p.add_argument('--config',type=Path);p.add_argument('--stamp');p.add_argument('--micro',type=int,default=16)
    a=p.parse_args()
    if a.action=='prepare':result=prepare(a.plan,a.arm,a.algorithm,a.preferences,a.calibration,a.stamp,a.micro)
    else:
        cfg=checked(a.config)
        if a.action=='check':result={'status':'ready','algorithm':cfg['algorithm'],'epochs':1,'train_examples':cfg['train_examples']}
        else:
            require_linux();from llm_rl.engine import run
            with training_lock(ROOT):
                gpu_idle();result=run(cfg,resume=Path(cfg['output_dir'])/'last.pt' if a.action=='resume' else None)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
