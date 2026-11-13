"""自研300M LoRA入口：冻结配置、共用GPU锁、受限短测与完整恢复。"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import shlex
import shutil

from llm_lab.common.io import read_json,sha256_file,write_json
from llm_sft.config import plan_details,under,GIB
from llm_lora.config import read_config,as_full,runtime_fingerprint,runtime_identity,FROZEN_FULL_RUNTIME
from llm_lora.data import protocol
import train_sft as legacy


def name_for(config,stamp):
    # 参数与数据规模进入目录，避免与既有Full实验混淆。
    original=legacy.experiment_name(as_full(config),stamp)
    rows=read_json(config['data_manifest'])['splits'][config['train_split']]['examples']
    adapter=config['adapter'];tag=f"r{adapter['rank']}_a{legacy._number_token(adapter['alpha'])}_{adapter['target_scope']}"
    return original.replace('sft_300m_full_',f'lora_300m_data{rows}_{tag}_')


def sources(config):
    view=as_full(config);base=legacy.base_identity(view);data=legacy.dataset_identity(view)
    current=runtime_fingerprint();info=runtime_identity()
    if not config['fixture'] and info['full_sft_runtime_sha256']!=FROZEN_FULL_RUNTIME:raise ValueError('Frozen Full source changed')
    if info['packages']['peft']!='0.17.1':raise ValueError('PEFT must be pinned to 0.17.1')
    for key,value in (('base_checkpoint_sha256',base['base_checkpoint_sha256']),
        ('data_manifest_sha256',data['manifest_sha256']),('data_identity',data['data_identity']),('runtime_sha256',current)):
        if config.get(key) and config[key]!=value:raise ValueError('Frozen LoRA source changed: '+key)
    protocol(config)
    return {**base,**data,'runtime_sha256':current,'evaluation_protocol_sha256':config.get('evaluation_protocol_sha256')}


def prepare(template,stamp=None):
    config=read_config(template);identity=sources(config);root=Path(config['project_root'])
    stamp=stamp or datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    name=name_for(config,stamp);output=root/'runs'/name
    if output.exists():raise FileExistsError('Run already exists')
    config['output_dir']=str(output);config['train']['wandb_run_name']=name
    config.update(base_checkpoint_sha256=identity['base_checkpoint_sha256'],data_manifest_sha256=identity['manifest_sha256'],
        data_identity=identity['data_identity'],runtime_sha256=identity['runtime_sha256'])
    frozen=root/'runs/configs'/(name+'.json');related=legacy.paths(frozen)
    if any(path.exists() or path.is_symlink() for path in related.values()):raise FileExistsError('Frozen LoRA config already exists')
    frozen.parent.mkdir(parents=True,exist_ok=True)
    # 明确拒绝覆盖，后续只允许按原配置resume。
    with frozen.open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,ensure_ascii=False,indent=2);stream.write('\n')
    frozen_identity={**identity,'name':name,'stamp':stamp,'config_sha256':sha256_file(frozen),
        'output_dir':str(output),'plan':plan_details(config)}
    with related['identity'].open('x',encoding='utf-8',newline='\n') as stream:json.dump(frozen_identity,stream,ensure_ascii=False,indent=2);stream.write('\n')
    values={'RUN_NAME':name,'RUN_CONFIG':str(frozen),'RUN_DIR':str(output),'RUN_LOG':str(root/'runs/logs'/(name+'.log'))}
    text=f"#!/usr/bin/env bash\ncd {shlex.quote(str(root))}\nsource {shlex.quote(str(root/'scripts/env_remote.sh'))}\n"
    text+='export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
    text+='mkdir -p '+shlex.quote(str(root/'runs/logs'))+'\n'
    text+=''.join(f'export {key}={shlex.quote(value)}\n' for key,value in values.items())
    with related['env'].open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    return {**values,'env':str(related['env']),'identity':str(related['identity']),'plan':frozen_identity['plan'],'gpu_training_started':False}


def frozen_config(path):
    path=Path(path).resolve();config=read_config(path);identity=read_json(path.with_suffix('.identity.json'))
    if sha256_file(path)!=identity['config_sha256']:raise ValueError('LoRA config changed')
    expected=name_for(config,identity['stamp']);root=Path(config['project_root'])
    if identity['name']!=expected or path.name!=expected+'.json' or Path(config['output_dir'])!=root/'runs'/expected:
        raise ValueError('LoRA name/output identity changed')
    if config['train']['wandb_run_name']!=expected:raise ValueError('LoRA tracking name changed')
    actual=sources(config)
    if any(identity.get(key)!=value for key,value in actual.items()):raise ValueError('LoRA frozen source identity changed')
    if identity['plan']!=plan_details(config):raise ValueError('LoRA plan changed')
    return config


def preflight(config,action):
    if legacy.training_processes():raise RuntimeError('Another training process is active')
    output=Path(config['output_dir'])
    if action in {'start','probe'} and output.exists() and any(output.iterdir()):raise FileExistsError('Nonempty run; use resume')
    if action=='resume' and not (output/'last.pt').is_file():raise ValueError('Complete Adapter last is required')
    if config['train']['device']=='cuda':legacy.gpu_idle();legacy.check_cuda(config)
    # 生产结构固定；每个Adapter参数预留权重、AdamW、原子临时文件及导出副本。
    rank=config['adapter']['rank'];per_rank=436224 if config['adapter']['target_scope']=='all' else 79872
    needed=config['min_free_bytes']+rank*per_rank*40+32*1024**2
    free=shutil.disk_usage(config['project_root']).free
    if free<needed:raise OSError('Insufficient Adapter checkpoint reserve')
    ledger_path=Path(config['budget']['ledger_path']);ledger=read_json(ledger_path) if ledger_path.exists() else {'runs':{}}
    recorded=sum(float(row['estimated_compute_cny']) for row in ledger['runs'].values())
    previous=float(ledger['runs'].get(str(output.resolve()),{}).get('estimated_compute_cny',0))
    if recorded>=config['budget']['project_limit_cny']-config['budget']['reserve_cny'] or previous>=config['budget']['run_limit_cny']:
        raise RuntimeError('Recorded LoRA/project budget exhausted')
    return {'gpu_training_started':False,'free_gib':free/GIB,'required_free_gib':needed/GIB,
        'recorded_project_compute_cny':recorded,'plan':plan_details(config)}


def execute(action,path,steps=None):
    config=frozen_config(path);plan=plan_details(config)
    if config['train']['device']=='cuda':legacy.require_linux()
    if steps is not None and (type(steps) is not int or not 0<steps<plan['stop_step']):raise ValueError('Pause step must be inside frozen plan')
    lock=legacy.training_lock(config['project_root']) if config['train']['device']=='cuda' else contextlib.nullcontext()
    with lock:
        checks=preflight(config,action)
        if action=='check':return checks
        from llm_lora.engine import run_lora
        import os
        state={'status':'running','action':action,'pid':os.getpid(),'config':str(Path(path).resolve())}
        state_path=Path(path).with_suffix('.state.json');write_json(state_path,state)
        try:
            result=run_lora(config,resume=Path(config['output_dir'])/'last.pt' if action=='resume' else None,stop_after_steps=steps)
            write_json(state_path,{**state,'status':'exited','summary':result});return result
        except BaseException as error:
            write_json(state_path,{**state,'status':'failed','error_type':type(error).__name__});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    item=commands.add_parser('prepare');item.add_argument('--template',required=True);item.add_argument('--stamp')
    for action in ('check','start','probe','resume','status'):
        item=commands.add_parser(action);item.add_argument('--config',required=True)
        if action=='probe':item.add_argument('--steps',type=int,default=32)
        if action=='resume':item.add_argument('--until-step',type=int)
    args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.template,args.stamp)
    elif args.action=='status':
        cfg=read_config(args.config);path=Path(cfg['output_dir'])/'summary.json';result=read_json(path) if path.exists() else {'status':'not_started'}
    else:result=execute(args.action,args.config,getattr(args,'steps',getattr(args,'until_step',None)))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
