"""数学全参数 SFT 的独立 prepare/check/start/resume；保持历史训练身份不变。"""
import argparse
import copy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_sft.config import read_config
from llm_sft.data import SFTCorpus
from llm_align.math_sft import runtime_identity,validate,run
from train_sft import training_lock,gpu_idle,require_linux


def prepare(template,stamp=None):
    cfg=read_config(template);validate(cfg);root=Path(cfg['project_root'])
    source=Path(cfg['base_checkpoint']);manifest=Path(cfg['data_manifest']);data=read_json(manifest)
    if data.get('status')!='published':raise ValueError('Public math data is not accepted/published')
    corpus=SFTCorpus(manifest,'train')
    cfg['base_checkpoint_sha256']=sha256_file(source);cfg['data_manifest_sha256']=sha256_file(manifest)
    cfg['data_identity']=corpus.identity['data_identity'];cfg['runtime_sha256']=canonical_hash(runtime_identity(root))
    stamp=stamp or datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    train=cfg['train'];micro=train['micro_batch_size'];accum=train['gradient_accumulation_steps']
    lr=format(train['learning_rate'],'.0e').replace('e-0','e-')
    name=f'math_sft_300m_full_public{len(corpus)}_lr{lr}_mb{micro}_ga{accum}_b{micro*accum}_ep1_{stamp}'
    cfg['output_dir']=str(root/'runs'/name);cfg['train']['wandb_run_name']=name
    target=root/'runs/configs'/(name+'.json');log=root/'runs/logs'/(name+'.log')
    if target.exists() or Path(cfg['output_dir']).exists():raise ValueError('Run identity already exists')
    write_json(target,cfg)
    write_json(target.with_suffix('.identity.json'),{'config_sha256':sha256_file(target),'runtime':runtime_identity(root),
        'initialization':'completed full-SFT model','epochs':1,'stop_epochs':None})
    target.with_suffix('.env').write_text('\n'.join(k+'='+shlex.quote(str(v)) for k,v in {
        'RUN_CONFIG':target,'RUN_DIR':cfg['output_dir'],'RUN_LOG':log}.items())+'\n',encoding='utf-8')
    return {'config':str(target),'output':cfg['output_dir'],'log':str(log),'env':str(target.with_suffix('.env'))}


def checked(path):
    path=Path(path);receipt=read_json(path.with_suffix('.identity.json'))
    if sha256_file(path)!=receipt['config_sha256']:raise ValueError('Frozen math config changed')
    cfg=read_config(path);validate(cfg)
    if runtime_identity(Path(cfg['project_root']))!=receipt['runtime']:raise ValueError('Frozen runtime changed')
    if sha256_file(cfg['base_checkpoint'])!=cfg['base_checkpoint_sha256']:raise ValueError('Initial full-SFT weights changed')
    corpus=SFTCorpus(cfg['data_manifest'],'train')
    if corpus.identity['manifest_sha256']!=cfg['data_manifest_sha256']:raise ValueError('Math data changed')
    return cfg


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','check','start','resume'])
    ap.add_argument('--template',type=Path);ap.add_argument('--config',type=Path);ap.add_argument('--stamp')
    args=ap.parse_args()
    if args.action=='prepare':result=prepare(args.template,args.stamp)
    else:
        cfg=checked(args.config)
        if args.action=='check':result={'status':'ready','epochs':1,'source_checkpoint':cfg['base_checkpoint'],'data':cfg['data_manifest']}
        else:
            require_linux()
            with training_lock(Path(cfg['project_root'])):
                gpu_idle();result=run(cfg,resume=Path(cfg['output_dir'])/'last.pt' if args.action=='resume' else None)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
