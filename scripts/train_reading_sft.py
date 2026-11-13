"""阅读理解领域全参数 SFT：冻结配置、单轮计划、共享GPU锁及精确断点恢复。"""
import argparse
from datetime import datetime,timedelta,timezone
import json,math,re,shlex,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_sft.config import read_config
from llm_sft.data import SFTCorpus
from llm_align.full_sft import runtime_identity,validate,run
from train_sft import training_lock,gpu_idle,require_linux

INITIAL_FULL='4a36a0862c9d0527a7ec3737c7bc9ed5c107a667d4c3378b0e14d120241c6060'


def accepted_data(config):
    """训练入口仅加载 train/validation；测试题留给独立评测。"""
    validate(config)
    if config.get('experiment',{}).get('domain')!='extractive_reading_qa':
        raise ValueError('This entry requires the reading QA domain')
    manifest=read_json(config['data_manifest'])
    if manifest.get('status')!='published' or manifest.get('task')!='extractive_reading_qa':
        raise ValueError('Reading QA data must be independently audited and published')
    corpus=SFTCorpus(config['data_manifest'],'train')
    validation=SFTCorpus(config['data_manifest'],'validation')
    if len(corpus)!=50000 or len(validation)!=2000:raise ValueError('Wrong accepted train/development sizes')
    verifier=Path(config['project_root'])/'src/llm_reading_data/verifier.py'
    if sha256_file(verifier)!=manifest['reward_protocol']['code_sha256']:
        raise ValueError('Published reward protocol changed')
    return corpus,validation


def plan(config,examples):
    train=config['train'];batch=train['micro_batch_size']*train['gradient_accumulation_steps']
    steps=math.ceil(examples/batch)
    return {'epochs':1,'stop_epochs':None,'train_examples':examples,'effective_batch_size':batch,
            'max_steps':steps,'warmup_steps':math.ceil(steps*train['warmup_ratio'])}


def prepare(template,stamp=None):
    cfg=read_config(template);corpus,_=accepted_data(cfg);root=Path(cfg['project_root'])
    source=Path(cfg['base_checkpoint']);manifest=Path(cfg['data_manifest'])
    if sha256_file(source)!=INITIAL_FULL:raise ValueError('Use the selected completed Full-SFT parent')
    cfg.update(base_checkpoint_sha256=INITIAL_FULL,data_manifest_sha256=sha256_file(manifest),
               data_identity=corpus.identity['data_identity'],runtime_sha256=canonical_hash(runtime_identity(root)))
    stamp=stamp or datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    if not re.fullmatch(r'\d{8}_\d{6}',stamp):raise ValueError('Timestamp must be YYYYMMDD_HHMMSS')
    train=cfg['train'];micro=train['micro_batch_size'];accum=train['gradient_accumulation_steps']
    lr=format(train['learning_rate'],'.0e').replace('e-0','e-')
    name=f'reading_sft_300m_full_data{len(corpus)}_lr{lr}_mb{micro}_ga{accum}_b{micro*accum}_ctx1024_ep1_{stamp}'
    cfg['output_dir']=str(root/'runs'/name);cfg['train']['wandb_run_name']=name
    target=root/'runs/configs'/(name+'.json');log=root/'runs/logs'/(name+'.log')
    if target.exists() or Path(cfg['output_dir']).exists():raise ValueError('Run identity already exists')
    write_json(target,cfg)
    write_json(target.with_suffix('.identity.json'),{'config_sha256':sha256_file(target),'runtime':runtime_identity(root),
        'initialization':'completed full-SFT model','plan':plan(cfg,len(corpus)),'test_used_for_selection':False})
    target.with_suffix('.env').write_text('\n'.join(k+'='+shlex.quote(str(v)) for k,v in {
        'RUN_CONFIG':target,'RUN_DIR':cfg['output_dir'],'RUN_LOG':log}.items())+'\n',encoding='utf-8')
    return {'config':str(target),'output':cfg['output_dir'],'log':str(log),'env':str(target.with_suffix('.env')),
            'plan':plan(cfg,len(corpus)),'training_started':False}


def checked(path):
    path=Path(path);receipt=read_json(path.with_suffix('.identity.json'))
    if sha256_file(path)!=receipt['config_sha256']:raise ValueError('Frozen reading configuration changed')
    cfg=read_config(path);corpus,_=accepted_data(cfg)
    if runtime_identity(Path(cfg['project_root']))!=receipt['runtime']:raise ValueError('Frozen runtime changed')
    if cfg['base_checkpoint_sha256']!=INITIAL_FULL or sha256_file(cfg['base_checkpoint'])!=INITIAL_FULL:
        raise ValueError('Selected completed Full-SFT parent changed')
    if corpus.identity['manifest_sha256']!=cfg['data_manifest_sha256'] or corpus.identity['data_identity']!=cfg['data_identity']:
        raise ValueError('Published reading data changed')
    if plan(cfg,len(corpus))!=receipt['plan']:raise ValueError('One-epoch update plan changed')
    return cfg


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','check','start','resume'])
    ap.add_argument('--template',type=Path);ap.add_argument('--config',type=Path);ap.add_argument('--stamp')
    args=ap.parse_args()
    if args.action=='prepare':
        if args.template is None:ap.error('prepare requires --template')
        result=prepare(args.template,args.stamp)
    else:
        if args.config is None:ap.error('check/start/resume requires --config')
        cfg=checked(args.config)
        if args.action=='check':
            result={'status':'ready','plan':plan(cfg,50000),'source_checkpoint':cfg['base_checkpoint'],'data':cfg['data_manifest']}
        else:
            require_linux()
            with training_lock(Path(cfg['project_root'])):
                gpu_idle();result=run(cfg,resume=Path(cfg['output_dir'])/'last.pt' if args.action=='resume' else None)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
