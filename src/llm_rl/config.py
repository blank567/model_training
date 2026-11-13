"""单轮后训练配置及恢复身份；预算与磁盘检查在实际运行前执行。"""
from pathlib import Path
import copy,math
from llm_lab.common.io import canonical_hash,read_json,sha256_file
from llm_sft.config import under,runtime_fingerprint

ROOT=Path(__file__).resolve().parents[2]
CORE_FILES=['src/llm_rl/generation.py','src/llm_rl/scoring.py','src/llm_rl/objectives.py','src/llm_rl/data.py',
 'src/llm_rl/evaluation.py','src/llm_rl/config.py','src/llm_rl/engine.py','scripts/train_reading_rl.py',
 'src/llm_reading_independent/comparison.py','src/llm_reading_data/verifier.py','scripts/chat_sft.py']
FROZEN_FULL='0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872'
DEFAULT_TRAIN={'device':'cuda','dtype':'bf16','seed':42,'deterministic':True,'epochs':1,'stop_epochs':None,
 'prompt_batch_size':16,'score_micro_batch_size':16,'group_size':4,'grpo_iterations':1,'ppo_updates':2,
 'max_new_tokens':64,'learning_rate':1e-6,'critic_learning_rate':1e-4,'beta':.1,'clip_range':.2,
 'value_clip':.2,'value_coefficient':.5,'kl_coefficient':.02,'gamma':1.,'gae_lambda':.95,
 'warmup_ratio':.03,'min_lr_ratio':.1,'weight_decay':0.,'betas':[.9,.95],'max_grad_norm':1.,
 'eval_size':256,'eval_batch_size':16,'eval_interval':100,'save_interval':100,'log_interval':10,
 'tensorboard':True,'wandb':True,'wandb_mode':'online','wandb_project':'bilingual-training-lab',
 'wandb_entity':'mblankm-peking-university','wandb_run_name':None}


def runtime_identity(root=ROOT):
    root=Path(root)
    return {'files':{name:sha256_file(root/name) for name in CORE_FILES},'frozen_full_runtime':runtime_fingerprint(root)}


def validate(cfg):
    allowed={'schema_version','stage','algorithm','project_root','initial_checkpoint','initial_checkpoint_sha256',
        'arm_manifest','arm_manifest_sha256','validation_manifest','output_dir','preference_manifest','train_examples',
        'fixture','min_free_bytes','train','budget','runtime_sha256','experiment','calibration_report','calibration_report_sha256'}
    if set(cfg)-allowed or set(cfg.get('train',{}))-set(DEFAULT_TRAIN):raise ValueError('Unknown posttraining configuration fields')
    if cfg.get('schema_version')!=1 or cfg.get('stage')!='reading_rl' or cfg.get('algorithm') not in {'dpo','ppo','grpo'}:
        raise ValueError('Unsupported posttraining configuration')
    t=cfg['train']
    if t['epochs']!=1 or t.get('stop_epochs') is not None:raise ValueError('Use exactly epochs=1 and stop_epochs=null')
    if cfg['algorithm']=='grpo' and t['grpo_iterations']!=1:raise ValueError('Only one GRPO iteration was authorized')
    if cfg['algorithm']=='ppo' and t['ppo_updates'] not in {1,2}:raise ValueError('PPO internal updates must be explicitly selected')
    if t['device'] not in {'cpu','cuda'} or t['dtype']!=('bf16' if t['device']=='cuda' else 'fp32'):
        raise ValueError('Use CUDA BF16 or explicit CPU FP32 fixture')
    if not cfg['fixture'] and t['device']!='cuda':raise ValueError('Production training requires remote CUDA')
    for name in ['prompt_batch_size','score_micro_batch_size','eval_interval','save_interval','log_interval','eval_size']:
        if type(t[name]) is not int or t[name]<=0:raise ValueError('Positive integer required: '+name)
    if t['eval_size']>2000 or t['eval_size']%2:raise ValueError('Balanced development subset required')
    for name in ['learning_rate','critic_learning_rate','beta','clip_range','value_clip','max_grad_norm']:
        if isinstance(t[name],bool) or not isinstance(t[name],(int,float)) or not math.isfinite(t[name]) or t[name]<=0:
            raise ValueError('Positive finite value required: '+name)
    if not 0<=t['warmup_ratio']<=1 or not 0<=t['kl_coefficient']<10:raise ValueError('Invalid schedule/KL')
    if t['max_new_tokens']!=64 or (t['group_size']<2 if cfg['algorithm']=='grpo' else t['group_size']!=1):
        raise ValueError('Keep GRPO groups>=2 and one PPO rollout per prompt')
    if cfg['algorithm']=='dpo' and not cfg.get('preference_manifest'):raise ValueError('DPO preferences are not ready')
    if type(cfg.get('fixture')) is not bool:raise ValueError('Explicit fixture/production identity required')


def read_config(path):
    path=Path(path);cfg=read_json(path);root=Path(cfg.get('project_root',ROOT)).resolve();cfg=copy.deepcopy(cfg)
    cfg['project_root']=str(root)
    for key in ['initial_checkpoint','arm_manifest','validation_manifest','output_dir']:
        cfg[key]=str(under(cfg[key],root))
    if cfg.get('preference_manifest'):cfg['preference_manifest']=str(under(cfg['preference_manifest'],root))
    cfg['budget']['ledger_path']=str(under(cfg['budget']['ledger_path'],root));validate(cfg);return cfg
