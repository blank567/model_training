"""LoRA配置与运行身份；借用已有SFT数值约束，不改旧协议。"""
from __future__ import annotations

import copy
from importlib import metadata
import math
from pathlib import Path

from llm_lab.common.io import canonical_hash,read_json,sha256_file
from llm_sft import config as full

FROZEN_FULL_RUNTIME='0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872'
ADAPTER_DEFAULTS={'rank':16,'alpha':32,'dropout':0.0,'target_scope':'all','bias':'none'}
EXTRA_FIELDS={'adapter','evaluation_protocol','evaluation_protocol_sha256'}


def as_full(config):
    result={key:copy.deepcopy(value) for key,value in config.items() if key not in EXTRA_FIELDS}
    result['method']='full'
    return result


def validate(config):
    if config.get('method')!='lora' or set(config)-full.TOP_FIELDS-EXTRA_FIELDS:
        raise ValueError('Only custom LoRA SFT configuration is supported')
    full.validate_config(as_full(config))
    adapter=config['adapter']
    if set(adapter)!=set(ADAPTER_DEFAULTS) or type(adapter['rank']) is not int or not 1<=adapter['rank']<=256:
        raise ValueError('Invalid LoRA rank or fields')
    for name in ('alpha','dropout'):
        value=adapter[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Invalid LoRA numeric setting')
    if adapter['alpha']<=0 or not 0<=adapter['dropout']<1 or adapter['bias']!='none' or adapter['target_scope'] not in {'all','qv'}:
        raise ValueError('Invalid LoRA scale/dropout/scope/bias')
    if not config['fixture'] and not config.get('evaluation_protocol'):
        raise ValueError('Production LoRA requires a frozen evaluation protocol')


def read_config(path):
    path=Path(path).resolve();raw=read_json(path)
    if set(raw)-full.TOP_FIELDS-EXTRA_FIELDS:raise ValueError('Unknown LoRA config fields')
    root=(path.parent/raw.get('project_root','../..')).resolve();result=copy.deepcopy(raw)
    for name,default in (('schema_version',1),('stage','sft'),('method','lora'),('fixture',False),
                         ('min_free_bytes',8*full.GIB),('train_split','train'),('experiment',{})):
        result.setdefault(name,default)
    result['project_root']=str(root)
    if set(raw.get('train',{}))-set(full.TRAIN_DEFAULTS) or set(raw.get('budget',{}))-set(full.BUDGET_DEFAULTS):
        raise ValueError('Unknown train/budget setting')
    result['train']={**copy.deepcopy(full.TRAIN_DEFAULTS),**raw.get('train',{})}
    result['budget']={**copy.deepcopy(full.BUDGET_DEFAULTS),**raw.get('budget',{})}
    result['adapter']={**ADAPTER_DEFAULTS,**raw.get('adapter',{})}
    if result['fixture'] and 'retention_eval_batches' not in raw.get('train',{}):result['train']['retention_eval_batches']=0
    for name in ('base_checkpoint','data_manifest','output_dir'):
        result[name]=str(full.under(result[name],root))
    result['budget']['ledger_path']=str(full.under(result['budget']['ledger_path'],root))
    if result.get('evaluation_protocol'):
        result['evaluation_protocol']=str(full.under(result['evaluation_protocol'],root))
    validate(result);return result


def runtime_identity():
    root=Path(__file__).resolve().parents[2]
    files=list((root/'src/llm_lora').rglob('*.py'))+[root/'scripts'/name for name in
        ('train_lora.py','evaluate_lora.py','evaluate_sft_models.py','chat_sft.py')]
    packages={}
    for name in ('torch','peft','transformers','accelerate','safetensors'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    return {'full_sft_runtime_sha256':full.runtime_fingerprint(root),
            'lora_source_sha256':canonical_hash({str(path.relative_to(root)):sha256_file(path) for path in sorted(files) if path.is_file()}),
            'packages':packages}


def runtime_fingerprint():return canonical_hash(runtime_identity())
