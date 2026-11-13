"""只从匹配的原Base和Adapter重建LoRA模型，不替换训练数据身份。"""
from pathlib import Path
import torch

from llm_lab.common.io import read_json,sha256_file
from llm_lab.tokenization.bpe import load_tokenizer
from .adapters import inject,load_adapter
from .engine import load_base


def load_model(checkpoint,device='cuda'):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('schema_version')!=1 or payload.get('method')!='lora' or payload.get('checkpoint_kind') not in {'adapter_full','adapter_only'}:
        raise ValueError('Not a custom LoRA checkpoint')
    config=payload['config'];model,spec,base=load_base(config)
    if base['sha256']!=payload['identity']['base_checkpoint_sha256']:raise ValueError('Adapter belongs to different Base')
    manifest_path=Path(config['data_manifest'])
    if sha256_file(manifest_path)!=payload['identity']['train_data']['manifest_sha256']:raise ValueError('Adapter training data identity changed')
    manifest=read_json(manifest_path);token_path=manifest_path.parent/manifest['tokenizer']['path']
    if sha256_file(token_path)!=payload['identity']['tokenizer_sha256']:raise ValueError('Adapter tokenizer changed')
    model,adapter=inject(model,config['adapter'])
    if adapter!=payload['adapter']:raise ValueError('Adapter target/rank structure changed')
    load_adapter(model,payload['adapter_state']);model.to(device).eval()
    return model,load_tokenizer(token_path),{'step':payload['step'],'adapter':adapter,'checkpoint_sha256':sha256_file(checkpoint)}
