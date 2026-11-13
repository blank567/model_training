"""PEFT低层注入、严格Adapter状态协议与冻结Base检查。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path

import torch
from torch import nn

ALL_SUFFIXES=('attn.q','attn.k','attn.v','attn.out','gate','up','down')


class ConfigView:
    """保留Decoder的属性读取，同时提供PEFT所需的dict接口。"""
    def __init__(self,original):self.original=original
    def to_dict(self):
        return {**asdict(self.original),'model_type':'llm_lab_decoder','tie_word_embeddings':self.original.tie_embeddings}
    def __getattr__(self,name):
        if name=='model_type':return 'llm_lab_decoder'
        if name=='tie_word_embeddings':return self.original.tie_embeddings
        return getattr(self.original,name)
    def get(self,key,default=None):return self.to_dict().get(key,default)
    def __getitem__(self,key):return self.to_dict()[key]


def is_adapter(name):return '.lora_A.' in name or '.lora_B.' in name


def inject(model,settings):
    if metadata.version('peft')!='0.17.1':raise RuntimeError('Custom LoRA is pinned to peft==0.17.1')
    from peft import LoraConfig,inject_adapter_in_model
    if hasattr(model,'peft_config'):raise ValueError('Refusing to inject a second adapter')
    suffixes=ALL_SUFFIXES if settings['target_scope']=='all' else ('attn.q','attn.v')
    targets=[name for name,module in model.named_modules() if isinstance(module,nn.Linear)
             and any(name.endswith('.'+suffix) for suffix in suffixes)]
    expected=len(model.blocks)*len(suffixes)
    if len(targets)!=expected:raise ValueError('LoRA target coverage differs from Decoder structure')
    rank=settings['rank'];expected_parameters=sum(rank*(model.get_submodule(name).in_features+model.get_submodule(name).out_features) for name in targets)
    model.requires_grad_(False);model.config=ConfigView(model.config)
    peft=LoraConfig(r=rank,lora_alpha=settings['alpha'],lora_dropout=settings['dropout'],bias='none',
                    target_modules=targets,init_lora_weights=True)
    model=inject_adapter_in_model(peft,model)
    trainable=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable!=expected_parameters:raise ValueError('Unexpected trainable parameters after injection')
    assert_frozen(model)
    return model,{'targets':targets,'target_modules':expected,'trainable_parameters':trainable,
                  'base_parameters':sum(p.numel() for n,p in model.named_parameters() if not is_adapter(n))}


def assert_frozen(model):
    for name,parameter in model.named_parameters():
        if parameter.requires_grad!=is_adapter(name):raise ValueError('Trainable mask escaped Adapter: '+name)
        if not is_adapter(name) and parameter.grad is not None:raise ValueError('Frozen Base accumulated gradients')


def frozen_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.named_parameters()):
        if is_adapter(name):continue
        key=name.replace('.base_layer','')
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adapter_state(model):
    from peft import get_peft_model_state_dict
    result=get_peft_model_state_dict(model,save_embedding_layers=False)
    if not result or any(not is_adapter(name) for name in result):raise ValueError('Export includes non-Adapter tensors')
    return {name:value.detach().cpu().contiguous().clone() for name,value in result.items()}


def load_adapter(model,state):
    from peft import set_peft_model_state_dict
    expected=adapter_state(model)
    if expected.keys()!=state.keys():raise ValueError('Adapter tensor keys differ')
    for key,value in state.items():
        if value.shape!=expected[key].shape or not torch.isfinite(value).all():raise ValueError('Adapter shape or finiteness mismatch')
    outcome=set_peft_model_state_dict(model,state)
    if outcome.unexpected_keys:raise ValueError('Unexpected Adapter keys')
    actual=adapter_state(model)
    if any(not torch.equal(actual[key],state[key].cpu()) for key in state):raise ValueError('Adapter load did not restore exact tensors')
    assert_frozen(model)


def export_best(directory,model,metadata_payload):
    from safetensors.torch import save_file
    directory=Path(directory);state=adapter_state(model)
    temporary=directory/'adapter_model.safetensors.tmp'
    save_file(state,str(temporary));temporary.replace(directory/'adapter_model.safetensors')
    from llm_lab.common.io import write_json
    write_json(directory/'adapter_config.json',metadata_payload)


def merge_for_inference(model):
    """在调用者自己的模型实例合并；不改磁盘Base或原Adapter。"""
    from peft.tuners.lora.layer import LoraLayer
    for module in model.modules():
        if isinstance(module,LoraLayer):module.merge(safe_merge=True)
    return model
