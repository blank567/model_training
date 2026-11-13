"""读取独立RL checkpoint；不把RL格式伪装成历史SFT格式。"""
from pathlib import Path
import torch
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_lab.common.io import sha256_file
from llm_lab.tokenization.bpe import load_tokenizer


def load_model(path,device='cpu',tokenizer_path=None):
    path=Path(path);payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('stage')!='reading_rl' or payload.get('checkpoint_kind') not in {'full','model_only'}:
        raise ValueError('A trusted reading-RL checkpoint is required')
    file=Path(tokenizer_path or payload['tokenizer']['path'])
    if sha256_file(file)!=payload['tokenizer']['sha256']:raise ValueError('Checkpoint-bound tokenizer changed')
    model=Decoder(ModelConfig(**payload['model_config']));model.load_state_dict(payload['model']);model.to(device).eval()
    return model,load_tokenizer(file),{'checkpoint_sha256':sha256_file(path),'algorithm':payload['algorithm'],
      'optimizer_step':payload['optimizer_step'],'data_batch_cursor':payload['data_batch_cursor']}
