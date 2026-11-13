"""从零实现 Dense Decoder：RoPE + GQA + RMSNorm + SwiGLU。

数据流：[B,T] → [B,T,C] → [B,H,T,D] → [B,T,V] → 标量交叉熵。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from llm_lab.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # 使用 FP32 计算均方根，避免低精度平方与求和溢出。
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def apply_rope(q: Tensor, k: Tensor, theta: float) -> tuple[Tensor, Tensor]:
    # q/k 的头数可以不同，但头维 D 相同；序列位置从 0 到 T-1。
    length, dim = q.shape[-2:]
    frequencies = theta ** (-torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim)
    angles = torch.outer(torch.arange(length, device=q.device, dtype=torch.float32), frequencies)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]

    def rotate(x: Tensor) -> Tensor:
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)

    return rotate(q), rotate(k)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads, self.kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.rope_theta, self.dropout = config.rope_theta, config.dropout
        self.q = nn.Linear(config.dim, self.heads * self.head_dim, bias=False)
        self.k = nn.Linear(config.dim, self.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.dim, self.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, channels = x.shape
        q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.rope_theta)
        # 显式扩展 KV 头兼容 CPU 数值测试；SDPA 在合适 GPU 上选择高效内核。
        repeats = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                 dropout_p=self.dropout if self.training else 0.0)
        return self.out(attended.transpose(1, 2).contiguous().view(batch, length, channels))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(config.dim), RMSNorm(config.dim)
        self.attn = Attention(config)
        self.gate = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        normalized = self.ffn_norm(x)
        # SwiGLU(x) = SiLU(W_gate x) ⊙ W_up x，然后映射回残差维度 C。
        return x + self.dropout(self.down(F.silu(self.gate(normalized)) * self.up(normalized)))


@dataclass
class LMOutput:
    logits: Tensor
    loss: Tensor | None


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        for block in self.blocks:
            # 残差分支按层数缩放初始化，控制深层残差方差。
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> LMOutput:
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("input_ids must have shape [B,T] within configured context")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have shape [B,T] and be shifted by the data pipeline")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True) if self.config.activation_checkpointing and self.training else block(x)
        logits = self.lm_head(self.norm(x))
        # 此处不再 shift：Dataset 已产生 x=t[i:i+T], y=t[i+1:i+T+1]。
        loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1), ignore_index=-100) if labels is not None else None
        return LMOutput(logits=logits, loss=loss)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

