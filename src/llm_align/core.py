"""共享的数据、采样与回答区间打分。位置只平移一次，右侧 padding 不参与损失。"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import json
import random

import torch
from torch.nn import functional as F

from llm_lab.common.io import read_json, sha256_file
from llm_math_data.build import manifest_identity, prefix
from llm_math_data.verifier import reward, numeric_value


def precision(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()


class MathCorpus:
    """仅允许训练/开发分区；测试题必须由独立的最终评测入口读取。"""
    def __init__(self, manifest, split, view='rlvr'):
        if split not in {'train', 'train_2000', 'train_10000', 'validation'} or view not in {'rlvr', 'dpo'}:
            raise ValueError('Training entry only accepts training/development splits')
        self.path = Path(manifest)
        self.manifest = read_json(self.path)
        m = self.manifest
        if m['data_identity'] != manifest_identity(m) or m['status'] != 'published':
            raise ValueError('Math data identity changed or not published')
        acc = m['acceptance']
        report = self.path.parent / acc['report']
        if acc['status'] != 'passed' or sha256_file(report) != acc['sha256']:
            raise ValueError('Math acceptance receipt changed')
        receipt = read_json(report)
        if receipt['status'] != 'passed' or receipt['data_identity'] != m['data_identity']:
            raise ValueError('Math acceptance identity mismatch')
        spec = m['files'][f'{view}/{split}']
        source = self.path.parent / spec['path']
        if sha256_file(source) != spec['sha256']:
            raise ValueError('Math split checksum changed')
        self.rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line]
        if len(self.rows) != spec['examples']:
            raise ValueError('Math split record count mismatch')
        self.identity = {'manifest_sha256': sha256_file(self.path), 'data_identity': m['data_identity'],
                         'split_sha256': spec['sha256'], 'view': view, 'split': split, 'examples': len(self.rows)}
        self.tokenizer_path = self.path.parent / m['tokenizer']['path']
        if sha256_file(self.tokenizer_path) != m['tokenizer']['sha256']:
            raise ValueError('Math tokenizer changed')

    def prompts(self, tokenizer, rows):
        return [tokenizer.encode(prefix(row['prompt'], self.manifest['tokenization']),
                                 add_special_tokens=False).ids for row in rows]

    def answers(self, tokenizer, rows, field='ground_truth'):
        eos = self.manifest['tokenizer']['eos_id']
        return [tokenizer.encode(row[field], add_special_tokens=False).ids + [eos] for row in rows]


def development_subset(rows, per_stratum=8, seed=4201):
    """固定的语言×题型分层开发样本；绝不从测试集抽样。"""
    groups = {}
    for row in rows:
        groups.setdefault((row['language'], row['metadata']['family']), []).append(row)
    rng = random.Random(seed)
    result = []
    for key in sorted(groups):
        values = list(groups[key])
        rng.shuffle(values)
        result.extend(values[:per_stratum])
    return result


def pack(prompts, answers, device, pad=0):
    """[B,T] 输入及 [B,T] 回答 mask；目标 token 与模型 logits 严格错开一位。"""
    if not prompts or len(prompts) != len(answers) or any(not p or not a for p, a in zip(prompts, answers)):
        raise ValueError('Nonempty paired prompts/responses required')
    length = max(len(p) + len(a) - 1 for p, a in zip(prompts, answers))
    inputs = torch.full((len(prompts), length), pad, dtype=torch.long, device=device)
    targets = torch.zeros_like(inputs)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for i, (prompt_ids, answer_ids) in enumerate(zip(prompts, answers)):
        ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long, device=device)
        n = len(ids) - 1
        inputs[i, :n], targets[i, :n] = ids[:-1], ids[1:]
        mask[i, len(prompt_ids) - 1:n] = True
    return inputs, targets, mask


def token_scores(model, prompts, answers, device, value_head=None):
    inputs, targets, mask = pack(prompts, answers, device)
    hidden = []
    # 最终归一化隐状态 [B,T,C] 同时提供给独立的 Critic 线性头。
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: hidden.append(args[0])) if value_head is not None else None
    try:
        with precision(device):
            logits = model(inputs).logits
            # Value 梯度只更新 Critic 头；策略梯度更新共享的 SFT LoRA。
            values = value_head(hidden[0].detach()).squeeze(-1).float() if value_head is not None else None
        logp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    finally:
        if handle is not None:
            handle.remove()
    return logp, mask, values


def sequence_scores(model, prompts, answers, device):
    logp, mask, _ = token_scores(model, prompts, answers, device)
    return (logp * mask).sum(-1)


@torch.no_grad()
def generate(model, tokenizer, prompts, device, max_new_tokens=16, sample=True, eos=2):
    """完整前缀重算：按真实长度取 logits，右填充不会进入任何真实 token 的因果上下文。

    温度固定 1，不做 top-k/top-p；采样分布与 PPO/GRPO 的 log-prob 分布一致。
    """
    model.eval()
    ids = [list(prompt_ids) for prompt_ids in prompts]
    answers = [[] for _ in prompts]
    done = [False for _ in prompts]
    for _ in range(max_new_tokens):
        active = [i for i in range(len(prompts)) if not done[i]]
        if not active:
            break
        length = max(len(ids[i]) for i in active)
        if length > model.config.max_seq_len:
            raise ValueError('Rollout exceeds model context')
        x = torch.zeros((len(active), length), dtype=torch.long, device=device)
        lengths = []
        for j, i in enumerate(active):
            x[j, :len(ids[i])] = torch.tensor(ids[i], device=device)
            lengths.append(len(ids[i]) - 1)
        with precision(device):
            logits = model(x).logits
        scores = logits[torch.arange(len(active), device=device), torch.tensor(lengths, device=device)].float()
        next_ids = torch.multinomial(scores.softmax(-1), 1).flatten() if sample else scores.argmax(-1)
        for j, i in enumerate(active):
            token = int(next_ids[j])
            ids[i].append(token)
            answers[i].append(token)
            done[i] = token == eos
    texts = [tokenizer.decode(answer[:-1] if answer and answer[-1] == eos else answer,
                              skip_special_tokens=False) for answer in answers]
    return {'answers': answers, 'texts': texts, 'eos': done}


def score_outputs(rows, generated):
    rewards = [reward(text, row['ground_truth'], row.get('reward_type', 'numeric_exact'))
               for row, text in zip(rows, generated['texts'])]
    return {'correct': int(sum(rewards)), 'examples': len(rows), 'accuracy': sum(rewards) / len(rows),
            'numeric_format_rate': sum(numeric_value(text) is not None for text in generated['texts']) / len(rows),
            'eos_rate': sum(generated['eos']) / len(rows), 'rewards': rewards}
