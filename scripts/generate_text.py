"""独立文本续写：采样可缓解重复，不增加模型知识；不修改训练代码或权重。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from llm_lab.common.io import read_json, sha256_file, write_json
from llm_lab.config import ModelConfig
from llm_lab.engine.checkpoint import load_checkpoint
from llm_lab.models import Decoder
from llm_lab.tokenization.bpe import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def validate_settings(*, max_new_tokens, temperature, top_p, top_k,
                      repetition_penalty, no_repeat_ngram_size, seed, greedy):
    for name, value in (("max_new_tokens", max_new_tokens), ("top_k", top_k),
                        ("no_repeat_ngram_size", no_repeat_ngram_size), ("seed", seed)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name, value in (("temperature", temperature), ("top_p", top_p),
                        ("repetition_penalty", repetition_penalty)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if temperature <= 0 or not 0 < top_p <= 1 or repetition_penalty < 1:
        raise ValueError("temperature>0, 0<top_p<=1 and repetition_penalty>=1 are required")
    if seed >= 2**64:
        raise ValueError("seed must be smaller than 2**64")
    if type(greedy) is not bool:
        raise ValueError("greedy must be boolean")


def forbidden_ngram_tokens(history, size):
    if not size or len(history) < size - 1:
        return set()
    prefix = tuple(history[-(size - 1):]) if size > 1 else ()
    return {history[index + size - 1] for index in range(len(history) - size + 1)
            if tuple(history[index:index + size - 1]) == prefix}


def sampling_scores(logits, history, *, temperature=0.8, top_p=0.9, top_k=50,
                    repetition_penalty=1.1, no_repeat_ngram_size=0,
                    forbidden_ids=(), eos_id=None, greedy=False):
    """返回 1D FP32 分数；-inf 为明确屏蔽，空候选由调用方停止。"""
    scores = logits.detach().float().clone()
    if scores.ndim != 1 or torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError("Expected finite-or-negative-infinity 1D next-token logits")
    if history and repetition_penalty != 1:
        seen = torch.tensor(sorted(set(history)), device=scores.device, dtype=torch.long)
        if int(seen.min()) < 0 or int(seen.max()) >= scores.numel():
            raise ValueError("History contains out-of-vocabulary token IDs")
        values = scores[seen]
        scores[seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    blocked = set(forbidden_ids) | forbidden_ngram_tokens(history, no_repeat_ngram_size)
    # EOS 始终可用于结束，不让重复限制把它误禁掉。
    blocked.discard(eos_id)
    for identifier in blocked:
        if type(identifier) is not int or not 0 <= identifier < scores.numel():
            raise ValueError("Invalid forbidden token ID")
        scores[identifier] = -torch.inf
    if greedy or not torch.isfinite(scores).any():
        return scores
    # 先平移最大分数，极低温度也不产生正无穷；softmax 分布不变。
    scores = (scores - scores.max()) / temperature
    if top_k:
        keep = min(top_k, scores.numel())
        threshold = torch.topk(scores, keep).values[-1]
        scores[scores < threshold] = -torch.inf
    if top_p < 1:
        ordered, indices = scores.sort(descending=True)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        # 保留概率累计首次越过阈值的 token，并保证最高分 token 不被移除。
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[indices[remove]] = -torch.inf
    return scores


def load_model(checkpoint, device="cpu", threads=4):
    if device not in {"cpu", "cuda"} or type(threads) is not int or threads < 1:
        raise ValueError("device must be cpu/cuda and threads must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(threads)
    checkpoint = Path(checkpoint).resolve()
    payload = load_checkpoint(checkpoint)
    raw, identity = payload["config"], payload["identity"]
    manifest_path = Path(raw["data_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = Path(raw.get("project_root", Path.cwd())) / manifest_path
    manifest_path = manifest_path.resolve()
    if sha256_file(manifest_path) != identity["manifest_sha256"]:
        raise ValueError("Manifest differs from checkpoint identity")
    manifest = read_json(manifest_path)
    spec = manifest["tokenizer"]
    token_path = manifest_path.parent / spec["path"]
    if token_path.is_symlink() or not token_path.resolve().is_relative_to(manifest_path.parent):
        raise ValueError("Tokenizer must stay within the corpus directory")
    if spec["sha256"] != identity["tokenizer_sha256"] or sha256_file(token_path) != spec["sha256"]:
        raise ValueError("Tokenizer differs from checkpoint identity")
    tokenizer = load_tokenizer(token_path)
    config = ModelConfig(**raw["model"])
    if tokenizer.get_vocab_size() != config.vocab_size or spec["vocab_size"] != config.vocab_size:
        raise ValueError("Tokenizer/model vocabularies differ")
    eos = tokenizer.token_to_id("<|eos|>")
    if eos != spec["eos_id"] or type(eos) is not int or not 0 <= eos < config.vocab_size:
        raise ValueError("Invalid EOS identity")
    # 初始化后立即加载已训练权重；保存/恢复 CPU RNG，避免污染调用方随机状态。
    with torch.random.fork_rng(devices=[]):
        model = Decoder(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    meta = {"checkpoint": str(checkpoint), "checkpoint_step": payload.get("step"),
            "training_tokens": payload.get("tokens_seen"), "manifest": str(manifest_path),
            "manifest_sha256": identity["manifest_sha256"], "tokenizer_sha256": spec["sha256"],
            "training_source_sha256": identity.get("source"), "model": asdict(config),
            "device": device, "threads": threads, "parameters": model.parameter_count(),
            "eos_id": eos, "pad_id": tokenizer.token_to_id("<|pad|>"),
            "unk_id": tokenizer.token_to_id("<|unk|>")}
    return model, tokenizer, meta


@torch.inference_mode()
def generate(model, tokenizer, prompt, *, max_new_tokens=100, temperature=0.8,
             top_p=0.9, top_k=50, repetition_penalty=1.1,
             no_repeat_ngram_size=0, seed=42, greedy=False):
    settings = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                    top_k=top_k, repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size, seed=seed, greedy=greedy)
    validate_settings(**settings)
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    eos = tokenizer.token_to_id("<|eos|>")
    if type(eos) is not int:
        raise ValueError("Tokenizer needs an EOS token")
    forbidden = {value for value in (tokenizer.token_to_id("<|pad|>"), tokenizer.token_to_id("<|unk|>"))
                 if value is not None and value != eos}
    original_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    history = list(original_ids or [eos])
    new_ids, stop_reason = [], "max_new_tokens"
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()
    for _ in range(max_new_tokens):
        context = history[-model.config.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x).logits[0, -1]
        scores = sampling_scores(logits, history, forbidden_ids=forbidden, eos_id=eos,
                                 temperature=temperature, top_p=top_p, top_k=top_k,
                                 repetition_penalty=repetition_penalty,
                                 no_repeat_ngram_size=no_repeat_ngram_size, greedy=greedy)
        if not torch.isfinite(scores).any():
            stop_reason = "constraints_exhausted"
            break
        identifier = int(scores.argmax()) if greedy else int(torch.multinomial(scores.cpu().softmax(-1), 1, generator=generator))
        history.append(identifier)
        new_ids.append(identifier)
        if identifier == eos:
            stop_reason = "eos"
            break
    visible = new_ids[:-1] if new_ids and new_ids[-1] == eos else new_ids
    # Byte BPE 在前缀边界可能拆分 UTF-8 字节，完整输出按整段统一解码。
    return {"prompt": prompt, "full_text": tokenizer.decode(original_ids + visible, skip_special_tokens=True),
            "completion": tokenizer.decode(visible, skip_special_tokens=True),
            "input_ids": original_ids, "generated_ids": new_ids, "stop_reason": stop_reason,
            "generated_token_count": len(new_ids), "settings": settings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output")
    args = vars(parser.parse_args(argv))
    checkpoint, device, threads, output = (args.pop(key) for key in ("checkpoint", "device", "threads", "output"))
    prompt = args.pop("prompt")
    validate_settings(**args)
    if output:
        target = Path(output).resolve()
        if not target.is_relative_to(ROOT) or target.is_symlink():
            raise ValueError("Output must stay within the project directory")
        if target.exists():
            raise FileExistsError("Refusing to overwrite an existing generation report")
    model, tokenizer, meta = load_model(checkpoint, device, threads)
    result = generate(model, tokenizer, prompt, **args)
    if output:
        write_json(target, {"metadata": meta, **result})
    print(result["full_text"])


if __name__ == "__main__":
    main()
