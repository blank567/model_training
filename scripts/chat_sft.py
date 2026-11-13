"""自研 Base/SFT 的单轮回答入口；使用同一数据模板做前后对照。

只读取本项目生成且可信的 Checkpoint。生成仅输出 assistant 回答，
不加载 HF 远程代码、不修改训练源码、分词器或模型权重。
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from generate_text import sampling_scores, validate_settings
from llm_lab.common.io import canonical_hash, read_json, sha256_file
from llm_lab.config import ModelConfig
from llm_lab.engine.checkpoint import load_checkpoint
from llm_lab.models import Decoder
from llm_lab.tokenization.bpe import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n"
EMPTY_PREFIX = "User:\n{prompt}\n\nAssistant:\n"
IDENTITY_FIELDS = ("schema_version", "kind", "splits", "tokenizer", "tokenization", "chinese_normalization")


def _config_path(value: str, config: dict) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(config.get("project_root", ROOT)) / path).resolve()


def _token_file(manifest_path: Path, manifest: dict) -> Path:
    relative = manifest["tokenizer"]["path"]
    path = manifest_path.parent / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
        raise ValueError("Tokenizer must be a regular file inside its data directory")
    path = path.resolve()
    if not path.is_relative_to(manifest_path.parent) or not path.is_file():
        raise ValueError("Tokenizer path escapes the data directory or is missing")
    if sha256_file(path) != manifest["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer SHA256 mismatch")
    return path


def _validate_template(settings: dict) -> None:
    if (settings.get("prefix_template") != PREFIX
            or settings.get("empty_system_prefix_template") != EMPTY_PREFIX
            or settings.get("encoding") != "prefix_and_response_separately"):
        raise ValueError("Unsupported or changed SFT chat template")


def _sft_manifest(path: Path) -> tuple[dict, Path]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("SFT manifest must be a regular existing file")
    if (path.parent / "SUPERSEDED.json").exists():
        raise ValueError("This SFT dataset is superseded; select the published data version")
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "sft_tokenized_conversations":
        raise ValueError("A published custom SFT dataset manifest is required")
    try:
        actual = canonical_hash({key: manifest[key] for key in IDENTITY_FIELDS})
    except KeyError as error:
        raise ValueError("Incomplete SFT manifest identity") from error
    if actual != manifest.get("data_identity"):
        raise ValueError("SFT data_identity does not match its manifest")
    _validate_template(manifest["tokenization"])
    if manifest.get("acceptance", {}).get("status") != "passed":
        raise ValueError("SFT data has not passed acceptance")
    return manifest, _token_file(path, manifest)


def load_model(checkpoint, device="cpu", *, data_manifest=None, threads=4):
    """同时读取旧 Base 与新 SFT 格式；分词器/模板身份先核验再分配模型。"""
    if device not in {"cpu", "cuda"} or type(threads) is not int or threads < 1:
        raise ValueError("device must be cpu/cuda and threads must be positive")
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("Checkpoint must be a regular existing trusted project file")
    checkpoint = checkpoint.resolve()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    before = checkpoint.stat()
    payload = load_checkpoint(checkpoint)
    checkpoint_digest = sha256_file(checkpoint)
    after = checkpoint.stat()
    if any(getattr(before, key) != getattr(after, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")):
        raise ValueError("Checkpoint changed while loading; choose a stable saved checkpoint")
    config, identity = payload.get("config", {}), payload.get("identity", {})
    is_sft = payload.get("stage") == "sft"
    if is_sft:
        if payload.get("checkpoint_kind") not in {"model_only", "full"}:
            raise ValueError("Unsupported SFT checkpoint kind")
        manifest_path = _config_path(config["data_manifest"], config)
        if data_manifest is not None and Path(data_manifest).resolve() != manifest_path:
            raise ValueError("SFT generation cannot replace its checkpoint-bound data manifest")
        if sha256_file(manifest_path) != identity.get("manifest_sha256"):
            raise ValueError("SFT manifest differs from checkpoint identity")
        manifest, token_path = _sft_manifest(manifest_path)
        if manifest["data_identity"] != identity.get("data_identity"):
            raise ValueError("SFT data identity differs from checkpoint")
        model_config = ModelConfig(**payload["model_config"])
    else:
        # Base 对照必须显式选择相同 SFT 模板，不能把普通续写当成指令回答。
        if data_manifest is None:
            raise ValueError("Base comparison requires --data-manifest for the shared SFT template")
        original_manifest = _config_path(config["data_manifest"], config)
        if sha256_file(original_manifest) != identity.get("manifest_sha256"):
            raise ValueError("Base pretraining manifest differs from checkpoint identity")
        original = read_json(original_manifest)
        _token_file(original_manifest, original)
        if original["tokenizer"]["sha256"] != identity.get("tokenizer_sha256"):
            raise ValueError("Base tokenizer identity mismatch")
        manifest_path = Path(data_manifest).resolve()
        manifest, token_path = _sft_manifest(manifest_path)
        model_config = ModelConfig(**config["model"])
    spec = manifest["tokenizer"]
    if spec["sha256"] != identity.get("tokenizer_sha256"):
        raise ValueError("SFT and checkpoint tokenizer identities differ")
    if manifest.get("fixture") is True and device != "cpu":
        raise ValueError("Fixture generation is limited to CPU correctness checks")
    tokenizer = load_tokenizer(token_path)
    if tokenizer.get_vocab_size() != model_config.vocab_size or spec["vocab_size"] != model_config.vocab_size:
        raise ValueError("Model/tokenizer vocabulary mismatch")
    for token, key, required in (("<|pad|>", "pad_id", 0), ("<|unk|>", "unk_id", 1), ("<|eos|>", "eos_id", 2)):
        if spec.get(key) != required or tokenizer.token_to_id(token) != required:
            raise ValueError("Tokenizer special-token identity mismatch")
    torch.set_num_threads(threads)
    # [B,T] -> [B,T,V]。初始化只用于构造结构，随后严格载入训练得到的参数。
    with torch.random.fork_rng(devices=[]):
        model = Decoder(model_config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_digest,
                "checkpoint_step": payload.get("step"), "checkpoint_kind": payload.get("checkpoint_kind", "pretrain_full"),
                "stage": "sft" if is_sft else "base_comparison", "model": asdict(model_config),
                "parameters": model.parameter_count(), "device": device,
                "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                "data_identity": manifest["data_identity"], "tokenizer_sha256": spec["sha256"],
                "tokenization": manifest["tokenization"], "base_checkpoint_sha256": identity.get("base_checkpoint_sha256"),
                "decode_strategy": "full_prefix_recompute_no_kv_cache"}
    return model, tokenizer, metadata


@torch.inference_mode()
def generate(model, tokenizer, prompt, *, tokenization, system="", max_new_tokens=128,
             temperature=.8, top_p=.9, top_k=50, repetition_penalty=1.1, greedy=False, seed=42):
    """单轮 assistant 回答；不静默裁剪 prefix，也不重置 RoPE 的上下文位置。"""
    _validate_template(tokenization)
    validate_settings(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                      repetition_penalty=repetition_penalty, no_repeat_ngram_size=0, seed=seed, greedy=greedy)
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(system, str):
        raise ValueError("prompt must be nonempty text and system must be text")
    template = tokenization["prefix_template"] if system else tokenization["empty_system_prefix_template"]
    prefix = template.format(system=system, prompt=prompt)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    special = {tokenizer.token_to_id(token) for token in ("<|pad|>", "<|unk|>", "<|eos|>")}
    if None in special or len(special) != 3:
        raise ValueError("PAD/UNK/EOS special-token identities are invalid")
    if not prefix_ids or special.intersection(prefix_ids):
        raise ValueError("Prompt prefix cannot contain embedded special tokens")
    if len(prefix_ids) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("Prompt plus generation budget exceeds model context; shorten the prompt or max-new-tokens")
    eos = tokenizer.token_to_id("<|eos|>")
    forbidden = special - {eos}
    history, generated = list(prefix_ids), []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    stop_reason = "max_new_tokens"
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            # 没有跨样本 packing 或缓存；单例完整 prefix [1,T]，读取末位置 [V] 分布。
            x = torch.tensor([history], dtype=torch.long, device=device)
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else contextlib.nullcontext()
            with precision:
                logits = model(x).logits[0, -1]
            scores = sampling_scores(logits, history, temperature=temperature, top_p=top_p, top_k=top_k,
                                     repetition_penalty=repetition_penalty, forbidden_ids=forbidden,
                                     eos_id=eos, greedy=greedy)
            if not torch.isfinite(scores).any():
                stop_reason = "constraints_exhausted"
                break
            identifier = int(scores.argmax()) if greedy else int(torch.multinomial(scores.cpu().softmax(-1), 1, generator=generator))
            generated.append(identifier);history.append(identifier)
            if identifier == eos:
                stop_reason = "eos"
                break
    finally:
        model.train(was_training)
    visible = generated[:-1] if generated and generated[-1] == eos else generated
    answer = tokenizer.decode(visible, skip_special_tokens=True)
    return {"prompt": prompt, "system": system, "prefix": prefix, "input_ids": prefix_ids,
            "answer": answer, "completion": answer, "generated_ids": generated,
            "generated_token_count": len(generated), "stop_reason": stop_reason,
            "settings": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p,
                         "top_k": top_k, "repetition_penalty": repetition_penalty, "greedy": greedy, "seed": seed}}


def _write_report(path: Path, result: dict) -> None:
    original = path
    path = path.resolve()
    if original.is_symlink() or not path.is_relative_to(ROOT) or path.exists():
        raise ValueError("Report must be a new file inside the project directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("Refusing to replace an existing report temporary file")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError("Generation report appeared while writing")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-manifest", help="Required for Base comparison; SFT keeps its own bound manifest")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output")
    args = vars(parser.parse_args(argv))
    checkpoint, device, data_manifest, threads, output = (args.pop(key) for key in ("checkpoint", "device", "data_manifest", "threads", "output"))
    try:
        model, tokenizer, metadata = load_model(checkpoint, device, data_manifest=data_manifest, threads=threads)
        result = generate(model, tokenizer, tokenization=metadata["tokenization"], **args)
        if output:
            _write_report(Path(output), {"metadata": metadata, **result})
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"错误：{error}\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
