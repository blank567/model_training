"""单机单卡 SFT/LoRA/DPO 实验入口。

配置校验只有标准库依赖；真正训练才导入 Hugging Face，并要求远程 CUDA。
预训练是学习语言建模，SFT 是训练目标，LoRA 是参数更新方法，二者不互斥。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import unicodedata
from typing import Any

from llm_lab.common.tracking import WANDB_DEFAULTS, validate_tracking


PINNED_PACKAGES = {
    "transformers": "4.57.1", "peft": "0.17.1", "trl": "0.24.0",
    "accelerate": "1.11.0", "datasets": "4.3.0",
}
TEMPLATE_VERSION = "plain_roles_v1"
TEMPLATE_TEXT = "User:\n{prompt}\nAssistant:\n"
DEFAULTS = {
    "schema_version": 1, "project_root": "../..", "seed": 42,
    "base_model": "Qwen/Qwen2.5-0.5B", "base_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "cache_dir": "artifacts/cache/huggingface", "precision": "bf16",
    "max_length": 1024, "batch_size": 2, "gradient_accumulation_steps": 8,
    "train_token_budget": 200000, "learning_rate": 2e-4,
    "warmup_ratio": 0.05, "weight_decay": 0.01, "max_grad_norm": 1.0,
    "gradient_checkpointing": True, "eval_steps": 25, "save_steps": 25,
    "logging_steps": 5, "save_total_limit": 2, "beta": 0.1,
    "tensorboard": False, "deterministic": True,
    **WANDB_DEFAULTS,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    "budget": {"hourly_rate_cny": None, "run_limit_cny": 80.0, "project_limit_cny": 500.0,
               "reserve_cny": 100.0, "max_runtime_seconds": 7200},
    "budget_ledger": "runs/budget_ledger.json", "sft_checkpoint": None,
}
REQUIRED = {"stage", "method", "train_file", "validation_file", "output_dir"}


def read_json(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是对象: {path}")
    return value


def write_json(path: Path | str, payload: Any) -> None:
    """同目录原子替换，避免中途退出留下半份 manifest。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path | str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: 非法 JSONL") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{number}: 每行必须是对象")
            rows.append(obj)
    if not rows:
        raise ValueError(f"JSONL 不能为空: {path}")
    return rows


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_fingerprint() -> str:
    """校验实际源码内容，不仅依赖 Git HEAD；未提交的代码改动也会拒绝续跑。"""
    package_root = Path(__file__).resolve().parents[1]
    files = sorted(package_root.rglob("*.py"))
    return canonical_hash({path.relative_to(package_root).as_posix(): sha256_file(path) for path in files})


def normalized_prompt_hash(prompt: str) -> str:
    # 只用于发现数据重叠，不用这个版本替换真实训练输入。
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def format_prompt(prompt: str) -> str:
    return TEMPLATE_TEXT.format(prompt=prompt)


def read_task_rows(path: Path, stage: str) -> list[dict]:
    from llm_lab.evaluation.task_schema import parse_task_json
    fields = {"source_id", "language", "prompt"} | ({"completion"} if stage == "sft" else {"chosen", "rejected"})
    rows = read_jsonl(path)
    seen = set()
    for index, row in enumerate(rows, 1):
        if set(row) != fields:
            raise ValueError(f"{path}:{index}: 字段必须为 {sorted(fields)}")
        if any(not isinstance(row[name], str) or not row[name].strip() for name in fields):
            raise ValueError(f"{path}:{index}: 所有字段必须为非空字符串")
        if row["language"] not in ("zh", "en"):
            raise ValueError(f"{path}:{index}: language 只能为 zh/en")
        key = (row["source_id"], row["language"])
        if key in seen:
            raise ValueError(f"{path}:{index}: 重复 source_id/language: {key}")
        seen.add(key)
        parse_task_json(row["completion"] if stage == "sft" else row["chosen"])
        if stage == "dpo" and row["chosen"].strip() == row["rejected"].strip():
            raise ValueError(f"{path}:{index}: chosen/rejected 不能相同")
    return rows


def check_split_leakage(train: list[dict], validation: list[dict]) -> None:
    # source_id 不带 language：同一原始问题的翻译必须在同一个 split。
    overlap = {r["source_id"] for r in train} & {r["source_id"] for r in validation}
    prompts = {normalized_prompt_hash(r["prompt"]) for r in train} & {normalized_prompt_hash(r["prompt"]) for r in validation}
    if overlap or prompts:
        raise ValueError(f"训练/验证泄漏: {len(overlap)} 个 source_id，{len(prompts)} 个归一化 prompt")


def _positive_int(value: Any, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} 必须为正整数")


def _number(value: Any, field: str, minimum: float = 0.0, strict: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} 必须是有限数值")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{field} 数值越界")


def _resolve_path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("文件路径必须是非空字符串")
    path = Path(value).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if os.name == "nt" and path.drive.upper() != "E:":
        raise ValueError(f"本地项目数据必须在 E: 盘: {path}")
    return path


def _artifact_weights(path: Path, checkpoint_type: str) -> None:
    names = ("adapter_model.safetensors", "adapter_model.bin") if checkpoint_type == "adapter" else (
        "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
    if not any((path / name).is_file() for name in names):
        raise ValueError(f"缺少 {checkpoint_type} 模型权重: {path}")
    if checkpoint_type == "adapter" and not (path / "adapter_config.json").is_file():
        raise ValueError(f"adapter 缺少 adapter_config.json: {path}")
    if checkpoint_type == "full" and not (path / "config.json").is_file():
        raise ValueError(f"完整模型缺少 config.json: {path}")


def read_artifact_manifest(path: Path) -> dict:
    if not path.is_dir():
        raise ValueError(f"模型目录不存在: {path}")
    manifest = read_json(path / "run_manifest.json")
    if manifest.get("workflow") != "posttrain" or manifest.get("schema_version") != 1:
        raise ValueError("模型必须包含本项目的后训练 manifest")
    if manifest.get("checkpoint_type") not in ("adapter", "full"):
        raise ValueError("未知 checkpoint_type，禁止把 adapter 当作完整模型加载")
    _artifact_weights(path, manifest["checkpoint_type"])
    return manifest


@dataclass
class ValidatedRun:
    config: dict
    train_rows: list[dict]
    validation_rows: list[dict]
    identity: dict
    resume_path: Path | None
    source_manifest: dict | None = None


def validate_config(config_path: Path | str, resume: str | None = None) -> ValidatedRun:
    """检查配置、原始数据和恢复身份；不导入 torch/transformers/peft/trl。"""
    config_path = Path(config_path).resolve()
    raw = read_json(config_path)
    if REQUIRED - raw.keys():
        raise ValueError(f"缺少配置: {sorted(REQUIRED - raw.keys())}")
    if raw.keys() - (DEFAULTS.keys() | REQUIRED):
        raise ValueError(f"未知配置项: {sorted(raw.keys() - (DEFAULTS.keys() | REQUIRED))}")
    cfg = {**DEFAULTS, **raw}
    validate_tracking(cfg)
    # 复制嵌套配置，避免不同运行共享可变 DEFAULTS。
    for key in ("lora", "budget"):
        if not isinstance(cfg[key], dict) or cfg[key].keys() - DEFAULTS[key].keys():
            raise ValueError(f"{key} 配置字段不合法")
        cfg[key] = {**DEFAULTS[key], **cfg[key]}
    if cfg["schema_version"] != 1 or cfg["stage"] not in ("sft", "dpo") or cfg["method"] not in ("full", "lora"):
        raise ValueError("schema_version=1，stage=sft/dpo，method=full/lora")
    if not isinstance(cfg["base_model"], str) or not cfg["base_model"].strip() or "instruct" in cfg["base_model"].casefold():
        raise ValueError("本实验必须指定 true Base 模型，不能使用 Instruct checkpoint")
    if not isinstance(cfg["base_revision"], str) or not re.fullmatch(r"[a-fA-F0-9]{40}", cfg["base_revision"]):
        raise ValueError("base_revision 必须固定到完整 40 位 commit SHA，不能使用 main 或短 SHA")
    if cfg["precision"] not in ("fp32", "bf16", "fp16"):
        raise ValueError("precision 只能为 fp32/bf16/fp16")
    for key in ("max_length", "batch_size", "gradient_accumulation_steps", "train_token_budget", "eval_steps", "save_steps", "logging_steps", "save_total_limit"):
        _positive_int(cfg[key], key)
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    for key in ("gradient_checkpointing", "tensorboard", "deterministic"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} 必须为布尔值")
    for key in ("learning_rate", "max_grad_norm", "beta"):
        _number(cfg[key], key, strict=True)
    for key in ("warmup_ratio", "weight_decay"):
        _number(cfg[key], key)
    if cfg["warmup_ratio"] > 1:
        raise ValueError("warmup_ratio 必须在 [0,1] 内")
    _positive_int(cfg["lora"]["r"], "lora.r")
    _positive_int(cfg["lora"]["alpha"], "lora.alpha")
    _number(cfg["lora"]["dropout"], "lora.dropout")
    if cfg["lora"]["dropout"] >= 1:
        raise ValueError("lora.dropout 必须小于 1")
    if not isinstance(cfg["lora"]["target_modules"], list) or not cfg["lora"]["target_modules"] or any(not isinstance(x, str) or not x for x in cfg["lora"]["target_modules"]):
        raise ValueError("lora.target_modules 必须为非空字符串列表")
    for key in ("run_limit_cny", "project_limit_cny", "max_runtime_seconds"):
        _number(cfg["budget"][key], f"budget.{key}", strict=True)
    _number(cfg["budget"]["reserve_cny"], "budget.reserve_cny")
    if cfg["budget"]["project_limit_cny"] > 500 or cfg["budget"]["reserve_cny"] >= cfg["budget"]["project_limit_cny"]:
        raise ValueError("项目总预算不可超过 500 元，reserve 必须小于总预算")
    if cfg["budget"]["hourly_rate_cny"] is not None:
        _number(cfg["budget"]["hourly_rate_cny"], "budget.hourly_rate_cny", strict=True)
    project_root = (config_path.parent / cfg["project_root"]).resolve()
    cfg["project_root"] = str(project_root)
    for key in ("train_file", "validation_file", "output_dir", "cache_dir", "budget_ledger"):
        cfg[key] = str(_resolve_path(cfg[key], project_root))
    if Path(cfg["budget_ledger"]) != (project_root / "runs" / "budget_ledger.json").resolve():
        raise ValueError("所有训练线必须共享项目 runs/budget_ledger.json，不能拆分账本绕过总预算")
    train_path, val_path = Path(cfg["train_file"]), Path(cfg["validation_file"])
    if train_path == val_path or not train_path.is_file() or not val_path.is_file():
        raise ValueError("train/validation 必须为存在的两个不同文件；请先制备任务数据")
    if any(path.stem.casefold() == "test" for path in (train_path, val_path)):
        raise ValueError("独立 test 文件不得作为训练或选模输入")
    output_dir = Path(cfg["output_dir"])
    if output_dir.is_file() or any(output_dir == path or output_dir in path.parents for path in (train_path, val_path)):
        raise ValueError("output_dir 必须与输入数据目录分开")
    train = read_task_rows(train_path, cfg["stage"])
    validation = read_task_rows(val_path, cfg["stage"])
    check_split_leakage(train, validation)
    source_manifest = None
    if cfg["stage"] == "dpo":
        if not cfg["sft_checkpoint"]:
            raise ValueError("DPO 必须显式指定 sft_checkpoint，禁止直接从原始 Base 开始")
        cfg["sft_checkpoint"] = str(_resolve_path(cfg["sft_checkpoint"], project_root))
        source_manifest = read_artifact_manifest(Path(cfg["sft_checkpoint"]))
        if source_manifest.get("stage") != "sft":
            raise ValueError("DPO 起点必须是 SFT 模型或 SFT adapter")
        if source_manifest.get("base_model") != cfg["base_model"] or source_manifest.get("base_revision") != cfg["base_revision"]:
            raise ValueError("DPO 与 SFT 来源的 base/revision 必须一致")
        if source_manifest.get("template", {}).get("version") != TEMPLATE_VERSION:
            raise ValueError("DPO 必须沿用 SFT 的模板")
        old_train = set(source_manifest.get("training_source_ids", []))
        old_val = set(source_manifest.get("validation_source_ids", []))
        old_train_prompts = set(source_manifest.get("training_prompt_hashes", []))
        old_val_prompts = set(source_manifest.get("validation_prompt_hashes", []))
        if (old_train & {r["source_id"] for r in validation} or old_val & {r["source_id"] for r in train}
                or old_train_prompts & {normalized_prompt_hash(r["prompt"]) for r in validation}
                or old_val_prompts & {normalized_prompt_hash(r["prompt"]) for r in train}):
            raise ValueError("DPO 与来源 SFT 的训练/验证划分交叉，存在数据泄漏")
    elif cfg["sft_checkpoint"] is not None:
        raise ValueError("SFT 应从统一 Base 开始，不能指定 sft_checkpoint")
    identity = {
        "config_hash": canonical_hash(cfg),
        "source_fingerprint": source_fingerprint(),
        "data_hashes": {"train": sha256_file(train_path), "validation": sha256_file(val_path)},
        "template": TEMPLATE_VERSION,
        "source_sft_manifest_hash": canonical_hash(source_manifest) if source_manifest else None,
    }
    resume_path = resolve_resume(output_dir, resume) if resume else None
    if resume_path:
        validate_resume(resume_path, cfg, identity)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir 已包含内容；请使用 --resume 或新的实验目录")
    return ValidatedRun(cfg, train, validation, identity, resume_path, source_manifest)


def resolve_resume(output_dir: Path, resume: str) -> Path:
    if resume == "latest":
        candidates = sorted((p for p in output_dir.glob("checkpoint-*") if p.name[11:].isdigit()), key=lambda p: int(p.name[11:]), reverse=True)
        for path in candidates:
            try:
                manifest = read_artifact_manifest(path)
                cfg = manifest["config"]
                validate_resume(path, cfg, manifest["identity"])
            except (ValueError, KeyError, FileNotFoundError):
                continue
            return path.resolve()
        raise ValueError("没有可恢复的完整 checkpoint")
    path = Path(resume).resolve()
    if not path.is_dir():
        raise ValueError(f"resume 目录不存在: {path}")
    return path


def validate_resume(path: Path, cfg: dict, identity: dict) -> None:
    manifest = read_artifact_manifest(path)
    if not manifest.get("complete_checkpoint"):
        raise ValueError("resume 需要完整训练 checkpoint，不能使用仅供推理的 final")
    if manifest.get("identity") != identity:
        raise ValueError("恢复拒绝：stage/config/data/来源身份变化，请创建新实验")
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "progress.json"]
    if cfg["precision"] == "fp16":
        required.append("scaler.pt")
    if any(not (path / name).is_file() for name in required):
        raise ValueError("checkpoint 缺少 optimizer/scheduler/RNG/data position/scaler，不能完整恢复")
    state = read_json(path / "trainer_state.json")
    progress = read_json(path / "progress.json")
    if state.get("global_step") != progress.get("global_step") or not isinstance(state.get("global_step"), int):
        raise ValueError("checkpoint 的 global_step/data position 不一致")
    expected_examples = state["global_step"] * cfg["batch_size"] * cfg["gradient_accumulation_steps"]
    if progress.get("examples_consumed") != expected_examples or progress.get("plan_hash") != manifest.get("training_plan", {}).get("plan_hash"):
        raise ValueError("checkpoint 的数据位置/plan hash 不一致")


def encode_example(row: dict, tokenizer: Any, max_length: int, stage: str) -> dict:
    """分别编码 prompt/completion，精确界定监督边界；训练、生成共用 prompt 前缀。"""
    prompt_ids = tokenizer.encode(format_prompt(row["prompt"]), add_special_tokens=False)
    if not prompt_ids or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须生成非空 prompt 且提供 EOS")

    def completion_ids(text: str) -> list[int]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids = ids + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(ids) > max_length:
            raise ValueError(f"样本 {row['source_id']} 超出 max_length={max_length}，拒绝静默截断答案")
        return ids

    if stage == "sft":
        answer = completion_ids(row["completion"])
        return {"input_ids": prompt_ids + answer, "attention_mask": [1] * (len(prompt_ids) + len(answer)),
                "labels": [-100] * len(prompt_ids) + answer}
    return {"prompt_input_ids": prompt_ids, "chosen_input_ids": completion_ids(row["chosen"]),
            "rejected_input_ids": completion_ids(row["rejected"])}


def example_token_counts(row: dict, stage: str) -> tuple[int, int]:
    if stage == "sft":
        return len(row["input_ids"]), sum(x != -100 for x in row["labels"][1:])
    return 2 * len(row["prompt_input_ids"]) + len(row["chosen_input_ids"]) + len(row["rejected_input_ids"]), len(row["chosen_input_ids"]) + len(row["rejected_input_ids"])


def make_training_plan(encoded: list[dict], *, token_budget: int, effective_batch_size: int, seed: int, stage: str,
                       languages: list[str] | None = None) -> tuple[list[int], dict]:
    """完整样本、完整优化步；中英按监督 loss_tokens 补足落后语言，不按样本数各半。"""
    if not encoded:
        raise ValueError("训练样本不能为空")
    _positive_int(token_budget, "token_budget")
    _positive_int(effective_batch_size, "effective_batch_size")
    rng = random.Random(seed)
    indices, total, loss_tokens, pending = [], 0, 0, []
    counts = [example_token_counts(r, stage) for r in encoded]
    if any(loss <= 0 for _, loss in counts):
        raise ValueError("每条训练样本必须有有效监督 Token")
    if languages is not None and (len(languages) != len(encoded) or set(languages) != {"zh", "en"}):
        raise ValueError("语言配额要求与样本等长的 languages，且同时包含 zh/en")

    def shuffled_stream(group):
        while True:
            order = list(group)
            rng.shuffle(order)
            yield from order

    streams = {language: shuffled_stream([i for i, value in enumerate(languages) if value == language]) for language in ("zh", "en")} if languages is not None else None
    ordinary_stream = shuffled_stream(range(len(encoded)))
    running_loss = {"zh": 0, "en": 0}
    committed_input, committed_loss = {"zh": 0, "en": 0}, {"zh": 0, "en": 0}
    while True:
        if streams:
            language = "zh" if running_loss["zh"] <= running_loss["en"] else "en"
            index = next(streams[language])
            running_loss[language] += counts[index][1]
        else:
            index = next(ordinary_stream)
        pending.append(index)
        if len(pending) != effective_batch_size:
            continue
        batch_tokens = sum(counts[i][0] for i in pending)
        if total + batch_tokens > token_budget:
            if not indices:
                raise ValueError("train_token_budget 不足一个有效 batch，请增大预算或减小 effective batch")
            summary = {"input_tokens": total, "loss_tokens": loss_tokens,
                       "unused_token_budget": token_budget - total,
                       "optimizer_steps": len(indices) // effective_batch_size,
                       "examples_with_repetition": len(indices), "unique_examples": len(set(indices)),
                       "plan_hash": canonical_hash(indices)}
            if languages is not None:
                summary.update(input_tokens_by_language=committed_input, loss_tokens_by_language=committed_loss,
                    loss_token_fraction_zh=committed_loss["zh"] / loss_tokens,
                    loss_token_relative_imbalance=abs(committed_loss["zh"] - committed_loss["en"]) / loss_tokens,
                    max_language_loss_token_difference=max(value[1] for value in counts),
                    balance_definition="completion loss_tokens; |zh-en| <= largest complete example loss_tokens")
            return indices, summary
        indices.extend(pending)
        total += batch_tokens
        loss_tokens += sum(counts[i][1] for i in pending)
        if languages is not None:
            for i in pending:
                committed_input[languages[i]] += counts[i][0]
                committed_loss[languages[i]] += counts[i][1]
        pending = []


def make_manifest(run: ValidatedRun) -> dict:
    cfg = run.config
    previous = run.source_manifest or {}
    return {
        "schema_version": 1, "workflow": "posttrain", "stage": cfg["stage"], "method": cfg["method"],
        "checkpoint_type": "adapter" if cfg["method"] == "lora" else "full",
        "base_model": cfg["base_model"], "base_revision": cfg["base_revision"],
        "config": cfg, "config_hash": run.identity["config_hash"], "identity": run.identity,
        "template": {"version": TEMPLATE_VERSION, "text": TEMPLATE_TEXT, "eos": "tokenizer.eos_token_id"},
        "training_source_ids": sorted({r["source_id"] for r in run.train_rows} | set(previous.get("training_source_ids", []))),
        "validation_source_ids": sorted({r["source_id"] for r in run.validation_rows} | set(previous.get("validation_source_ids", []))),
        "training_prompt_hashes": sorted({normalized_prompt_hash(r["prompt"]) for r in run.train_rows} | set(previous.get("training_prompt_hashes", []))),
        "validation_prompt_hashes": sorted({normalized_prompt_hash(r["prompt"]) for r in run.validation_rows} | set(previous.get("validation_prompt_hashes", []))),
        "sft_checkpoint": cfg["sft_checkpoint"],
        "reference_semantics": "frozen_sft_start" if cfg["stage"] == "dpo" else None,
        "weights_base_checkpoint": cfg["sft_checkpoint"] if cfg["stage"] == "dpo" and cfg["method"] == "lora" and previous.get("checkpoint_type") == "full" else None,
        "complete_checkpoint": False,
        "checkpoint_selection": "final_step_is_not_validation_best; select a checkpoint using validation before final test",
    }


