"""自研 SFT 的纯配置协议；不导入 Torch，不改变旧 llm_lab 源码身份。"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from llm_lab.common.io import canonical_hash, read_json, sha256_file
from llm_lab.common.tracking import validate_tracking

GIB = 1024**3
PARAMETER_COUNT = 303_350_784
TRAIN_DEFAULTS = {
    "device": "cuda", "dtype": "bf16", "seed": 42, "deterministic": True,
    "micro_batch_size": 8, "gradient_accumulation_steps": 4,
    "epochs": 1.0, "stop_epochs": None, "learning_rate": 2e-5,
    "warmup_ratio": .03, "min_lr_ratio": .1, "weight_decay": .01,
    "max_grad_norm": 1.0, "betas": [.9, .95], "eval_interval": 100,
    "save_interval": 100, "log_interval": 10, "tensorboard": True,
    "activation_checkpointing": False, "eval_micro_batch_size": 8,
    "retention_eval_batches": 2, "retention_eval_seq_len": 512,
    "wandb": True, "wandb_project": "bilingual-training-lab", "wandb_entity": None,
    "wandb_mode": "online", "wandb_run_name": None,
}
BUDGET_DEFAULTS = {
    "hourly_rate_cny": 3.03, "run_limit_cny": 15, "project_limit_cny": 500,
    "reserve_cny": 100, "max_runtime_seconds": 7200, "ledger_path": "runs/budget_ledger.json",
}
TOP_FIELDS = {
    "schema_version", "stage", "method", "project_root", "base_checkpoint", "data_manifest",
    "output_dir", "train_split", "train", "budget", "min_free_bytes", "fixture", "experiment",
    "base_checkpoint_sha256", "data_manifest_sha256", "data_identity", "runtime_sha256",
}


def under(path, root) -> Path:
    """所有训练配置、日志、缓存引用均留在当前项目，拒绝符号链接逃逸。"""
    root, path = Path(root).resolve(), Path(path)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError(f"SFT 路径越出项目或为符号链接：{path}")
    return path.resolve()


def _finite(value, name, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须为有限数值")
    if positive and value <= 0 or minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} 超出允许范围")


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or set(config) - TOP_FIELDS:
        raise ValueError("未知的 SFT 配置字段")
    if type(config.get('schema_version')) is not int or config.get("schema_version") != 1 or config.get("stage") != "sft" or config.get("method") != "full":
        raise ValueError("此入口仅支持 schema=1 的全参数 SFT；LoRA/DPO 是后续独立实验")
    if config.get("train_split") not in {"train", "pilot"}:
        raise ValueError("训练分区只允许 train/pilot，禁止用 validation/test 训练")
    if type(config.get("fixture")) is not bool:
        raise ValueError("fixture 必须为 bool")
    train, budget = config["train"], config["budget"]
    if set(train) != set(TRAIN_DEFAULTS) or set(budget) != set(BUDGET_DEFAULTS):
        raise ValueError("train/budget 必须符合完整 SFT 配置协议")
    if train["device"] not in {"cpu", "cuda"} or train["dtype"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("无效的设备或精度")
    if train["device"] == "cpu" and (train["dtype"] != "fp32" or not config["fixture"]):
        raise ValueError("CPU 仅用于显式 fixture=true 的 FP32 正确性测试")
    if config["fixture"] and train["device"] != "cpu":
        raise ValueError("fixture 不允许启动 GPU 训练")
    if type(train["seed"]) is not int or not 0 <= train["seed"] < 2**32:
        raise ValueError("seed 必须为 [0,2**32) 整数")
    for key in ("micro_batch_size", "gradient_accumulation_steps", "eval_interval", "save_interval", "log_interval", "eval_micro_batch_size", "retention_eval_seq_len"):
        if type(train[key]) is not int or train[key] <= 0:
            raise ValueError(f"train.{key} 必须为正整数")
    if type(train['retention_eval_batches']) is not int or train['retention_eval_batches'] < 0:
        raise ValueError('retention_eval_batches 必须为非负整数')
    if train['retention_eval_seq_len'] > 1024:
        raise ValueError('retention_eval_seq_len 不得超过当前Base的1024上下文')
    for key in ("deterministic", "tensorboard", "activation_checkpointing"):
        if type(train[key]) is not bool:
            raise ValueError(f"train.{key} 必须为 bool")
    _finite(train["epochs"], "epochs", positive=True, maximum=20)
    if train["stop_epochs"] is not None:
        _finite(train["stop_epochs"], "stop_epochs", positive=True, maximum=train["epochs"])
    _finite(train["learning_rate"], "learning_rate", positive=True)
    _finite(train["warmup_ratio"], "warmup_ratio", minimum=0, maximum=1)
    _finite(train["min_lr_ratio"], "min_lr_ratio", minimum=0, maximum=1)
    _finite(train["weight_decay"], "weight_decay", minimum=0)
    _finite(train["max_grad_norm"], "max_grad_norm", positive=True)
    if not isinstance(train["betas"], list) or len(train["betas"]) != 2:
        raise ValueError("betas 必须包含两个数值")
    for beta in train["betas"]:
        _finite(beta, "beta", minimum=0)
        if beta >= 1:
            raise ValueError("AdamW beta 必须小于 1")
    validate_tracking(train)
    for key in ("run_limit_cny", "project_limit_cny", "max_runtime_seconds"):
        _finite(budget[key], key, positive=True)
    _finite(budget["reserve_cny"], "reserve_cny", minimum=0)
    _finite(budget["hourly_rate_cny"], "hourly_rate_cny", minimum=0)
    if budget["project_limit_cny"] > 500 or budget["reserve_cny"] >= budget["project_limit_cny"]:
        raise ValueError("项目总预算不超过 500 元，余量必须小于项目额度")
    if type(config["min_free_bytes"]) is not int or config["min_free_bytes"] < 0:
        raise ValueError("min_free_bytes 必须为非负整数")
    if train["device"] == "cuda":
        if budget["hourly_rate_cny"] <= 0 or config["min_free_bytes"] < 8 * GIB:
            raise ValueError("GPU 实验需实际小时价与至少 8GiB 磁盘水位")
        if budget["project_limit_cny"] != 500 or budget["reserve_cny"] != 100:
            raise ValueError("保留项目 500 元总额与 100 元余量")
        if Path(budget["ledger_path"]) != Path(config["project_root"]) / "runs/budget_ledger.json":
            raise ValueError("GPU 实验必须使用项目共享 budget_ledger.json")
    if not isinstance(config["experiment"], dict):
        raise ValueError("experiment 必须为只含记录信息的对象")
    for key in ("base_checkpoint_sha256", "data_manifest_sha256", "data_identity", "runtime_sha256"):
        value = config.get(key)
        if value is not None and (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError(f"{key} 必须为空或完整 SHA256")


def read_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = read_json(path)
    if not isinstance(raw, dict) or set(raw) - TOP_FIELDS:
        raise ValueError("未知的 SFT 配置字段")
    if not isinstance(raw.get("train", {}), dict) or set(raw.get("train", {})) - set(TRAIN_DEFAULTS):
        raise ValueError("未知 train 字段")
    if not isinstance(raw.get("budget", {}), dict) or set(raw.get("budget", {})) - set(BUDGET_DEFAULTS):
        raise ValueError("未知 budget 字段")
    config = copy.deepcopy(raw)
    if not isinstance(config.get('project_root','../..'),str) or not config.get('project_root','../..'):
        raise ValueError('project_root 必须为非空路径字符串')
    root = (path.parent / config.get("project_root", "../..")).resolve()
    config.update(project_root=str(root), train={**copy.deepcopy(TRAIN_DEFAULTS), **raw.get("train", {})},
                  budget={**copy.deepcopy(BUDGET_DEFAULTS), **raw.get("budget", {})})
    if raw.get('fixture') and 'retention_eval_batches' not in raw.get('train', {}):
        config['train']['retention_eval_batches'] = 0
    for key, default in (("schema_version", 1), ("stage", "sft"), ("method", "full"),
                         ("train_split", "train"), ("min_free_bytes", 8 * GIB), ("fixture", False), ("experiment", {})):
        config.setdefault(key, default)
    for key in ("base_checkpoint", "data_manifest", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"缺少 {key}")
        config[key] = str(under(config[key], root))
    if not isinstance(config['budget']['ledger_path'],str) or not config['budget']['ledger_path']:
        raise ValueError('ledger_path 必须为非空路径字符串')
    config["budget"]["ledger_path"] = str(under(config["budget"]["ledger_path"], root))
    validate_config(config)
    return config


def runtime_fingerprint(root: str | Path | None = None) -> str:
    """冻结模型、训练包与启动器的内容身份；不将文档和测试纳入恢复身份。"""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    files = [path for package in ("llm_lab", "llm_sft") for path in (root / "src" / package).rglob("*.py")]
    launcher = root / "scripts/train_sft.py"
    if launcher.is_file():
        files.append(launcher)
    if not files:
        raise ValueError("没有找到可冻结的 SFT 运行源码")
    return canonical_hash({path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)})


def epoch_sample_counts(examples: int, epochs: float) -> list[int]:
    if type(examples) is not int or examples <= 0:
        raise ValueError("训练样本数必须为正整数")
    full = math.floor(epochs)
    counts = [examples] * full
    remainder = epochs - full
    if remainder > 1e-12:
        counts.append(min(examples, math.ceil(examples * remainder)))
    return counts


def plan_details(config: dict, example_count: int | None = None) -> dict:
    if example_count is None:
        manifest = read_json(config["data_manifest"])
        example_count = manifest["splits"][config["train_split"]]["examples"]
    train = config["train"]
    counts = epoch_sample_counts(example_count, train["epochs"])
    effective = train["micro_batch_size"] * train["gradient_accumulation_steps"]
    max_steps = sum(math.ceil(count / effective) for count in counts)
    target = sum(counts) if train["stop_epochs"] is None else math.ceil(example_count * train["stop_epochs"])
    actual, stop_step = 0, 0
    for count in counts:
        for offset in range(0, count, effective):
            if actual >= target:
                break
            actual += min(effective, count - offset)
            stop_step += 1
        if actual >= target:
            break
    return {"train_examples": example_count, "train_split": config["train_split"],
            "effective_batch_size": effective, "epoch_sample_counts": counts,
            "updates_per_epoch": [math.ceil(count / effective) for count in counts],
            "max_steps": max_steps, "warmup_steps": math.ceil(max_steps * train["warmup_ratio"]),
            "schedule_examples": sum(counts), "planned_examples": actual,
            "stop_step": stop_step, "actual_examples_at_stop": actual,
            "stop_epochs": train["stop_epochs"], "epochs": train["epochs"]}
