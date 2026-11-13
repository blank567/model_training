"""显式、严格的配置结构；拼错字段会报错，配置不执行任意 Python。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from typing import TypeVar

from .common.io import read_json
from .common.tracking import validate_tracking

T = TypeVar("T")


def strict_dataclass(cls: type[T], values: dict) -> T:
    unknown = set(values) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.1
    tie_embeddings: bool = True
    activation_checkpointing: bool = True

    def validate(self) -> None:
        for name in ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "ffn_dim", "max_seq_len"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.dim % self.n_heads or self.n_heads % self.n_kv_heads or (self.dim // self.n_heads) % 2:
            raise ValueError("GQA requires dim divisible by n_heads, n_heads divisible by n_kv_heads, and even head_dim")
        if not math.isfinite(self.dropout) or not math.isfinite(self.rope_theta) or not 0 <= self.dropout < 1 or self.rope_theta <= 0:
            raise ValueError("Invalid dropout or RoPE theta")


@dataclass
class TrainConfig:
    device: str = "cuda"
    dtype: str = "bf16"
    seed: int = 42
    deterministic: bool = True
    seq_len: int = 1024
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    learning_rate: float = 0.0003
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 8
    save_interval: int = 100
    log_interval: int = 10
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "bilingual-training-lab"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_run_name: str | None = None

    def validate(self) -> None:
        validate_tracking(asdict(self))
        if self.device not in ("cpu", "cuda") or self.dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError("Supported device: cpu/cuda; dtype: fp32/fp16/bf16")
        if self.device == "cpu" and self.dtype != "fp32":
            raise ValueError("CPU correctness checks use fp32")
        for name in ("seq_len", "micro_batch_size", "gradient_accumulation_steps", "max_steps", "eval_interval", "eval_batches", "save_interval", "log_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"train.{name} must be a positive integer")
        if not 0 <= self.warmup_steps <= self.max_steps or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid warmup or minimum learning rate ratio")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.weight_decay, self.max_grad_norm, self.min_lr_ratio)):
            raise ValueError("Optimizer values must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer parameters")


@dataclass
class ExperimentConfig:
    data_manifest: str
    output_dir: str
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    budget: dict = field(default_factory=lambda: {"hourly_rate_cny": None, "run_limit_cny": 50,
                                               "project_limit_cny": 500, "reserve_cny": 100,
                                               "max_runtime_seconds": 1800, "ledger_path": "runs/budget_ledger.json"})
    project_root: str = "."

    def validate(self) -> None:
        self.model.validate()
        self.train.validate()
        if self.train.seq_len > self.model.max_seq_len:
            raise ValueError("Training sequence length exceeds model context")
        # 语言逐个交替，每次更新必须让两种语言处理相同数量的有效 Token。
        if self.train.gradient_accumulation_steps % 2:
            raise ValueError("Bilingual accumulation must be even for exact 50/50 token balance per optimizer step")
        allowed = {"hourly_rate_cny", "run_limit_cny", "project_limit_cny", "reserve_cny", "max_runtime_seconds", "ledger_path"}
        if not isinstance(self.budget, dict) or set(self.budget) - allowed:
            raise ValueError("Unknown budget fields")
        for name in ("run_limit_cny", "project_limit_cny", "max_runtime_seconds"):
            value = self.budget.get(name, {"run_limit_cny": 50, "project_limit_cny": 500, "max_runtime_seconds": 1800}[name])
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"budget.{name} must be positive and finite")
        if self.budget.get("project_limit_cny", 500) > 500:
            raise ValueError("Total project budget cannot exceed 500 CNY")
        root = Path(self.project_root).resolve()
        ledger = Path(self.budget.get("ledger_path", "runs/budget_ledger.json"))
        ledger = ledger.resolve() if ledger.is_absolute() else (root / ledger).resolve()
        if self.train.device == "cuda" and ledger != root / "runs" / "budget_ledger.json":
            raise ValueError("Paid runs must share the project runs/budget_ledger.json")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = read_json(path)
    values = dict(raw)
    if "project_root" in values:
        root = (path.parent / values["project_root"]).resolve()
    else:
        root = next((parent for parent in path.parents if (parent / "pyproject.toml").is_file()), Path.cwd()).resolve()
    values["project_root"] = str(root)
    for key in ("data_manifest", "output_dir"):
        if key in values:
            values[key] = str((root / values[key]).resolve())
    if "budget" in values:
        values["budget"] = dict(values["budget"])
        ledger = values["budget"].get("ledger_path", "runs/budget_ledger.json")
        values["budget"]["ledger_path"] = str((root / ledger).resolve())
    values["model"] = strict_dataclass(ModelConfig, values.get("model", {}))
    values["train"] = strict_dataclass(TrainConfig, values.get("train", {}))
    config = strict_dataclass(ExperimentConfig, values)
    config.validate()
    return config

