"""训练指标的可选 W&B 记录；配置校验不依赖 W&B 或训练框架。"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import math
from numbers import Real
import os
from pathlib import Path
import random
import re
import sys
from typing import Any
from uuid import uuid4
import warnings

from .io import read_json, write_json

WANDB_VERSION = "0.30.0"
WANDB_DEFAULTS = {"wandb": False, "wandb_project": "bilingual-training-lab", "wandb_entity": None,
                  "wandb_mode": "online", "wandb_run_name": None}


def validate_tracking(config: dict) -> None:
    values = {**WANDB_DEFAULTS, **{key: config[key] for key in WANDB_DEFAULTS if key in config}}
    if type(values["wandb"]) is not bool:
        raise ValueError("wandb must be a boolean")
    if values["wandb_mode"] not in ("online", "offline"):
        raise ValueError("wandb_mode must be online or offline")
    for name in ("wandb_project", "wandb_entity"):
        value = values[name]
        if value is None and name == "wandb_entity":
            continue
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
            raise ValueError(f"Invalid {name}; use a project/team slug")
    name = values["wandb_run_name"]
    if name is not None and (not isinstance(name, str) or not name.strip() or len(name) > 128):
        raise ValueError("wandb_run_name must be a nonempty string of at most 128 characters")


def configure_tracking_environment(root: Path) -> None:
    """所有 SDK 配置、登录凭据和缓存留在项目目录；不改系统配置。"""
    root = root.resolve()
    if os.name == "nt" and root.drive.upper() != "E:":
        raise ValueError("Local W&B state must stay on E:")
    cache = root / ".cache/wandb"
    locations = {"WANDB_CONFIG_DIR": cache / "config", "WANDB_CACHE_DIR": cache / "cache",
                 "WANDB_DATA_DIR": cache / "staging", "WANDB_ARTIFACT_DIR": root / "artifacts/wandb",
                 "WANDB_DIR": root / "runs/wandb"}
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["NETRC"] = str(cache / "netrc")
    os.environ["WANDB_CREDENTIALS_FILE"] = str(cache / "credentials.json")
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ.pop("WANDB_DISABLED", None)


def numeric_metrics(values: dict, prefix: str = "") -> dict:
    """嵌套标量用 / 展开；不上传数据样例、文本和文件路径。"""
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(numeric_metrics(value, name))
        elif isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
            result[name] = value
    return result


@contextmanager
def preserved_rng():
    """SDK 不能推进训练使用的 Python/NumPy/Torch RNG；不主动导入训练依赖。"""
    python_state = random.getstate()
    numpy, torch = sys.modules.get("numpy"), sys.modules.get("torch")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch_state = torch.get_rng_state() if torch is not None else None
    cuda_state = torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class Tracker:
    def __init__(self, run=None, output: Path | None = None):
        self.run, self.output = run, output
        self.failed = False

    def log(self, values: dict, step: int) -> None:
        if self.run is None or self.failed:
            return
        metrics = numeric_metrics(values)
        metrics["optimizer_step"] = int(step)
        try:
            with preserved_rng():
                # 使用自定义横轴，Trainer 同一步的 train/eval 多次 log 都会保留。
                self.run.log(metrics, commit=True)
        except Exception as exc:
            self.failed = True
            write_json(self.output / "wandb_error.json", {"status": "logging_failed", "error_type": type(exc).__name__})
            warnings.warn("W&B logging failed; training continues with local metrics.jsonl", RuntimeWarning)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is None:
            return
        run, self.run = self.run, None
        try:
            with preserved_rng():
                run.finish(exit_code=exit_code or int(self.failed))
        except Exception:
            warnings.warn("W&B finish failed; retain local logs and use wandb sync later", RuntimeWarning)


def start_tracker(config: dict, output: Path, project_root: Path, *, job_type: str,
                  run_config: dict | None = None, resume: bool = False) -> Tracker:
    """在线恢复复用 run id；离线恢复新建分段并沿用 group，避免覆盖离线历史。"""
    validate_tracking(config)
    cfg = {**WANDB_DEFAULTS, **{key: config[key] for key in WANDB_DEFAULTS if key in config}}
    if not cfg["wandb"]:
        return Tracker()
    output = Path(output)
    state_path = output / "wandb_run.json"
    identity = {"project": cfg["wandb_project"], "entity": cfg["wandb_entity"], "mode": cfg["wandb_mode"], "job_type": job_type}
    previous = read_json(state_path) if state_path.exists() else None
    if resume:
        if previous is None or previous.get("identity") != identity:
            raise ValueError("W&B resume requires the original wandb_run.json and tracking identity")
        experiment_id = previous["experiment_id"]
        run_id = previous["id"] if cfg["wandb_mode"] == "online" else uuid4().hex[:12]
    else:
        if previous is not None:
            raise ValueError("W&B output already has a run; resume it or choose a new directory")
        experiment_id = run_id = uuid4().hex[:12]
    configure_tracking_environment(Path(project_root))
    output.mkdir(parents=True, exist_ok=True)
    with preserved_rng():
        try:
            sdk = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(f"Install wandb=={WANDB_VERSION} in the remote training environment") from exc
        if sdk.__version__ != WANDB_VERSION:
            raise RuntimeError(f"W&B version must be {WANDB_VERSION}; found {sdk.__version__}")
        options = dict(project=cfg["wandb_project"], entity=cfg["wandb_entity"] or (previous or {}).get("resolved_entity"),
            name=cfg["wandb_run_name"] or output.name, id=run_id, group=experiment_id, job_type=job_type,
            mode=cfg["wandb_mode"], force=cfg["wandb_mode"] == "online", dir=str(output),
            config=run_config or {}, save_code=False,
            settings=sdk.Settings(disable_code=True, disable_git=True, console="off", init_timeout=30, finish_timeout=15))
        if cfg["wandb_mode"] == "online":
            options["resume"] = "must" if resume else "never"
        run = sdk.init(**options)
        tracker = Tracker(run, output)
        try:
            run.define_metric("optimizer_step")
            run.define_metric("*", step_metric="optimizer_step")
            write_json(state_path, {"schema_version": 1, "identity": identity, "experiment_id": experiment_id,
                "id": run_id, "resolved_entity": run.entity, "url": run.url if cfg["wandb_mode"] == "online" else None,
                "segments": [*((previous or {}).get("segments", [])), run_id] if cfg["wandb_mode"] == "offline" else [run_id]})
        except Exception:
            tracker.finish(exit_code=1)
            raise
    return tracker
