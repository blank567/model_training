"""只在完整 optimizer update 的边界保存，临时文件成功后原子替换。"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(path: str | Path) -> dict:
    # Checkpoint 含 RNG 等 Python 状态；只加载自己生成且可信的文件。
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return value

