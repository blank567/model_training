"""数据步骤共享的小型工具；不在导入时联网或写文件。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator


LANGUAGES = ("en", "zh")
SPLITS = ("train", "validation", "test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _category_c_pattern() -> re.Pattern[str]:
    """按当前 UCD 构建允许范围的否定类；首次清洗才初始化，无硬编码字符黑名单。"""
    ranges = []
    first = None
    for code in range(0x110000):
        keep = code in (9, 10) or unicodedata.category(chr(code))[0] != "C"
        if keep:
            if first is None:
                first = code
        elif first is not None:
            ranges.append((first, code - 1))
            first = None
    if first is not None:
        ranges.append((first, 0x10FFFF))
    # 否定允许类较直接罗列 C 类范围快；两者都完整包含控制/格式/代理/私用/未指派类别。
    pieces = [f"\\U{begin:08x}" + (f"-\\U{end:08x}" if end != begin else "") for begin, end in ranges]
    return re.compile("[^" + "".join(pieces) + "]")


def normalize_text(text: str) -> str:
    """统一 Unicode 和空白，保留段落；不做会改变词义的繁简转换。"""
    if not isinstance(text, str):
        raise ValueError("text 必须是字符串")
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _category_c_pattern().sub("", text)
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_name(value: str) -> str:
    value = {"dev": "validation", "valid": "validation", "val": "validation"}.get(value, value)
    if value not in SPLITS:
        raise ValueError(f"未知 split/partition: {value!r}")
    return value


def assigned_split(source_id: str, seed: int, fractions: dict[str, float]) -> str:
    """先按源 ID 分区；同源的语言版本和后续任务变体共享同一个分区。"""
    values = [float(fractions.get(key, 0.0)) for key in SPLITS]
    if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-8:
        raise ValueError("split_fractions 必须非负且合计为 1")
    number = int(text_hash(f"{seed}:{source_id}")[:16], 16) / 2**64
    if number < values[0]:
        return "train"
    return "validation" if number < values[0] + values[1] else "test"


def json_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"


def read_jsonl(path: str | Path, *, max_line_bytes: int = 8 * 1024 * 1024) -> Iterator[dict[str, Any]]:
    """逐行且有单行内存上限，错误带行号，避免静默吞掉坏样本。"""
    with Path(path).open("rb") as handle:
        index = 0
        while True:
            raw = handle.readline(max_line_bytes + 1)
            if not raw:
                break
            index += 1
            if len(raw) > max_line_bytes:
                raise ValueError(f"{path}:{index}: 单行超过 {max_line_bytes} 字节")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8-sig" if index == 1 else "utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{index}: 非法 UTF-8 JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index}: JSONL 每行必须是对象")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json_line(row))
            count += 1
    return count


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    root = (path.parent / config["project_root"]).resolve() if "project_root" in config else Path.cwd()
    return config, root


def config_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} 必须显式设置为正整数")
    return value


def make_output_dir(path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录非空：{path}；请换目录或显式设置 overwrite=true")
    path.mkdir(parents=True, exist_ok=True)


def ensure_under(path: Path, root: Path) -> Path:
    """manifest 中的相对路径不能越出它自己的数据目录。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"路径越出数据目录：{path}")
    return resolved


def input_provenance(paths: list[str], root: Path, max_hash_bytes: int) -> list[dict[str, Any]]:
    """记录来源指纹；大于本步字节预算的原文件不再额外全量扫描。"""
    result = []
    for value in paths:
        path = config_path(value, root)
        size = path.stat().st_size
        item = {"path": str(path), "size_bytes": size,
                "sha256": sha256_file(path) if size <= max_hash_bytes else None}
        upstream = path.parent / "acquisition_manifest.json"
        if upstream.exists() and upstream.stat().st_size <= 1024 * 1024:
            item["acquisition_manifest_sha256"] = sha256_file(upstream)
            item["acquisition"] = json.loads(upstream.read_text(encoding="utf-8"))
        result.append(item)
    return result
