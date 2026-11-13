"""磁盘索引与顺序编码的预训练制备；仅在命令显式运行时写入产物。

此版本不支持从未完成目录恢复。manifest.json 仅在容量、隔离和校验通过后原子发布。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from llm_lab.tokenization.bpe import EOS_TOKEN, load_tokenizer, train_bpe

from .common import (
    LANGUAGES, SPLITS, assigned_split, config_path, ensure_under, json_line,
    load_config, normalize_text, positive_int, read_jsonl, sha256_file, split_name, text_hash,
)


PRIORITY = {"test": 0, "validation": 1, "train": 2}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _progress(output: Path, phase: str, **values) -> None:
    _atomic_json(output / "preparation_state.json", {"schema_version": 1, "status": "running", "phase": phase,
                 "resume_supported": False, "updated_at": datetime.now(timezone.utc).isoformat(), **values})


class DiskGuard:
    """间隔检查可用空间，并为下一个有界写入区间预留余量。"""

    def __init__(self, path: Path, minimum: int, interval: int = 16 * 1024 * 1024):
        if minimum < 0 or interval < 1:
            raise ValueError("磁盘水位和检查间隔非法")
        self.path, self.minimum, self.interval = path, minimum, interval
        self.pending = 0
        self.check()

    def check(self, reserve: int = 0) -> None:
        existing = self.path
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if free < self.minimum + self.interval + reserve:
            raise OSError(f"磁盘低于安全水位：free={free}, min_free_disk_bytes={self.minimum}, reserve={reserve}")
        self.pending = 0

    def before_write(self, size: int) -> None:
        if self.pending + size >= self.interval:
            self.check(reserve=size)
        self.pending += size


def resolve_inputs(config: dict[str, Any], root: Path) -> list[Path]:
    """启动时固定输入快照；只消费完成的 JSONL，不追踪正在增长的目录。"""
    direct, patterns = config.get("inputs", []), config.get("input_globs", [])
    if not isinstance(direct, list) or not isinstance(patterns, list):
        raise ValueError("inputs/input_globs 必须是列表")
    values = direct + patterns
    if not values:
        raise ValueError("inputs/input_globs 至少需要一个输入")
    paths = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("输入路径必须是字符串")
        pattern = str(config_path(value, root))
        matches = glob.glob(pattern, recursive=True) if glob.has_magic(pattern) else [pattern]
        if not matches:
            raise FileNotFoundError(f"输入 glob 没有匹配完成的 JSONL：{value}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file() or path.suffix.lower() != ".jsonl":
                raise ValueError(f"只接受完整的 .jsonl 输入文件：{path}")
            paths.add(path)
    return sorted(paths, key=lambda path: path.as_posix())


def _check_acquisition(config: dict[str, Any], root: Path, inputs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not config.get("acquisition_manifest"):
        return None
    path = config_path(config["acquisition_manifest"], root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "streaming_wikipedia_acquisition" or manifest.get("status") != "completed":
        raise ValueError("acquisition_manifest 必须是已完成的流式 Wikipedia 获取清单")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("获取清单 revision 必须固定为 40 位 SHA")
    required = config.get("required_source_revision")
    if required is not None and required != revision:
        raise ValueError("获取清单 revision 与 required_source_revision 不一致")
    files = {}
    for record in manifest.get("files", []):
        raw = ensure_under(path.parent / record["path"], path.parent)
        if str(raw) in files:
            raise ValueError("获取清单有重复 raw 路径")
        if record.get("source_revision") != revision:
            raise ValueError("获取清单 raw revision 不一致")
        source = record.get("source", {})
        if not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
            raise ValueError("获取清单缺少固定 Parquet SHA256")
        positive_int(source, "size_bytes")
        files[str(raw)] = record
    for item in inputs:
        record = files.get(item["path"])
        if record is None or record["size_bytes"] != item["size_bytes"] or record["sha256"] != item["sha256"]:
            raise ValueError("raw 输入未匹配 acquisition_manifest 的大小/SHA256")
    return {"path": str(path), "sha256": sha256_file(path), "revision": revision,
            "repository": manifest.get("repository"), "snapshot": manifest.get("snapshot"),
            "matched_raw_files": len(inputs)}


def _open_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-32768;
        PRAGMA mmap_size=0;
        PRAGMA wal_autocheckpoint=1000;
        CREATE TABLE sources(source_id TEXT PRIMARY KEY, split TEXT NOT NULL, blocked INTEGER NOT NULL DEFAULT 0,
                             protected_rank INTEGER NOT NULL DEFAULT 3);
        CREATE TABLE contents(text_sha TEXT PRIMARY KEY, split TEXT NOT NULL, source_id TEXT NOT NULL);
        CREATE TABLE source_hashes(text_sha TEXT NOT NULL, source_id TEXT NOT NULL, PRIMARY KEY(text_sha, source_id));
        CREATE INDEX hashes_by_source ON source_hashes(source_id);
        CREATE TABLE records(source_id TEXT NOT NULL, language TEXT NOT NULL, text_sha TEXT NOT NULL,
                             PRIMARY KEY(source_id, language));
        CREATE TABLE frozen_sources(source_id TEXT PRIMARY KEY, split TEXT NOT NULL);
        CREATE TABLE frozen_hashes(text_sha TEXT PRIMARY KEY, split TEXT NOT NULL);
        CREATE TABLE frozen_records(source_id TEXT NOT NULL, language TEXT NOT NULL, text_sha TEXT NOT NULL,
                                    split TEXT NOT NULL, converted INTEGER NOT NULL DEFAULT 0,
                                    PRIMARY KEY(source_id,language,text_sha));
        CREATE TABLE emitted_holdout_hashes(text_sha TEXT PRIMARY KEY, split TEXT NOT NULL);
    """)
    return connection


def _audit_record(row: dict[str, Any]) -> dict[str, str]:
    source_id, language = row.get("source_id"), row.get("language")
    digest = row.get("text_sha256")
    if not isinstance(source_id, str) or not source_id or language not in LANGUAGES:
        raise ValueError("冻结审计需要非空 source_id 和 en/zh language")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("冻结审计缺少合法 text_sha256")
    return {"source_id": source_id, "language": language, "split": split_name(row.get("split", "")), "text_sha256": digest}


def _load_frozen(config: dict[str, Any], root: Path, index: sqlite3.Connection,
                 audit, guard: DiskGuard) -> dict[str, Any] | None:
    if not config.get("frozen_holdout_manifest"):
        return None
    path = config_path(config["frozen_holdout_manifest"], root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest["tokenizer"].get("trained_on_split") != "train":
        raise ValueError("冻结来源必须为 train-only tokenizer 的 schema_version=1 manifest")
    old_audit = ensure_under(path.parent / "document_audit.jsonl", path.parent)
    if not old_audit.is_file():
        raise FileNotFoundError("冻结 holdout 需要旧 document_audit.jsonl，不能仅依赖 token 文件")
    if manifest.get("audit", {}).get("sha256") and sha256_file(old_audit) != manifest["audit"]["sha256"]:
        raise ValueError("冻结 document_audit.jsonl SHA256 不匹配")
    for number, raw in enumerate(read_jsonl(old_audit), 1):
        row = _audit_record(raw)
        index.execute("INSERT OR IGNORE INTO frozen_records(source_id,language,text_sha,split) VALUES (?,?,?,?)",
                      (row["source_id"], row["language"], row["text_sha256"], row["split"]))
        for table, key_name, key in (("frozen_sources", "source_id", row["source_id"]),
                                     ("frozen_hashes", "text_sha", row["text_sha256"])):
            previous = index.execute(f"SELECT split FROM {table} WHERE {key_name}=?", (key,)).fetchone()
            if previous and previous[0] != row["split"]:
                # 新审计兼作隔离账本：弃用别名可以连接两个已冻结 holdout。
                # 该连接只加强 train 排除，不会把任何新文本放入已冻结的验证/测试 bin。
                ledger = manifest.get("audit", {}).get("role") == "isolation_ledger"
                if not ledger or "train" in (previous[0], row["split"]):
                    raise ValueError("旧审计清单存在跨 split 的源 ID/文档 hash")
                protected = min((previous[0], row["split"]), key=PRIORITY.__getitem__)
                index.execute(f"UPDATE {table} SET split=? WHERE {key_name}=?", (protected, key))
            index.execute(f"INSERT OR IGNORE INTO {table} VALUES (?, ?)", (key, row["split"]))
        # 包括旧 token 预算未输出的 holdout 文档；未来冻结时依然必须排除它们。
        if row["split"] != "train":
            line = json_line(dict(row, status="frozen_holdout"))
            guard.before_write(len(line.encode("utf-8")))
            audit.write(line)
        if number % 1000 == 0:
            index.commit()
            guard.check()
    index.commit()
    return {"path": str(path), "sha256": sha256_file(path), "manifest": manifest,
            "audit_path": str(old_audit), "audit_sha256": sha256_file(old_audit)}


def _prepare_frozen_texts(config: dict[str, Any], root: Path, output: Path, index: sqlite3.Connection,
                          frozen: dict[str, Any] | None, normalizer, audit, guard: DiskGuard) -> None:
    if frozen is None:
        return
    old_root = Path(frozen["path"]).parent
    documents = old_root / "documents.jsonl"
    new_mode = normalizer.provenance["mode"]
    old_normalization = frozen["manifest"].get("chinese_normalization", {"mode": "none"})
    already_same = new_mode == old_normalization.get("mode")
    if new_mode == "t2s" and old_normalization.get("mode") == "t2s":
        if old_normalization != normalizer.provenance:
            raise ValueError("冻结 holdout 的 t2s normalizer provenance 已变化，不能静默改写")
    frozen["rebuild_zh"] = new_mode == "t2s" and not already_same
    if not documents.is_file():
        if frozen["rebuild_zh"]:
            raise FileNotFoundError("转换旧中文 holdout 需要旧 documents.jsonl emitted 文本")
        return
    expected_docs = frozen["manifest"].get("documents", {}).get("sha256")
    if expected_docs and sha256_file(documents) != expected_docs:
        raise ValueError("冻结 documents.jsonl SHA256 不匹配")
    old_tokenizer = None
    if frozen["rebuild_zh"]:
        token_info = frozen["manifest"]["tokenizer"]
        token_path = ensure_under(old_root / token_info["path"], old_root)
        if sha256_file(token_path) != token_info["sha256"]:
            raise ValueError("旧 tokenizer SHA256 不匹配，无法验证 holdout 文本")
        old_tokenizer = load_tokenizer(token_path)
    parents = {split: {"sha256": hashlib.sha256(), "token_count": 0} for split in ("validation", "test")}

    def protect(row):
        source_id, language = row.get("source_id"), row.get("language")
        if language != "zh":
            return None
        original = normalize_text(row.get("text"))
        original_sha = text_hash(original)
        known = index.execute("SELECT split FROM frozen_records WHERE source_id=? AND language='zh' AND text_sha=? AND split!='train'",
                              (source_id, original_sha)).fetchone()
        if known is None:
            return None
        converted = normalize_text(normalizer.convert(original, language="zh"))
        if normalizer.convert(converted, language="zh") != converted:
            raise ValueError("中文 t2s 转换未满足幂等性")
        digest, split = text_hash(converted), known[0]
        previous = index.execute("SELECT split FROM frozen_hashes WHERE text_sha=?", (digest,)).fetchone()
        if previous and previous[0] == "train":
            # 新预处理把旧 train 文本与旧 holdout 合并时，安全地升级为 holdout 保护。
            index.execute("UPDATE frozen_hashes SET split=? WHERE text_sha=?", (split, digest))
        else:
            protected = min((split, previous[0]), key=PRIORITY.__getitem__) if previous else split
            index.execute("INSERT INTO frozen_hashes VALUES (?,?) ON CONFLICT(text_sha) DO UPDATE SET split=excluded.split",
                          (digest, protected))
        updated = index.execute("UPDATE frozen_records SET converted=1 WHERE source_id=? AND language='zh' AND text_sha=? AND converted=0",
                                (source_id, original_sha)).rowcount
        if updated:
            line = json_line({"source_id": source_id, "language": "zh", "split": split, "text_sha256": digest,
                              "original_text_sha256": original_sha, "status": "frozen_holdout_normalized"})
            guard.before_write(len(line.encode("utf-8")))
            audit.write(line)
        return {"source_id": source_id, "language": "zh", "split": split, "text": converted, "text_sha256": digest}

    with (output / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as heldout:
        for row in read_jsonl(documents):
            split = split_name(row.get("split"))
            if split == "train":
                continue
            original = normalize_text(row.get("text"))
            source = index.execute("SELECT split FROM frozen_sources WHERE source_id=?", (row.get("source_id"),)).fetchone()
            isolation_ledger = frozen["manifest"].get("audit", {}).get("role") == "isolation_ledger"
            if source is None or source[0] == "train" or (source[0] != split and not isolation_ledger):
                raise ValueError("旧 emitted holdout 文本与冻结 source_id 分区不一致")
            if row.get("language") == "zh":
                new = protect(row)
                if new is None:
                    raise ValueError("旧 emitted 中文 holdout 文本未匹配原审计 hash")
                if old_tokenizer is not None:
                    ids = old_tokenizer.encode(original, add_special_tokens=False).ids + [frozen["manifest"]["tokenizer"]["eos_id"]]
                    parents[split]["sha256"].update(np.asarray(ids, dtype="<u4").tobytes())
                    parents[split]["token_count"] += len(ids)
            else:
                if row.get("language") != "en":
                    raise ValueError("旧 documents 只允许 en/zh")
                known = index.execute("SELECT 1 FROM frozen_records WHERE source_id=? AND language='en' AND text_sha=? AND split!='train'",
                                      (row["source_id"], text_hash(original))).fetchone()
                if known is None:
                    raise ValueError("旧 emitted 英文 holdout 文本未匹配原审计 hash")
                new = {"source_id": row["source_id"], "language": "en", "split": split,
                       "text": original, "text_sha256": text_hash(original)}
            existing = index.execute("SELECT split FROM emitted_holdout_hashes WHERE text_sha=?", (new["text_sha256"],)).fetchone()
            if existing and existing[0] != split:
                raise ValueError("转换后已冻结 validation/test 出现相同文档，不能改变源集合来静默修复")
            index.execute("INSERT OR IGNORE INTO emitted_holdout_hashes VALUES (?,?)", (new["text_sha256"], split))
            line = json_line(new)
            guard.before_write(len(line.encode("utf-8")))
            heldout.write(line)
    frozen["documents_path"] = str(documents)
    frozen["documents_sha256"] = sha256_file(documents)
    if frozen["rebuild_zh"]:
        for split, parent in parents.items():
            expected = frozen["manifest"]["splits"][split]["zh"]
            if parent["token_count"] != expected["token_count"] or parent["sha256"].hexdigest() != expected["sha256"]:
                raise ValueError("旧 documents 中文文本无法复现旧 holdout token SHA256")
        text_inputs = config.get("frozen_holdout_text_inputs", [])
        if not isinstance(text_inputs, list):
            raise ValueError("frozen_holdout_text_inputs 必须是路径列表")
        frozen["text_input_provenance"] = []
        total_bytes = 0
        for value in text_inputs:
            path = config_path(value, root)
            size = path.stat().st_size
            total_bytes += size
            if total_bytes > int(config.get("max_frozen_holdout_text_bytes", config["max_input_bytes"])):
                raise ValueError("旧 holdout 补全文本超过 max_frozen_holdout_text_bytes")
            frozen["text_input_provenance"].append({"path": str(path), "size_bytes": size, "sha256": sha256_file(path)})
            for number, row in enumerate(read_jsonl(path), 1):
                protect(row)
                if number % 1000 == 0:
                    index.commit()
                    guard.check()
                    _progress(output, "frozen_chinese_holdout", text_input=str(path), scanned=number)
        missing = index.execute("SELECT COUNT(*) FROM frozen_records WHERE language='zh' AND split!='train' AND converted=0").fetchone()[0]
        if missing:
            raise ValueError(f"{missing} 篇旧中文 holdout 缺少原始文本，无法保护转换后 hash；请提供 frozen_holdout_text_inputs")
        frozen["rebuilt_zh_holdout_parent"] = {"normalization": normalizer.provenance,
            "parent_token_streams": {split: frozen["manifest"]["splits"][split]["zh"] for split in ("validation", "test")},
            "note": "保留旧 emitted 源集合与顺序，统一转简体后重新编码；不能与旧繁简混合 validation loss 直接横比。"}
    index.commit()


def _quality_ok(text: str, quality: dict[str, Any]) -> bool:
    minimum, maximum = int(quality.get("min_chars", 80)), int(quality.get("max_chars", 100000))
    if minimum < 1 or maximum < minimum:
        raise ValueError("quality.min_chars/max_chars 非法")
    if not minimum <= len(text) <= maximum:
        return False
    if max(Counter(text).values()) / len(text) > float(quality.get("max_character_fraction", 0.5)):
        return False
    lines = [line for line in text.splitlines() if line]
    return len(set(lines)) / max(1, len(lines)) >= float(quality.get("min_unique_line_ratio", 0.5))


def _block(index: sqlite3.Connection, source_id: str, rank: int) -> None:
    index.execute("UPDATE sources SET blocked=1, protected_rank=MIN(protected_rank,?) WHERE source_id=?", (rank, source_id))


def _index_document(index: sqlite3.Connection, row: dict[str, Any], config: dict[str, Any], normalizer=None) -> tuple[dict[str, str], str]:
    source_id, language = row.get("source_id"), row.get("language")
    if not isinstance(source_id, str) or not source_id or language not in LANGUAGES:
        raise ValueError("输入需要 source_id 字符串及 language=en/zh")
    required_revision = config.get("required_source_revision")
    if required_revision is not None and row.get("source_revision") != required_revision:
        raise ValueError("source_revision 与 required_source_revision 不一致或缺失")
    text = normalize_text(row.get("text"))
    if normalizer is not None:
        text = normalize_text(normalizer.convert(text, language=language))
        if language == "zh" and normalizer.convert(text, language="zh") != text:
            raise ValueError("中文转换未满足幂等性")
    digest = text_hash(text)
    frozen_source = index.execute("SELECT split FROM frozen_sources WHERE source_id=?", (source_id,)).fetchone()
    explicit = row.get("split", row.get("partition"))
    split = assigned_split(
        source_id, int(config.get("seed", 42)), config.get("split_fractions", {"train": .98, "validation": .01, "test": .01}))
    if explicit is not None:
        explicit = split_name(explicit)
        if not frozen_source and explicit != split:
            raise ValueError("输入显式 split 与稳定 source_id 分区不一致，不能覆盖分区")
    if frozen_source:
        if explicit is not None and frozen_source[0] != explicit:
            raise ValueError("输入源 ID 的显式 split 与冻结审计不一致")
        split = frozen_source[0]
    normalized = {"source_id": source_id, "language": language, "split": split,
                  "text_sha256": digest, "text": text}
    previous = index.execute("SELECT split, blocked FROM sources WHERE source_id=?", (source_id,)).fetchone()
    if previous and previous[0] != split:
        raise ValueError(f"同源跨 split：{source_id}")
    index.execute("INSERT OR IGNORE INTO sources(source_id, split, protected_rank) VALUES (?, ?, ?)",
                  (source_id, split, PRIORITY[split] if split != "train" else 3))
    # 磁盘上的边同时记录重复别名；稍后才能与后出现的 holdout 传播整组保护。
    index.execute("INSERT OR IGNORE INTO source_hashes VALUES (?, ?)", (digest, source_id))
    frozen_hash = index.execute("SELECT split FROM frozen_hashes WHERE text_sha=?", (digest,)).fetchone()
    if (frozen_source and frozen_source[0] != "train") or (frozen_hash and frozen_hash[0] != "train"):
        ranks = [PRIORITY[item[0]] for item in (frozen_source, frozen_hash) if item and item[0] != "train"]
        _block(index, source_id, min(ranks))
        index.execute("INSERT OR IGNORE INTO contents VALUES (?, ?, ?)", (digest, split, source_id))
        return normalized, "excluded_frozen_holdout"
    if not _quality_ok(text, config.get("quality", {})):
        return normalized, "quality_filtered"
    old_record = index.execute("SELECT text_sha FROM records WHERE source_id=? AND language=?", (source_id, language)).fetchone()
    if old_record:
        if old_record[0] != digest:
            raise ValueError(f"同一 source_id/language 出现多个不同文档版本：{source_id}/{language}")
        return normalized, "duplicate_source_record"
    index.execute("INSERT INTO records VALUES (?, ?, ?)", (source_id, language, digest))
    previous_content = index.execute(
        "SELECT c.split, c.source_id, s.blocked, s.protected_rank FROM contents c JOIN sources s ON s.source_id=c.source_id WHERE text_sha=?",
        (digest,)).fetchone()
    if previous_content:
        old_split, owner, owner_blocked, owner_rank = previous_content
        if old_split != split and config.get("cross_split_duplicate_policy", "error") == "error":
            raise ValueError("相同规范化文档跨 split；请审计或显式选择 drop_conflicting_train")
        if owner_blocked and owner_rank < 2:
            _block(index, source_id, owner_rank)
            return normalized, "excluded_duplicate_blocked_group"
        if old_split == split:
            return normalized, "duplicate_content"
        if PRIORITY[old_split] <= PRIORITY[split]:
            _block(index, source_id, PRIORITY[old_split])
            return normalized, "excluded_cross_split_duplicate"
        _block(index, owner, PRIORITY[split])
        index.execute("UPDATE contents SET split=?, source_id=? WHERE text_sha=?", (split, source_id, digest))
    else:
        index.execute("INSERT INTO contents VALUES (?, ?, ?)", (digest, split, source_id))
    if config.get("frozen_holdout_manifest") and split != "train":
        return normalized, "reserved_new_holdout"
    return normalized, "accepted"


def _finalize_isolation(index: sqlite3.Connection, output: Path, guard: DiskGuard) -> dict[str, int]:
    """用磁盘联接传播 holdout 保护，消除 train 先出现所造成的顺序依赖。"""
    index.executescript("""
        CREATE TEMP TABLE hash_protection(text_sha TEXT PRIMARY KEY, rank INTEGER NOT NULL);
        CREATE TEMP TABLE source_protection(source_id TEXT PRIMARY KEY, rank INTEGER NOT NULL);
    """)
    passes = 0
    while True:
        _progress(output, "isolation_graph", completed_passes=passes)
        guard.check()
        # 每个 hash 聚合一次；不能为同一模板的 N 个别名展开 N×N 联接。
        index.execute("DELETE FROM hash_protection")
        index.execute("""INSERT INTO hash_protection
            SELECT edge.text_sha,MIN(source.protected_rank) FROM source_hashes edge
            JOIN sources source ON source.source_id=edge.source_id GROUP BY edge.text_sha""")
        guard.check()
        index.execute("DELETE FROM source_protection")
        index.execute("""INSERT INTO source_protection
            SELECT edge.source_id,MIN(protection.rank) FROM source_hashes edge
            JOIN hash_protection protection ON protection.text_sha=edge.text_sha GROUP BY edge.source_id""")
        changed = index.execute("""
            UPDATE sources AS target SET protected_rank=(
                SELECT rank FROM source_protection WHERE source_id=target.source_id
            ) WHERE protected_rank > (
                SELECT rank FROM source_protection WHERE source_id=target.source_id
            )
        """).rowcount
        index.commit()
        passes += 1
        if not changed:
            break
    index.executescript("DROP TABLE hash_protection; DROP TABLE source_protection;")
    index.execute("""UPDATE sources SET blocked=1 WHERE
        (split='train' AND protected_rank<2) OR (split='validation' AND protected_rank=0)""")
    index.commit()
    # 发布的是最终保护分区；未来再冻结此审计时，重复别名不会被误认为 train。
    audit = output / "document_audit.jsonl"
    temporary = output / "document_audit.final.jsonl"
    protected = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for number, row in enumerate(read_jsonl(audit), 1):
            if number % 1000 == 0:
                guard.check()
                _progress(output, "isolation_audit", rewritten_rows=number)
            source = index.execute("SELECT blocked,protected_rank FROM sources WHERE source_id=?", (row["source_id"],)).fetchone()
            if source and source[1] < 2:
                effective = ("test", "validation")[source[1]]
                if effective != row["split"]:
                    row["assigned_split"], row["split"] = row["split"], effective
                if source[0] and row.get("status") != "frozen_holdout":
                    row["status"] = "excluded_protected_source_group"
                    protected += 1
            line = json_line(row)
            guard.before_write(len(line.encode("utf-8")))
            handle.write(line)
    temporary.replace(audit)
    return {"protection_passes": passes, "protected_audit_rows": protected,
            "blocked_source_groups": index.execute("SELECT COUNT(*) FROM sources WHERE blocked=1").fetchone()[0]}


def _active(index: sqlite3.Connection, row: dict[str, Any]) -> bool:
    result = index.execute(
        "SELECT s.blocked, c.source_id FROM sources s JOIN contents c ON c.text_sha=? WHERE s.source_id=?",
        (row["text_sha256"], row["source_id"])).fetchone()
    return result is not None and result[0] == 0 and result[1] == row["source_id"]


def _index_inputs(config: dict[str, Any], paths: list[Path], output: Path, index: sqlite3.Connection,
                  audit, guard: DiskGuard, normalizer=None) -> dict[str, int]:
    limit = positive_int(config, "max_docs")
    clean_limit = positive_int(config, "max_clean_bytes")
    stats: Counter = Counter()
    with (output / "cleaned.jsonl").open("w", encoding="utf-8", newline="\n") as cleaned:
        for path in paths:
            for raw in islice(read_jsonl(path), limit - stats["scanned"]):
                row, status = _index_document(index, raw, config, normalizer)
                stats["scanned"] += 1
                stats[status] += 1
                if row["language"] == "zh" and normalizer is not None and normalizer.provenance["mode"] == "t2s":
                    stats["chinese_t2s_idempotence_checked"] += 1
                audit_line = json_line({key: row[key] for key in ("source_id", "language", "split", "text_sha256")} | {"status": status})
                guard.before_write(len(audit_line.encode("utf-8")))
                audit.write(audit_line)
                if status == "accepted":
                    line = json_line(row)
                    size = len(line.encode("utf-8"))
                    if stats["clean_bytes"] + size > clean_limit:
                        raise ValueError("cleaned.jsonl 超过 max_clean_bytes，未发布完整 manifest")
                    guard.before_write(size)
                    cleaned.write(line)
                    stats["clean_bytes"] += size
                if stats["scanned"] % 1000 == 0:
                    index.commit()
                    guard.check()
                    _progress(output, "indexing", scanned=stats["scanned"], clean_bytes=stats["clean_bytes"],
                              input_path=str(path), statistics=dict(stats))
            if stats["scanned"] >= limit:
                stats["stopped_max_docs"] = 1
                break
    index.commit()
    _progress(output, "indexing_complete", scanned=stats["scanned"], clean_bytes=stats["clean_bytes"], statistics=dict(stats))
    return dict(stats)


def _make_tokenizer(config: dict[str, Any], root: Path, output: Path, frozen: dict[str, Any] | None,
                    index: sqlite3.Connection, guard: DiskGuard) -> tuple[Any, dict[str, Any]]:
    spec = config.get("tokenizer", {})
    vocabulary = int(spec.get("vocab_size", 32000))
    destination = output / "tokenizer.json"
    if spec.get("path") or frozen:
        source = config_path(spec["path"], root) if spec.get("path") else ensure_under(
            Path(frozen["path"]).parent / frozen["manifest"]["tokenizer"]["path"], Path(frozen["path"]).parent)
        digest = sha256_file(source)
        expected = spec.get("sha256")
        if expected and digest != expected:
            raise ValueError("固定 tokenizer SHA256 不匹配")
        if frozen and digest != frozen["manifest"]["tokenizer"]["sha256"]:
            raise ValueError("tokenizer 与冻结 holdout 的 tokenizer SHA256 不一致")
        if not frozen and (not expected or spec.get("trained_on_split") != "train"):
            raise ValueError("无冻结 manifest 时复用 tokenizer 需显式 sha256 与 trained_on_split=train 来源声明")
        guard.check(reserve=source.stat().st_size)
        shutil.copyfile(source, destination)
        tokenizer = load_tokenizer(destination)
        if sha256_file(destination) != digest:
            raise ValueError("tokenizer 复制校验失败")
        provenance = {"mode": "reused_train_only", "source_path": str(source), "source_sha256": digest}
    else:
        sample_limit = positive_int(spec, "sample_max_bytes")
        per_language = sample_limit // 2
        if per_language < 1:
            raise ValueError("tokenizer.sample_max_bytes 至少为 2")
        samples = {lang: output / f"tokenizer_sample.{lang}.jsonl" for lang in LANGUAGES}
        counts = {lang: 0 for lang in LANGUAGES}
        with ExitStack() as stack:
            handles = {lang: stack.enter_context(path.open("w", encoding="utf-8", newline="\n")) for lang, path in samples.items()}
            for row in read_jsonl(output / "cleaned.jsonl"):
                if row["split"] != "train" or not _active(index, row):
                    continue
                language = row["language"]
                size = len(row["text"].encode("utf-8"))
                if counts[language] + size > per_language:
                    continue
                line = json_line(row)
                guard.before_write(len(line.encode("utf-8")))
                handles[language].write(line)
                counts[language] += size
        if any(value == 0 for value in counts.values()):
            raise ValueError("train-only tokenizer 样本需要中英均非空；请增加 sample_max_bytes 或训练文档")

        def texts():
            for language in LANGUAGES:
                for row in read_jsonl(samples[language]):
                    yield row["text"]

        tokenizer = train_bpe(texts(), destination, vocab_size=vocabulary, min_frequency=int(spec.get("min_frequency", 2)))
        provenance = {"mode": "bounded_train_only_sample", "text_bytes_by_language": counts,
                      "sample_max_bytes": sample_limit,
                      "samples": {lang: {"path": path.name, "sha256": sha256_file(path)} for lang, path in samples.items()}}
    if tokenizer.get_vocab_size() != vocabulary:
        raise ValueError(f"tokenizer 实际 vocab_size={tokenizer.get_vocab_size()} 与请求 {vocabulary} 不一致")
    eos = tokenizer.token_to_id(EOS_TOKEN)
    if frozen:
        old = frozen["manifest"]["tokenizer"]
        if eos is None:
            eos = int(old["eos_id"])
        if vocabulary != int(old["vocab_size"]) or eos != int(old["eos_id"]):
            raise ValueError("tokenizer vocab/EOS 与冻结 holdout 不一致")
    if eos is None or not 0 <= eos < vocabulary:
        raise ValueError("tokenizer 缺少合法 EOS")
    return tokenizer, {"path": destination.name, "sha256": sha256_file(destination), "vocab_size": vocabulary,
                       "eos_id": eos, "pad_id": tokenizer.token_to_id("<|pad|>"), "trained_on_split": "train", "provenance": provenance}


def _copy_holdouts(frozen: dict[str, Any], output: Path, guard: DiskGuard, tokenizer, tokenizer_info) -> dict[str, Any]:
    root = Path(frozen["path"]).parent
    result = {}
    for split in ("validation", "test"):
        result[split] = {}
        for language in LANGUAGES:
            item = frozen["manifest"]["splits"][split][language]
            source = ensure_under(root / item["path"], root)
            count = int(item["token_count"])
            if item["dtype"] != "uint32" or source.stat().st_size != count * 4 or sha256_file(source) != item.get("sha256"):
                raise ValueError(f"冻结 holdout 大小或 SHA256 不匹配：{split}/{language}")
            guard.check(reserve=source.stat().st_size)
            destination = output / f"{split}.{language}.bin"
            if language == "zh" and frozen.get("rebuild_zh"):
                count = 0
                with destination.open("wb") as handle:
                    for row in read_jsonl(output / "documents.jsonl"):
                        if row["split"] == split and row["language"] == "zh":
                            ids = tokenizer.encode(row["text"], add_special_tokens=False).ids + [tokenizer_info["eos_id"]]
                            data = np.asarray(ids, dtype="<u4").tobytes()
                            guard.before_write(len(data))
                            handle.write(data)
                            count += len(ids)
                digest = sha256_file(destination)
            else:
                shutil.copyfile(source, destination)
                digest = item["sha256"]
            if destination.is_symlink() or sha256_file(destination) != digest:
                raise ValueError("冻结 holdout 复制校验失败")
            result[split][language] = {"path": destination.name, "token_count": count, "dtype": "uint32",
                                       "byte_order": "little", "sha256": digest}
    return result


def _encoding_batch_limits(config: dict[str, Any]) -> tuple[int, int]:
    batch_limits = {"encoding_batch_size": config.get("encoding_batch_size", 64),
                    "encoding_batch_max_bytes": config.get("encoding_batch_max_bytes", 4 * 1024 * 1024)}
    batch_size = positive_int(batch_limits, "encoding_batch_size")
    max_batch_bytes = positive_int(batch_limits, "encoding_batch_max_bytes")
    if batch_size > 1024:
        raise ValueError("encoding_batch_size 不能超过 1024")
    return batch_size, max_batch_bytes


def _encode(config: dict[str, Any], output: Path, tokenizer, tokenizer_info: dict[str, Any],
            index: sqlite3.Connection, frozen: dict[str, Any] | None, guard: DiskGuard) -> tuple[dict[str, Any], dict[str, int]]:
    train_caps = config.get("max_train_tokens_per_language", {})
    for language in LANGUAGES:
        positive_int(train_caps, language)
    batch_size, max_batch_bytes = _encoding_batch_limits(config)
    counts = {split: {lang: 0 for lang in LANGUAGES} for split in SPLITS}
    splits = _copy_holdouts(frozen, output, guard, tokenizer, tokenizer_info) if frozen else {}
    encoded_splits = ("train",) if frozen else SPLITS
    stats: Counter = Counter()
    with ExitStack() as stack:
        handles = {(split, lang): stack.enter_context((output / f"{split}.{lang}.bin").open("wb"))
                   for split in encoded_splits for lang in LANGUAGES}
        heldout = None if frozen else stack.enter_context((output / "documents.jsonl").open("w", encoding="utf-8", newline="\n"))
        batch: list[dict[str, str]] = []
        batch_bytes = 0

        def flush_batch() -> None:
            nonlocal batch_bytes
            if not batch:
                return
            encodings = tokenizer.encode_batch([row["text"] for row in batch], add_special_tokens=False)
            if len(encodings) != len(batch):
                raise ValueError("Tokenizer 批编码返回文档数不一致")
            # 批量只改变分词调度；顺序、逐文 EOS 和整篇容量判断仍按原规则执行。
            for row, encoding in zip(batch, encodings):
                split, language = row["split"], row["language"]
                cap = train_caps[language] if split == "train" else int(config.get("max_eval_tokens_per_language", {}).get(split, 400000))
                ids = encoding.ids
                ids.append(tokenizer_info["eos_id"])
                if len(ids) + counts[split][language] > cap:
                    stats[f"over_token_cap_{split}_{language}"] += 1
                    continue
                data = np.asarray(ids, dtype="<u4").tobytes()
                guard.before_write(len(data))
                handles[(split, language)].write(data)
                if heldout is not None and split != "train":
                    line = json_line(row)
                    guard.before_write(len(line.encode("utf-8")))
                    heldout.write(line)
                counts[split][language] += len(ids)
                stats[f"documents_{split}_{language}"] += 1
            batch.clear()
            batch_bytes = 0

        for number, row in enumerate(read_jsonl(output / "cleaned.jsonl"), 1):
            if number % 1000 == 0:
                flush_batch()
                guard.check()
                _progress(output, "encoding", scanned_cleaned=number, token_counts=counts, statistics=dict(stats))
            if not _active(index, row):
                stats["excluded_after_group_audit"] += 1
                continue
            split, language = row["split"], row["language"]
            if split not in encoded_splits:
                continue
            text_bytes = len(row["text"].encode("utf-8"))
            if batch and batch_bytes + text_bytes > max_batch_bytes:
                flush_batch()
            batch.append(row)
            batch_bytes += text_bytes
            # 单篇可超过字节阈值，但必须独占批次；原有文档质量/长度限制不变。
            if len(batch) >= batch_size or batch_bytes >= max_batch_bytes:
                flush_batch()
        flush_batch()
    for split in encoded_splits:
        splits[split] = {}
        for language in LANGUAGES:
            path = output / f"{split}.{language}.bin"
            splits[split][language] = {"path": path.name, "token_count": counts[split][language], "dtype": "uint32",
                                       "byte_order": "little", "sha256": sha256_file(path)}
    minimum_stream = int(config.get("min_stream_tokens", 1025))
    _progress(output, "capacity_validation", token_counts={split: {language: item["token_count"] for language, item in stream.items()}
              for split, stream in splits.items()}, statistics=dict(stats))
    if minimum_stream < 1:
        raise ValueError("min_stream_tokens 必须为正")
    for split in SPLITS:
        for language in LANGUAGES:
            minimum = int(config.get("min_train_tokens_per_language", {}).get(language, minimum_stream)) if split == "train" else minimum_stream
            if minimum < 1 or splits[split][language]["token_count"] < minimum:
                raise ValueError(f"容量验收未通过：{split}/{language}={splits[split][language]['token_count']} tokens，至少需要 {minimum}；不发布 manifest")
    return splits, dict(stats)


def prepare_streaming(config: dict[str, Any], root: str | Path | None = None) -> Path:
    from .text_normalization import build_chinese_normalizer

    root = Path(root or Path.cwd()).resolve()
    _encoding_batch_limits(config)
    normalizer = build_chinese_normalizer(config.get("chinese_normalization", "none"))
    if config.get("cross_split_duplicate_policy", "error") not in ("error", "drop_conflicting_train"):
        raise ValueError("cross_split_duplicate_policy 只能为 error/drop_conflicting_train")
    required_revision = config.get("required_source_revision")
    if required_revision is not None and (not isinstance(required_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", required_revision)):
        raise ValueError("required_source_revision 必须为固定 40 位 SHA")
    for language in LANGUAGES:
        cap = positive_int(config.get("max_train_tokens_per_language", {}), language)
        minimum = config.get("min_train_tokens_per_language", {}).get(language, config.get("min_stream_tokens", 1025))
        if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= cap:
            raise ValueError("min_train_tokens_per_language 必须为正且不超过对应 max_train_tokens_per_language")
    paths = resolve_inputs(config, root)
    max_input = positive_int(config, "max_input_bytes")
    if sum(path.stat().st_size for path in paths) > max_input:
        raise ValueError("输入文件总大小超过 max_input_bytes")
    output = config_path(config["output_dir"], root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空；此版本不支持从未完成目录恢复或覆盖，请审计后使用新目录")
    guard = DiskGuard(output, int(config.get("min_free_disk_bytes", 1024**3)))
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "preparation_state.json"
    _atomic_json(state_path, {"schema_version": 1, "status": "running", "phase": "input_fingerprints", "resume_supported": False})
    connection = None
    try:
        inputs = []
        for path in paths:
            stat = path.stat()
            inputs.append({"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "sha256": sha256_file(path)})
            _progress(output, "input_fingerprints", completed_files=len(inputs), total_files=len(paths),
                      completed_bytes=sum(item["size_bytes"] for item in inputs))
        acquisition = _check_acquisition(config, root, inputs)
        effective_config = dict(config)
        if acquisition:
            effective_config["required_source_revision"] = acquisition["revision"]
        connection = _open_index(output / "prepare_index.sqlite")
        with (output / "document_audit.jsonl").open("w", encoding="utf-8", newline="\n") as audit:
            frozen = _load_frozen(config, root, connection, audit, guard)
            _prepare_frozen_texts(config, root, output, connection, frozen, normalizer, audit, guard)
            _atomic_json(state_path, {"schema_version": 1, "status": "running", "phase": "indexing", "resume_supported": False})
            statistics = _index_inputs(effective_config, paths, output, connection, audit, guard, normalizer)
        statistics.update(_finalize_isolation(connection, output, guard))
        tokenizer, tokenizer_info = _make_tokenizer(config, root, output, frozen, connection, guard)
        _atomic_json(state_path, {"schema_version": 1, "status": "running", "phase": "encoding", "resume_supported": False})
        splits, encoding = _encode(config, output, tokenizer, tokenizer_info, connection, frozen, guard)
        for item in inputs:
            stat = Path(item["path"]).stat()
            if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError("制备期间输入文件被修改；不发布 manifest")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        connection = None
        manifest = {"schema_version": 1, "kind": "pretrain_token_streams", "preparation": "disk_indexed_streaming_v1",
                    "tokenizer": tokenizer_info, "splits": splits, "config": config, "input_provenance": inputs,
                    "acquisition_provenance": acquisition,
                    "chinese_normalization": normalizer.provenance,
                    "statistics": {"cleaning": statistics, "encoding": encoding},
                    "token_count": sum(item["token_count"] for stream in splits.values() for item in stream.values()),
                    "sampling": {"unit": "effective_tokens", "language_probabilities": {"en": .5, "zh": .5},
                                 "note": "这是语料容量；训练采样是否重复由训练器决定。"},
                    "audit": {"path": "document_audit.jsonl", "sha256": sha256_file(output / "document_audit.jsonl"),
                              "role": "isolation_ledger", "note": "split 包含保守保护分区；弃用的新 holdout 候选不加入冻结 bin。"},
                    "cleaned": {"path": "cleaned.jsonl", "sha256": sha256_file(output / "cleaned.jsonl"),
                                "note": "候选清洗文本；最终源组隔离由磁盘 index 和 document_audit 决定，不能独立作训练集。"},
                    "index": {"path": "prepare_index.sqlite", "sha256": sha256_file(output / "prepare_index.sqlite")},
                    "frozen_holdout": None if frozen is None else {key: frozen[key] for key in ("path", "sha256", "audit_path", "audit_sha256")}}
        if (output / "documents.jsonl").is_file():
            manifest["documents"] = {"path": "documents.jsonl", "sha256": sha256_file(output / "documents.jsonl"),
                                     "role": "emitted_holdout_texts"}
        if frozen and frozen.get("rebuilt_zh_holdout_parent"):
            manifest["rebuilt_zh_holdout_parent"] = frozen["rebuilt_zh_holdout_parent"]
            manifest["frozen_holdout"]["text_input_provenance"] = frozen.get("text_input_provenance", [])
            manifest["frozen_holdout"]["documents_path"] = frozen["documents_path"]
            manifest["frozen_holdout"]["documents_sha256"] = frozen["documents_sha256"]
        guard.check()
        _atomic_json(state_path, {"schema_version": 1, "status": "complete", "phase": "complete", "resume_supported": False,
                                  "note": "只以原子发布的 manifest.json 作为语料完成凭证。"})
        # manifest 是最后一个写入动作，容量不足、输入变化或失败目录均没有此完成凭证。
        _atomic_json(output / "manifest.json", manifest)
        return output / "manifest.json"
    except BaseException as exc:
        if connection is not None:
            connection.close()
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
            _atomic_json(state_path, {**previous, "schema_version": 1, "status": "failed", "resume_supported": False,
                                     "error": str(exc), "note": "部分文件不是可用语料；审计后用新输出目录重新制备。"})
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="磁盘索引、冻结 holdout 与有界 BPE 的顺序数据制备")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    print(prepare_streaming(config, root))


if __name__ == "__main__":
    main()
