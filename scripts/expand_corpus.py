"""固定网页 Parquet 的 CPU 增量编码；保留原 tokenizer/holdout，不修改旧语料。

SQLite 的逐分片 commit 是恢复位置的唯一权威：分片先写 .part 并 fsync，
追加主文件并 fsync 后再提交索引；崩溃恢复会截断所有未提交的尾部。
不落地巨大 raw/cleaned JSONL，也不在模块导入时联网或初始化 CUDA。
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from llm_lab.data.common import config_path, ensure_under, json_line, load_config, normalize_text, read_jsonl, sha256_file, text_hash
from llm_lab.data.text_normalization import build_chinese_normalizer
from llm_lab.tokenization.bpe import load_tokenizer

LANGUAGES = ("en", "zh")
WIKI_DOMAINS = ("wikipedia.org", "wikimedia.org", "wikibooks.org", "wikisource.org",
                "wiktionary.org", "wikiquote.org", "wikiversity.org", "wikivoyage.org", "wikidata.org")
MASK = (1 << 64) - 1
ROLLING_BASE = 1000003


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def disk_room(path: Path, reserve: int, incoming: int = 0) -> None:
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < reserve + incoming:
        raise OSError("剩余磁盘不足，必须保留 min_free_bytes；当前分片不会提交")


@contextmanager
def writer_lock(output: Path):
    lock = output / "writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"已有 writer.lock，请先核实原 PID 是否停止：{lock}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_unix": time.time()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def canonical_url(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 8192:
        raise ValueError("无有效 URL")
    parts = urlsplit(value)
    host = (parts.hostname or "").rstrip(".").lower()
    if parts.scheme.lower() not in {"http", "https"} or not host or parts.username or parts.password:
        raise ValueError("URL 必须为无认证信息的 HTTP(S) 地址")
    if any(host == suffix or host.endswith("." + suffix) for suffix in WIKI_DOMAINS):
        raise ValueError("excluded_wikimedia_domain")
    port = parts.port
    authority = host if port in {None, 80, 443} else f"{host}:{port}"
    # 合并 HTTP/HTTPS 和 fragment 变体；保留 query，避免合并真正不同的页面。
    return urlunsplit(("https", authority, parts.path or "/", parts.query, ""))


def fingerprint_text(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())


def window_hashes(text: str, width: int = 100) -> Iterator[int]:
    """每字符滑动的 64-bit 多项式指纹，捕获任意偏移的 >=100 字符旧片段。"""
    if len(text) < width:
        return
    power = pow(ROLLING_BASE, width - 1, 1 << 64)
    value = 0
    for character in text[:width]:
        value = (value * ROLLING_BASE + ord(character) + 1) & MASK
    yield value
    for offset in range(width, len(text)):
        value = ((value - (ord(text[offset - width]) + 1) * power) * ROLLING_BASE + ord(text[offset]) + 1) & MASK
        yield value


def quality_ok(text: str, quality: dict[str, Any]) -> bool:
    if not int(quality["min_chars"]) <= len(text) <= int(quality["max_chars"]):
        return False
    if max(Counter(text).values()) / len(text) > float(quality["max_character_fraction"]):
        return False
    lines = [line for line in text.splitlines() if line]
    return len(set(lines)) / max(1, len(lines)) >= float(quality["min_unique_line_ratio"])


def validate_config(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = Path(root or Path.cwd()).resolve()
    settings = json.loads(json.dumps(config))
    for key, default in (("cache_dir", ".cache/data/expansion_2b"), ("output_dir", None), ("parent_manifest", None)):
        value = settings.get(key, default)
        if not isinstance(value, str) or not value:
            raise ValueError(f"必须设置 {key}")
        settings[key] = str(config_path(value, root))
    output, cache = Path(settings["output_dir"]), Path(settings["cache_dir"])
    parent_directory = Path(settings["parent_manifest"]).parent
    for path in (output, cache, Path(settings["parent_manifest"])):
        ensure_under(path, root)
    if output == root or cache == root:
        raise ValueError("不能把项目根目录作为增量输出/缓存")
    if output == parent_directory or output.is_relative_to(parent_directory) or parent_directory.is_relative_to(output):
        raise ValueError("增量输出不能覆盖、包含父语料目录或位于其内部")
    if cache == parent_directory or cache.is_relative_to(parent_directory) or parent_directory.is_relative_to(cache):
        raise ValueError("专属下载缓存不能覆盖或包含父语料")
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("output_dir 和 cache_dir 不能互相包含")
    settings.setdefault("target_tokens_per_language", {lang: 1_000_000_000 for lang in LANGUAGES})
    for lang in LANGUAGES:
        value = settings["target_tokens_per_language"].get(lang)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("目标 Token 数必须为正整数")
    settings.setdefault("min_free_bytes", 12 * 1024**3)
    settings.setdefault("batch_size", 64)
    settings.setdefault("progress_seconds", 5)
    settings.setdefault("delete_committed_cache", True)
    settings.setdefault("prepend_parent_train", True)
    settings.setdefault("fingerprint_chars", 100)
    settings.setdefault("chinese_normalization", "t2s")
    if settings["chinese_normalization"] != "t2s":
        raise ValueError("增量中文必须使用官方 OpenCC t2s")
    if not 1 <= int(settings["batch_size"]) <= 128 or int(settings["fingerprint_chars"]) < 100:
        raise ValueError("batch_size 必须在 1..128，holdout 指纹至少 100 字符")
    quality = {"min_chars": 80, "max_chars": 100000, "max_character_fraction": .5, "min_unique_line_ratio": .5}
    quality.update(settings.get("quality", {}))
    if not 1 <= int(quality["min_chars"]) <= int(quality["max_chars"]) <= 100000:
        raise ValueError("quality 字符长度范围非法或超过单篇 100000 上限")
    if any(not 0 < float(quality[key]) <= 1 for key in ("max_character_fraction", "min_unique_line_ratio")):
        raise ValueError("quality 比例必须在 (0,1]")
    settings["quality"] = quality
    sources = settings.get("sources")
    if not isinstance(sources, list) or len(sources) != 2 or {x.get("language") for x in sources} != set(LANGUAGES):
        raise ValueError("sources 必须恰好包含一份 en 和一份 zh")
    for source in sources:
        if not isinstance(source.get("repository"), str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", source["repository"]):
            raise ValueError("repository 必须是 owner/name")
        if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", ""))):
            raise ValueError("revision 必须固定为 40 位 commit SHA")
        if not isinstance(source.get("data_files"), list) or not source["data_files"]:
            raise ValueError("必须显式提供有限 data_files")
        for item in source["data_files"]:
            item["size"] = item.get("size", item.get("size_bytes"))
            if not isinstance(item["size"], int) or item["size"] < 1 or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                raise ValueError("每片必须有正 size 和 SHA256")
            path = item.get("path", "")
            if not isinstance(path, str) or not path.endswith(".parquet") or Path(path).is_absolute() or ".." in path.split("/"):
                raise ValueError("分片 path 必须为仓库内相对 .parquet 路径")
    return settings


def open_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA cache_size=-32768")
    connection.executescript("""
      CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS seen(kind TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(kind,value)) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS commits(sequence INTEGER PRIMARY KEY,source_json TEXT NOT NULL,counts_json TEXT NOT NULL,
        stats_json TEXT NOT NULL,audit_bytes INTEGER NOT NULL,documents INTEGER NOT NULL);
    """)
    return connection


def verified_parent(settings: dict[str, Any]):
    path = Path(settings["parent_manifest"])
    parent = json.loads(path.read_text(encoding="utf-8"))
    if parent.get("schema_version") != 1 or parent["tokenizer"].get("trained_on_split") != "train":
        raise ValueError("父语料必须为 schema 1 和 train-only tokenizer")
    paths = {}
    for name in ("tokenizer", "audit", "documents"):
        item = parent.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"父 manifest 必须包含可验证的 {name}")
        target = ensure_under(path.parent / item["path"], path.parent)
        if sha256_file(target) != item["sha256"]:
            raise ValueError(f"父 {name} SHA256 不匹配")
        paths[name] = target
    return parent, paths


def _seen(connection, keys) -> bool:
    return any(connection.execute("SELECT 1 FROM seen WHERE kind=? AND value=?", key).fetchone() for key in keys)


def seed_parent(connection, settings, paths, normalizer, progress):
    if connection.execute("SELECT value FROM meta WHERE key='seed_complete'").fetchone():
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        for number, row in enumerate(read_jsonl(paths["audit"]), 1):
            keys = [("source_id", row["source_id"]), ("hash", row["text_sha256"])]
            if row.get("original_text_sha256"):
                keys.append(("hash", row["original_text_sha256"]))
            connection.executemany("INSERT OR IGNORE INTO seen VALUES (?,?)", keys)
            if number % 100000 == 0:
                progress("seeding_parent", old_audit_rows=number, force=True)
        connection.execute("INSERT INTO meta VALUES ('seed_complete',?)", (str(number if 'number' in locals() else 0),))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def holdout_fingerprints(settings, parent, paths, normalizer):
    fingerprints = set()
    count = 0
    for row in read_jsonl(paths["documents"]):
        if row.get("split") not in {"validation", "test"}:
            continue
        text = normalize_text(normalizer.convert(normalize_text(row["text"]), row["language"]))
        if row.get("text_sha256") and text_hash(text) != row["text_sha256"]:
            raise ValueError("父 holdout 文本与审计 SHA256 不一致")
        fingerprints.update(window_hashes(fingerprint_text(text), int(settings["fingerprint_chars"])))
        count += 1
    if not count:
        raise ValueError("没有读到冻结 holdout 正文，拒绝关闭镜像隔离")
    return fingerprints, count


def parquet_batches(path: Path, batch_size: int):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    required = {"text", "id", "url"}
    if not required.issubset(parquet.schema.names):
        raise ValueError("Parquet 必须包含 text/id/url")
    columns = [key for key in ("text", "id", "url", "dump", "date", "file_path") if key in parquet.schema.names]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        yield batch.to_pylist()


def recover_outputs(output: Path, connection, base_counts=None, base_paths=None, reserve=0):
    base_counts = base_counts or {lang: 0 for lang in LANGUAGES}
    last = connection.execute("SELECT sequence,counts_json,stats_json,audit_bytes,documents FROM commits ORDER BY sequence DESC LIMIT 1").fetchone()
    counts = json.loads(last[1]) if last else {lang: 0 for lang in LANGUAGES}
    audit_bytes = last[3] if last else 0
    for lang in LANGUAGES:
        path = output / f"train.{lang}.bin"
        if not path.exists() and base_counts[lang]:
            if last:
                raise ValueError("已提交的 Token 文件丢失，不能以旧前缀覆盖")
            disk_room(output, reserve, base_counts[lang] * 4)
            temporary = output / f"base-{lang}.part"
            with base_paths[lang].open("rb") as reader, temporary.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, path)
    for name, expected in [(f"train.{lang}.bin", (base_counts[lang] + counts[lang]) * 4) for lang in LANGUAGES] + [("incremental_audit.jsonl", audit_bytes)]:
        path = output / name
        if path.is_symlink():
            raise ValueError("增量输出不能是符号链接")
        if path.exists() and path.stat().st_size < expected:
            raise ValueError(f"已提交输出被截短：{name}")
        with path.open("a+b") as handle:
            handle.truncate(expected)
            handle.flush()
            os.fsync(handle.fileno())
    for part in output.glob("shard-*.part"):
        if part.is_symlink() or not part.is_file():
            raise ValueError("发现非普通分片临时文件")
        part.unlink()
    return {"sequence": last[0] if last else -1, "counts": counts, "stats": json.loads(last[2]) if last else {},
            "audit_bytes": audit_bytes, "documents": last[4] if last else 0}


def safe_delete_cache(path: Path, cache: Path) -> bool:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(cache.resolve()):
        return False
    path.unlink()
    return True


def _append_fsynced(source: Path, destination: Path):
    with source.open("rb") as reader, destination.open("ab") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _crash_hook(settings, phase):
    """仅供 fixture 注入崩溃点；生产 JSON 不暴露此 hook。"""
    callback = settings.get("_failure_hook")
    if callback:
        callback(phase)


def run_expansion(config: dict[str, Any], root: str | Path | None = None, *, resume: bool = False,
                  fetch_shard: Callable | None = None, batch_reader: Callable | None = None,
                  _failure_hook: Callable | None = None) -> Path:
    root = Path(root or Path.cwd()).resolve()
    settings = validate_config(config, root)
    output, cache = Path(settings["output_dir"]), Path(settings["cache_dir"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("输出非空；继续已有任务请显式 --resume")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    # 先拿 writer 锁，任何恢复截断/索引写入都必须位于单 writer 边界内。
    with writer_lock(output):
        parent, parent_paths = verified_parent(settings)
        parent_sha = sha256_file(Path(settings["parent_manifest"]))
        normalizer = build_chinese_normalizer("t2s")
        if normalizer.provenance != parent.get("chinese_normalization"):
            raise ValueError("OpenCC 完整 provenance 与父语料不同，不能改变清洗/去重语义")
        downloader = Path(__file__).with_name("corpus_download.py")
        pipeline = {"worker_sha256": sha256_file(Path(__file__)),
                    "downloader_sha256": sha256_file(downloader) if downloader.is_file() else None,
                    "data_common_sha256": sha256_file(Path(sys.modules[normalize_text.__module__].__file__)),
                    "normalizer_module_sha256": sha256_file(Path(sys.modules[build_chinese_normalizer.__module__].__file__)),
                    "python_version": sys.version, "unicode_version": unicodedata.unidata_version,
                    "chinese_normalization": normalizer.provenance}
        identity_settings = {key: value for key, value in settings.items() if key not in {"project_root", "progress_seconds"}}
        identity = text_hash(json.dumps({"config": identity_settings, "parent_sha256": parent_sha, "pipeline": pipeline}, sort_keys=True, separators=(",", ":")))
        completed = output / "expansion_manifest.json"
        if completed.is_file():
            manifest = json.loads(completed.read_text(encoding="utf-8"))
            if manifest.get("identity") != identity or any(manifest.get("new_tokens_per_language", {}).get(lang, 0) < settings["target_tokens_per_language"][lang] for lang in LANGUAGES):
                raise ValueError("已有 manifest 与当前配置/目标身份不符")
            for split in manifest["splits"].values():
                for stream in split.values():
                    path = ensure_under(output / stream["path"], output)
                    if path.stat().st_size != stream["token_count"] * 4 or sha256_file(path) != stream["sha256"]:
                        raise ValueError("已发布语料完整性失败，拒绝重新覆盖")
            audit = ensure_under(output / manifest["audit"]["path"], output)
            if sha256_file(audit) != manifest["audit"]["sha256"] or manifest["parent_manifest"]["sha256"] != parent_sha:
                raise ValueError("已发布增量审计/父语料指纹失败")
            return completed
        connection = open_index(output / "dedup.sqlite")
        old_identity = connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if old_identity and old_identity[0] != identity:
            connection.close()
            raise ValueError("恢复配置/父语料身份已改变，拒绝混写")
        connection.execute("INSERT OR IGNORE INTO meta VALUES ('identity',?)", (identity,))
        connection.commit()
        base_counts = {lang: 0 for lang in LANGUAGES}
        base_paths = {}
        if settings["prepend_parent_train"]:
            for lang in LANGUAGES:
                item = parent["splits"]["train"][lang]
                path = ensure_under(Path(settings["parent_manifest"]).parent / item["path"], Path(settings["parent_manifest"]).parent)
                if path.stat().st_size != item["token_count"] * 4 or sha256_file(path) != item["sha256"]:
                    connection.close()
                    raise ValueError("父 train 前缀大小/SHA256 校验失败")
                base_counts[lang] = item["token_count"]
                base_paths[lang] = path
        try:
            runtime = recover_outputs(output, connection, base_counts, base_paths, int(settings["min_free_bytes"]))
        except BaseException:
            connection.close()
            raise
        last_progress = [0.0]

        def progress(phase, force=False, **extra):
            now = time.monotonic()
            if not force and now - last_progress[0] < float(settings["progress_seconds"]):
                return
            disk_room(output, int(settings["min_free_bytes"]))
            value = {"status": "running", "phase": phase, "updated_unix": time.time(), "pid": os.getpid(),
                     "committed_sequence": runtime["sequence"], "committed_tokens": runtime["counts"],
                     "base_tokens": base_counts, "total_committed_tokens": {lang: base_counts[lang] + runtime["counts"][lang] for lang in LANGUAGES},
                     "target_tokens_per_language": settings["target_tokens_per_language"],
                     "committed_documents": runtime["documents"], "disk_free_bytes": shutil.disk_usage(output).free,
                     **extra}
            atomic_json(output / "progress.json", value)
            print(json.dumps(value, ensure_ascii=False), flush=True)
            last_progress[0] = now

        settings["_failure_hook"] = _failure_hook
        try:
            os.environ.setdefault("RAYON_NUM_THREADS", "4")
            progress("verifying_parent", force=True)
            seed_parent(connection, settings, parent_paths, normalizer, progress)
            progress("indexing_holdout_fragments", force=True)
            heldout, holdout_documents = holdout_fingerprints(settings, parent, parent_paths, normalizer)
            tokenizer = load_tokenizer(parent_paths["tokenizer"])
            info = parent["tokenizer"]
            if tokenizer.get_vocab_size() != info["vocab_size"] or not 0 <= info["eos_id"] < info["vocab_size"]:
                raise ValueError("父 tokenizer 词表/EOS 无效")
            if tokenizer.token_to_id("<|eos|>") != info["eos_id"]:
                raise ValueError("父 EOS token ID 与 tokenizer 不符")
            if fetch_shard is None:
                try:
                    from corpus_download import fetch_shard
                except ImportError:
                    def fetch_shard(source, item, settings, root):
                        return config_path(item["local_path"], root) if item.get("local_path") else Path(settings["cache_dir"]) / f"{item['sha256']}.parquet"
            reader = batch_reader or parquet_batches
            tasks = [(source, item) for source in settings["sources"] for item in source["data_files"]]
            for sequence, (source, item) in enumerate(tasks):
                if sequence <= runtime["sequence"]:
                    continue
                lang = source["language"]
                if runtime["counts"][lang] >= settings["target_tokens_per_language"][lang]:
                    continue
                progress("fetching_shard", shard=item["path"], language=lang, force=True)
                path = Path(fetch_shard(source, item, settings, root))
                if not path.is_file() or path.is_symlink() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
                    raise ValueError("原始 Parquet 大小/SHA256 校验失败")
                initial_stat = path.stat()
                part_bin = output / f"shard-{sequence}.{lang}.part"
                part_audit = output / f"shard-{sequence}.audit.part"
                counts = dict(runtime["counts"])
                stats = Counter(runtime["stats"])
                documents = runtime["documents"]
                row_number = 0
                connection.execute("BEGIN IMMEDIATE")
                try:
                    with part_bin.open("wb") as binary, part_audit.open("wb") as audit:
                        for batch in reader(path, int(settings["batch_size"])):
                            pending = []
                            pending_keys = set()
                            records = []
                            for row in batch:
                                row_number += 1
                                stats["rows_seen"] += 1
                                identifier, url = row.get("id"), row.get("url")
                                record = {"repository": source["repository"], "revision": source["revision"],
                                          "shard": item["path"], "row": row_number - 1, "language": lang,
                                          "split": "train", "id": identifier, "url": url,
                                          "token_count": 0, "token_offset": None, "text_sha256": None}
                                records.append(record)
                                try:
                                    canonical = canonical_url(url)
                                except (ValueError, TypeError):
                                    record["status"] = "excluded_wikimedia_domain" if isinstance(url, str) and any(suffix in url for suffix in WIKI_DOMAINS) else "invalid_url"
                                    continue
                                if not isinstance(identifier, str) or not identifier or len(identifier) > 8192:
                                    record["status"] = "invalid_id"
                                    continue
                                if not isinstance(row.get("text"), str) or len(row["text"]) > int(settings["quality"]["max_chars"]) * 2:
                                    record["status"] = "invalid_or_oversized_text"
                                    continue
                                original = normalize_text(row["text"])
                                text = normalize_text(normalizer.convert(original, lang))
                                if lang == "zh" and normalize_text(normalizer.convert(text, lang)) != text:
                                    raise ValueError("中文 t2s 结果未达到幂等固定点")
                                digest, original_digest = text_hash(text), text_hash(original)
                                record.update(source_id="web:" + text_hash(canonical), canonical_url=canonical,
                                              text_sha256=digest, original_text_sha256=original_digest)
                                if not quality_ok(text, settings["quality"]):
                                    record["status"] = "quality_filtered"
                                    continue
                                keys = [("url", canonical), ("source_id", identifier),
                                        ("source_id", record["source_id"]),
                                        ("source_id", source["repository"] + ":" + identifier),
                                        ("hash", digest), ("hash", original_digest)]
                                if _seen(connection, keys) or any(key in pending_keys for key in keys):
                                    record["status"] = "duplicate_old_or_new"
                                    continue
                                if heldout and any(value in heldout for value in window_hashes(fingerprint_text(text), int(settings["fingerprint_chars"]))):
                                    record["status"] = "heldout_fragment_match"
                                    continue
                                pending_keys.update(keys)
                                pending.append((record, text, keys))
                            encoded = tokenizer.encode_batch([candidate[1] for candidate in pending], add_special_tokens=False) if pending else []
                            for (record, text, keys), result in zip(pending, encoded):
                                ids = result.ids + [info["eos_id"]]
                                if not ids or min(ids) < 0 or max(ids) >= info["vocab_size"]:
                                    raise ValueError("编码产生词表范围之外的 Token")
                                # 保留整篇文档；最多超出目标一篇，绝不截断或人为填充。
                                if counts[lang] >= settings["target_tokens_per_language"][lang]:
                                    record["status"] = "target_already_reached"
                                    continue
                                disk_room(output, int(settings["min_free_bytes"]), len(ids) * 4 + 16384)
                                binary.write(np.asarray(ids, dtype="<u4").tobytes())
                                record.update(status="encoded", token_count=len(ids), token_offset=base_counts[lang] + counts[lang], new_token_offset=counts[lang])
                                counts[lang] += len(ids)
                                documents += 1
                                connection.executemany("INSERT OR IGNORE INTO seen VALUES (?,?)", keys)
                            for record in records:
                                stats[record["status"]] += 1
                                audit.write(json_line(record).encode("utf-8"))
                            progress("encoding_shard", shard=item["path"], language=lang, row=row_number,
                                     tentative_tokens=counts, tentative_statistics=dict(stats))
                            if counts[lang] >= settings["target_tokens_per_language"][lang]:
                                break
                        for handle in (binary, audit):
                            handle.flush()
                            os.fsync(handle.fileno())
                    stat = path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != (initial_stat.st_size, initial_stat.st_mtime_ns):
                        raise ValueError("处理期间原始 Parquet 被修改")
                    _crash_hook(settings, "parts_fsynced")
                    disk_room(output, int(settings["min_free_bytes"]), part_bin.stat().st_size + part_audit.stat().st_size)
                    _append_fsynced(part_bin, output / f"train.{lang}.bin")
                    _append_fsynced(part_audit, output / "incremental_audit.jsonl")
                    _crash_hook(settings, "outputs_fsynced")
                    provenance = {"repository": source["repository"], "revision": source["revision"], "language": lang,
                                  "path": item["path"], "sha256": item["sha256"], "size_bytes": item["size"], "rows_scanned": row_number}
                    audit_bytes = (output / "incremental_audit.jsonl").stat().st_size
                    connection.execute("INSERT INTO commits VALUES (?,?,?,?,?,?)",
                                       (sequence, json.dumps(provenance), json.dumps(counts), json.dumps(dict(stats)), audit_bytes, documents))
                    connection.commit()
                    runtime.update(sequence=sequence, counts=counts, stats=dict(stats), audit_bytes=audit_bytes, documents=documents)
                    _crash_hook(settings, "sqlite_committed")
                    atomic_json(output / "checkpoint.json", {**runtime, "identity": identity})
                    part_bin.unlink()
                    part_audit.unlink()
                    if settings["delete_committed_cache"]:
                        safe_delete_cache(path, cache)
                    progress("shard_committed", force=True, shard=item["path"], language=lang)
                except BaseException:
                    connection.rollback()
                    raise
            missing = {lang: settings["target_tokens_per_language"][lang] - runtime["counts"][lang] for lang in LANGUAGES
                       if runtime["counts"][lang] < settings["target_tokens_per_language"][lang]}
            if missing:
                raise ValueError(f"选定真实分片耗尽而有效 Token 不足；不发布 manifest，不重复补齐：{missing}")
            progress("verifying_token_streams", force=True)
            streams = {}
            for lang in LANGUAGES:
                path = output / f"train.{lang}.bin"
                total_count = base_counts[lang] + runtime["counts"][lang]
                if path.stat().st_size != total_count * 4:
                    raise ValueError("Token 文件大小与提交计数不符")
                stream = np.memmap(path, dtype="<u4", mode="r")
                for offset in range(0, len(stream), 4_000_000):
                    chunk = stream[offset:offset + 4_000_000]
                    if int(chunk.max()) >= info["vocab_size"]:
                        raise ValueError("最终 Token 校验发现越界")
                del stream
                streams[lang] = {"path": path.name, "dtype": "uint32", "token_count": total_count, "sha256": sha256_file(path)}
            provenance = [json.loads(row[0]) for row in connection.execute("SELECT source_json FROM commits ORDER BY sequence")]
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            manifest = {"schema_version": 1, "kind": "incremental_train_token_streams", "identity": identity,
                        "pipeline": pipeline,
                        "parent_manifest": {"path": str(Path(settings["parent_manifest"])), "sha256": parent_sha},
                        "tokenizer": {**info, "path": str(parent_paths["tokenizer"])},
                        "splits": {"train": streams}, "token_count": sum(stream["token_count"] for stream in streams.values()),
                        "new_token_count": sum(runtime["counts"].values()), "new_tokens_per_language": runtime["counts"],
                        "base_prefix": {lang: {"token_count": base_counts[lang], "sha256": parent["splits"]["train"][lang]["sha256"] if base_counts[lang] else None} for lang in LANGUAGES},
                        "statistics": runtime["stats"], "encoded_documents": runtime["documents"],
                        "input_provenance": provenance, "chinese_normalization": normalizer.provenance,
                        "holdout_fragment_protection": {"algorithm": "rolling64-casefold-whitespace-free-v1", "window_chars": settings["fingerprint_chars"],
                                                        "fingerprints": len(heldout), "holdout_documents": holdout_documents,
                                                        "limit": "精确规范化字符片段；不声称识别语义改写或所有近重复。"},
                        "audit": {"path": "incremental_audit.jsonl", "sha256": sha256_file(output / "incremental_audit.jsonl")},
                        "index": {"path": "dedup.sqlite", "sha256": sha256_file(output / "dedup.sqlite")},
                        "note": "train 包含已验证父前缀及新增真实 Token；尚未加入原验证/测试；不能直接取代完整训练 manifest。"}
            atomic_json(output / "expansion_manifest.json", manifest)
            progress("complete", force=True, status="complete", manifest="expansion_manifest.json")
            return output / "expansion_manifest.json"
        except BaseException as exc:
            try:
                progress("failed", force=True, status="failed", error=str(exc))
            except OSError:
                pass
            raise
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, root = load_config(args.config)
    if args.validate_only:
        print(json.dumps(validate_config(config, root), ensure_ascii=False, indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if hasattr(os, "nice"):
        os.nice(10)
    print(run_expansion(config, root, resume=args.resume))


if __name__ == "__main__":
    main()
