"""逐片获取固定 Wikipedia 快照：下载有界、JSONL 原子提交、失败可恢复。

只把完成的 ``raw/*.jsonl`` 暴露给下游；下载缓存和本步 raw 文本上限分别计量。
不调用 datasets 的仓库枚举，也不把整份 Parquet/JSONL 装进 Python 列表。
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

from llm_lab.common.io import canonical_hash, read_json, sha256_file, write_json
from .acquire import WIKIPEDIA_REVISION, _explicit_wikipedia_files, _wikipedia_endpoint, cache_wikipedia_file
from .common import config_path, json_line, load_config, positive_int


SNAPSHOT = "20231101"
REPOSITORY = "wikimedia/wikipedia"


def validate_config(config: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    """纯标准库预检，不联网、不创建目录、不导入 Arrow/Hub。"""
    root = Path(root or Path.cwd()).resolve()
    settings = dict(config)
    for key in ("output_dir", "cache_dir"):
        value = settings.get(key, ".cache/data" if key == "cache_dir" else None)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} 必须为非空路径")
        settings[key] = str(config_path(value, root))
    output, cache = Path(settings["output_dir"]), Path(settings["cache_dir"])
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("output_dir 与 cache_dir 必须是互不包含的目录")
    for key in ("max_docs", "max_bytes", "raw_text_bytes_per_language"):
        positive_int(settings, key)
    for key, default in (("batch_size", 64), ("min_free_bytes", 2_000_000_000),
                         ("max_line_bytes", 8 * 1024 * 1024), ("download_attempts", 3)):
        settings.setdefault(key, default)
        positive_int(settings, key)
    if settings["batch_size"] > 128:
        raise ValueError("batch_size 不得超过 128，避免无意扩大 Arrow 批次")
    settings.setdefault("transport", "official_head_direct")
    if settings["transport"] not in {"official_head_direct", "standard"}:
        raise ValueError("transport 必须为 official_head_direct 或 standard")
    sources = settings.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("sources 必须恰好包含 en、zh 两个 Wikipedia source")
    checked = {}
    for source in sources:
        if not isinstance(source, dict) or source.get("type") != "wikipedia":
            raise ValueError("本入口仅接受固定 Wikipedia source")
        language = source.get("language")
        if language not in {"en", "zh"} or language in checked:
            raise ValueError("sources 必须各含一个 en 与 zh")
        if source.get("revision") != WIKIPEDIA_REVISION or source.get("snapshot") != SNAPSHOT:
            raise ValueError("必须保持项目固定 Wikipedia snapshot 与完整 revision")
        if source.get("repository", REPOSITORY) != REPOSITORY:
            raise ValueError("仅允许 wikimedia/wikipedia 仓库")
        item = dict(source)
        item["data_files"] = _explicit_wikipedia_files(source)
        checked[language] = item
    settings["sources"] = [checked[language] for language in ("en", "zh")]
    # endpoint/代理/证书由进程外环境设置；绝不把代理值或签名 URL 写进配置/日志。
    settings["endpoint"] = _wikipedia_endpoint()
    if settings["transport"] == "official_head_direct" and settings["endpoint"] != "https://huggingface.co":
        raise ValueError("official_head_direct 必须使用官方 HF_ENDPOINT=https://huggingface.co")
    return settings


def _identity(settings: dict[str, Any]) -> str:
    return canonical_hash({key: value for key, value in settings.items()
                           if key not in {"project_root", "output_dir", "cache_dir"}})


def _disk_room(path: Path, reserve: int, incoming: int = 0) -> None:
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < reserve + incoming:
        raise OSError("磁盘剩余空间不足：保留 min_free_bytes，未启动/继续本次写入")


def _https_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("拒绝非 HTTPS 或含认证信息的下载目标")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _network_openers():
    # SSL_CERT_FILE 可指向平台已有系统 CA；保持证书验证，绝不使用 verify=False。
    context = ssl.create_default_context()
    metadata = urllib.request.build_opener(urllib.request.ProxyHandler(),
        urllib.request.HTTPSHandler(context=context), _NoRedirect())
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context), _HTTPSRedirect())
    return metadata, direct


def _resolve_target(metadata, url: str, digest: str, size: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "llm-lab-data/1"}, method="HEAD")
    try:
        with metadata.open(request, timeout=30) as response:
            target, headers = response.geturl(), response.headers
    except urllib.error.HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise
        target, headers = exc.headers.get("Location", ""), exc.headers
    linked = headers.get("X-Linked-Etag", "").strip('"')
    if linked and linked.lower() != digest:
        raise ValueError("官方 X-Linked-Etag 与固定 SHA256 不匹配")
    linked_size = headers.get("X-Linked-Size")
    if linked_size is not None and int(linked_size) != size:
        raise ValueError("官方 X-Linked-Size 与固定 size_bytes 不匹配")
    return _https_url(target)


def get_wikipedia_file(item: dict[str, Any], revision: str, cache: Path, settings: dict[str, Any],
                       progress: Callable[[int, str], None]) -> Path:
    """复用旧缓存验证；新下载支持官方 HEAD 经代理、签名 CAS 直连与 Range 恢复。"""
    destination = cache / "wikipedia" / f"{item['sha256']}.parquet"
    if destination.exists():
        return cache_wikipedia_file(item, revision, cache, settings["endpoint"])
    if settings["transport"] == "standard":
        _disk_room(cache, settings["min_free_bytes"], item["size_bytes"])
        return cache_wikipedia_file(item, revision, cache, settings["endpoint"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 本步专用 partial；不删除旧 acquire 或用户已有的其他 partial/cache。
    owner = canonical_hash(settings["output_dir"])[:12]
    partial = destination.with_name(destination.name + f".stream-{owner}.part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError("本步下载 partial 不是普通文件")
    _disk_room(cache, settings["min_free_bytes"], max(0, item["size_bytes"] - (partial.stat().st_size if partial.exists() else 0)))
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{item['path']}"
    metadata, direct = _network_openers()
    for attempt in range(settings["download_attempts"]):
        received = partial.stat().st_size if partial.exists() else 0
        if received > item["size_bytes"]:
            raise ValueError("当前下载 partial 超过固定分片大小；请先审计")
        hasher = hashlib.sha256()
        if received:
            with partial.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        try:
            if received < item["size_bytes"]:
                target = _resolve_target(metadata, url, item["sha256"], item["size_bytes"])
                headers = {"User-Agent": "llm-lab-data/1"}
                if received:
                    headers["Range"] = f"bytes={received}-"
                with direct.open(urllib.request.Request(target, headers=headers), timeout=60) as response:
                    _https_url(response.geturl())
                    if received and (response.status != 206 or not re.fullmatch(
                            rf"bytes {received}-\d+/{item['size_bytes']}", response.headers.get("Content-Range", ""))):
                        raise ValueError("远端未正确接受 Range，保留 partial，拒绝错误拼接")
                    with partial.open("ab" if partial.exists() else "xb") as output:
                        last_report = time.monotonic()
                        while True:
                            chunk = response.read(min(1024 * 1024, item["size_bytes"] - received + 1))
                            if not chunk:
                                break
                            if received + len(chunk) > item["size_bytes"]:
                                raise ValueError("下载超过固定 size_bytes")
                            _disk_room(cache, settings["min_free_bytes"], len(chunk))
                            output.write(chunk)
                            hasher.update(chunk)
                            received += len(chunk)
                            if time.monotonic() - last_report >= 2:
                                progress(received, "downloading")
                                last_report = time.monotonic()
                        output.flush()
                        os.fsync(output.fileno())
            if received != item["size_bytes"] or hasher.hexdigest() != item["sha256"]:
                raise ValueError("下载实际 size/SHA256 不匹配；保留 partial 供审计")
            os.replace(partial, destination)
            progress(received, "verified")
            return cache_wikipedia_file(item, revision, cache, settings["endpoint"])
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException) as exc:
            progress(received, "retry")
            if attempt + 1 == settings["download_attempts"]:
                # 网络异常可能含带签名 URL/代理凭据；只公开类型，禁止原异常链泄漏。
                raise RuntimeError(f"Wikipedia 下载失败（{type(exc).__name__}）；已保留当前 partial") from None
    raise AssertionError("unreachable")


def parquet_records(path: Path, source: dict[str, Any], batch_size: int) -> Iterator[dict[str, Any]]:
    """Arrow 每次最多转换一个小 batch；不依赖 datasets 构建额外磁盘缓存。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("需要 pyarrow；请在远程数据环境安装项目 data 依赖") from exc
    with pq.ParquetFile(path) as parquet:
        columns = set(parquet.schema_arrow.names)
        if not {"id", "text"} <= columns:
            raise ValueError("Wikipedia Parquet 缺少 id/text 列")
        selected = [key for key in ("id", "text", "url", "title") if key in columns]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=selected, use_threads=False):
            for row in batch.to_pylist():
                if not isinstance(row["text"], str) or not isinstance(row["id"], (str, int)):
                    raise ValueError("Wikipedia id/text 类型不正确")
                yield {"source_id": f"wikipedia:{source['snapshot']}:{source['language']}:{row['id']}",
                       "language": source["language"], "text": row["text"], "url": row.get("url"),
                       "title": row.get("title"), "source_revision": source["revision"]}


def _safe_output(output: Path, relative: str) -> Path:
    path = output / relative
    if not path.resolve().is_relative_to(output.resolve()) or path.is_symlink():
        raise ValueError("manifest 输出路径越界或为符号链接")
    return path


def _check_committed(output: Path, record: dict[str, Any]) -> None:
    path = _safe_output(output, record["path"])
    if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError("已完成 raw 分片大小/SHA256 不匹配，拒绝恢复")


def _commit(state: dict[str, Any], record: dict[str, Any]) -> None:
    state["files"].append(record)
    language = record["language"]
    state["next_file_index"][language] = record["source_file_index"] + 1
    for key in ("documents", "jsonl_bytes", "text_bytes"):
        state["totals"][key] += record[key]
        state["by_language"][language][key] += record[key]
    if record["stop_reason"]:
        state["language_stop_reasons"][language] = record["stop_reason"]
    state["pending"] = None
    state["active"] = None


def _new_state(settings: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "kind": "streaming_wikipedia_acquisition", "status": "running",
            "identity": _identity(settings), "repository": REPOSITORY, "revision": WIKIPEDIA_REVISION,
            "snapshot": SNAPSHOT, "settings": settings, "sources": settings["sources"], "files": [],
            "next_file_index": {"en": 0, "zh": 0}, "next_language": "en",
            "totals": {"documents": 0, "jsonl_bytes": 0, "text_bytes": 0},
            "by_language": {lang: {"documents": 0, "jsonl_bytes": 0, "text_bytes": 0} for lang in ("en", "zh")},
            "language_stop_reasons": {}, "active": None, "pending": None}


def _validate_saved_state(state: dict[str, Any], settings: dict[str, Any], output: Path) -> None:
    if (state.get("schema_version") != 1 or state.get("kind") != "streaming_wikipedia_acquisition"
            or state.get("identity") != _identity(settings)):
        raise ValueError("恢复配置/source 身份变化，拒绝复用旧 raw")
    sources = {source["language"]: source for source in settings["sources"]}
    expected = _new_state(settings)
    for record in state["files"]:
        language = record["language"]
        index = expected["next_file_index"].get(language)
        if (index is None or record["source_file_index"] != index
                or index >= len(sources[language]["data_files"])
                or record["source"] != sources[language]["data_files"][index]
                or record["source_revision"] != WIKIPEDIA_REVISION
                or record["path"] != f"raw/{language}-{index:05d}.jsonl"):
            raise ValueError("已完成 raw 的 source/index/path 与固定来源不匹配")
        if (any(isinstance(record[key], bool) or not isinstance(record[key], int) or record[key] < 0
                for key in ("documents", "jsonl_bytes", "text_bytes"))
                or record["size_bytes"] != record["jsonl_bytes"]
                or record["source_row_start"] != 0 or record["source_row_end_exclusive"] != record["documents"]):
            raise ValueError("已完成 raw 的计数/来源行范围无效")
        _check_committed(output, record)
        _commit(expected, record)
    if any(state[key] != expected[key] for key in ("totals", "by_language", "next_file_index")):
        raise ValueError("manifest 累计计数与已完成 raw 不一致")


def acquire_stream(config: dict[str, Any], root: str | Path | None = None, *, resume: bool = False,
                   validate_only: bool = False) -> Path | dict[str, Any]:
    settings = validate_config(config, root)
    if validate_only:
        return {"valid": True, "identity": _identity(settings), "settings": settings}
    output, cache = Path(settings["output_dir"]), Path(settings["cache_dir"])
    manifest = output / "acquisition_manifest.json"
    if resume and not output.is_dir():
        raise FileNotFoundError("恢复需要 acquisition_manifest.json")
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空；用新目录或 --resume，不覆盖旧数据")
    _disk_room(output, settings["min_free_bytes"])
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".acquisition.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("已有 acquisition writer lock；确认旧进程已结束后才可移除该锁") from None
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))
    state = None
    try:
        if resume:
            if not manifest.is_file():
                raise FileNotFoundError("恢复需要 acquisition_manifest.json")
            candidate = read_json(manifest)
            _validate_saved_state(candidate, settings, output)
            state = candidate
        else:
            if any(path != lock for path in output.iterdir()):
                raise FileExistsError("输出目录非空；不覆盖已有数据")
            (output / "raw").mkdir(exist_ok=True)
            state = _new_state(settings)
            write_json(manifest, state)
        # 事务窗口：JSONL 完成后先登记 pending，再 rename，最后登记 completed。
        # 必须先持有 writer lock，再修改恢复状态。
        if state.get("pending"):
            pending = state["pending"]
            final = _safe_output(output, pending["path"])
            temporary = final.with_suffix(".jsonl.part")
            if not final.exists() and temporary.exists():
                if temporary.stat().st_size != pending["size_bytes"] or sha256_file(temporary) != pending["sha256"]:
                    raise ValueError("pending raw 指纹不匹配")
                os.replace(temporary, final)
            _check_committed(output, pending)
            _commit(state, pending)
            write_json(manifest, state)
        state["status"] = "running"
        state.pop("error_type", None)
        while True:
            if state["totals"]["documents"] >= settings["max_docs"]:
                state["stop_reason"] = "max_docs"
                break
            if state["totals"]["jsonl_bytes"] >= settings["max_bytes"]:
                state["stop_reason"] = "max_bytes"
                break
            available = []
            for source in settings["sources"]:
                language = source["language"]
                if state["by_language"][language]["text_bytes"] >= settings["raw_text_bytes_per_language"]:
                    state["language_stop_reasons"][language] = "language_text_cap"
                if language not in state["language_stop_reasons"] and state["next_file_index"][language] < len(source["data_files"]):
                    available.append(source)
            if not available:
                state["stop_reason"] = "sources_exhausted_or_language_caps"
                break
            source = next((item for item in available if item["language"] == state["next_language"]), available[0])
            language = source["language"]
            index = state["next_file_index"][language]
            item = source["data_files"][index]
            relative = f"raw/{language}-{index:05d}.jsonl"
            final = _safe_output(output, relative)
            temporary = final.with_suffix(".jsonl.part")
            if final.exists():
                raise ValueError("发现未登记的完成 JSONL；拒绝覆盖，请先审计")
            state["active"] = {"language": language, "source_file_index": index, "source": item,
                               "stage": "starting", "download_bytes": 0}
            write_json(manifest, state)

            def progress(received: int, stage: str) -> None:
                state["active"].update(download_bytes=received, stage=stage)
                write_json(manifest, state)

            local = get_wikipedia_file(item, source["revision"], cache, settings, progress)
            count = text_bytes = jsonl_bytes = scanned = 0
            digest = hashlib.sha256()
            reason = None
            state["active"]["stage"] = "converting"
            write_json(manifest, state)
            # 仅覆盖本 source 的未提交 .part，不碰任何已完成文件。
            if temporary.is_symlink():
                raise ValueError("raw partial 不得为符号链接")
            with temporary.open("wb") as handle:
                for row in parquet_records(local, source, settings["batch_size"]):
                    scanned += 1
                    if state["totals"]["documents"] + count >= settings["max_docs"]:
                        reason = "max_docs"
                        break
                    text_size = len(row["text"].encode("utf-8"))
                    line = json_line(row).encode("utf-8")
                    if len(line) > settings["max_line_bytes"]:
                        raise ValueError("Wikipedia JSONL 单行超过 max_line_bytes，拒绝静默丢文档")
                    if state["by_language"][language]["text_bytes"] + text_bytes + text_size > settings["raw_text_bytes_per_language"]:
                        reason = "language_text_cap"
                        break
                    if state["totals"]["jsonl_bytes"] + jsonl_bytes + len(line) > settings["max_bytes"]:
                        reason = "max_bytes"
                        break
                    _disk_room(output, settings["min_free_bytes"], len(line))
                    handle.write(line)
                    digest.update(line)
                    count += 1
                    text_bytes += text_size
                    jsonl_bytes += len(line)
                    if count % 1024 == 0:
                        state["active"].update(documents=count, jsonl_bytes=jsonl_bytes, text_bytes=text_bytes)
                        write_json(manifest, state)
                handle.flush()
                os.fsync(handle.fileno())
            record = {"language": language, "source_file_index": index, "source": item,
                      "source_revision": source["revision"], "path": relative, "sha256": digest.hexdigest(),
                      "size_bytes": jsonl_bytes, "documents": count, "jsonl_bytes": jsonl_bytes,
                      "text_bytes": text_bytes, "stop_reason": reason,
                      "source_row_start": 0, "source_row_end_exclusive": count,
                      "source_rows_scanned": scanned, "source_exhausted": reason is None}
            state["pending"] = record
            state["next_language"] = "zh" if language == "en" else "en"
            write_json(manifest, state)
            os.replace(temporary, final)
            _commit(state, record)
            write_json(manifest, state)
            if reason in {"max_docs", "max_bytes"}:
                state["stop_reason"] = reason
                break
        state["status"] = "completed"
        state["active"] = None
        write_json(manifest, state)
    except BaseException as exc:
        if state is not None:
            state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            state["error_type"] = type(exc).__name__
            write_json(manifest, state)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="逐分片、可恢复、有磁盘预算的 Wikipedia 获取")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, root = load_config(args.config)
    result = acquire_stream(config, root, resume=args.resume, validate_only=args.validate_only)
    print(json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
