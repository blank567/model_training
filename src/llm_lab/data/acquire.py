"""有界获取入口；记录发行包指纹，不把观测散列冒充独立来源认证。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import urllib.request
from urllib.parse import urlsplit
from collections import Counter
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .common import config_path, json_line, load_config, make_output_dir, positive_int, read_jsonl, sha256_file, write_json


WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
MASSIVE_RELEASE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"


def _explicit_wikipedia_files(source: dict[str, Any]) -> list[dict[str, Any]]:
    """可选分片清单只允许当前快照/语言目录，身份由固定 revision 和内容 SHA 共同固定。"""
    files = source["data_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Wikipedia data_files 必须是非空分片对象列表")
    cap = positive_int(source, "max_download_bytes")
    prefix = f"{source['snapshot']}.{source['language']}"
    checked = []
    paths = set()
    digest_sizes = {}
    total = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("每个 data_files 项必须含 path、sha256、size_bytes")
        path = item.get("path")
        # 不接受 URL、query、百分号转义、反斜杠或路径规范化后的穿越写法。
        if (not isinstance(path, str)
                or not re.fullmatch(re.escape(prefix) + r"/[A-Za-z0-9][A-Za-z0-9_.-]*\.parquet", path)
                or ".." in PurePosixPath(path).parts):
            raise ValueError(f"data_files path 必须是 {prefix}/ 下的相对 Parquet 文件：{path!r}")
        if path in paths:
            raise ValueError(f"data_files 含重复路径：{path}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("data_files sha256 必须为完整 64 位十六进制散列")
        digest = digest.lower()
        size = positive_int(item, "size_bytes")
        if digest in digest_sizes and digest_sizes[digest] != size:
            raise ValueError("同一分片 SHA256 被声明为不同大小")
        total += size
        if total > cap:
            raise ValueError("声明分片总大小超过 source.max_download_bytes")
        paths.add(path)
        digest_sizes[digest] = size
        checked.append({"path": path, "sha256": digest, "size_bytes": size})
    return checked


def _wikipedia_endpoint() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    parts = urlsplit(endpoint)
    if (parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or parts.path not in {"", "/"}
            or any(char.isspace() for char in endpoint)):
        raise ValueError("HF_ENDPOINT 必须是无用户信息、查询、fragment 或额外路径的 HTTPS 端点")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("HF_ENDPOINT 端口非法") from exc
    return endpoint.rstrip("/")


def _hub_download_headers() -> dict[str, str]:
    from huggingface_hub.utils import build_hf_headers

    # Wikipedia 是公开数据；保留 Hub 的 User-Agent，但不发送本机保存的登录 token。
    return build_hf_headers(token=False, library_name="llm-lab", library_version="0.1")


def cache_wikipedia_file(item: dict[str, Any], revision: str, cache_dir: Path, endpoint: str) -> Path:
    """完整分片先校验落缓存，再由本地 Parquet reader 流式消费；绝不触发 Hub 仓库枚举。"""
    directory = cache_dir / "wikipedia"
    path = directory / f"{item['sha256']}.parquet"
    if path.exists():
        if (not path.is_file() or not path.resolve().is_relative_to(directory.resolve())
                or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]):
            raise ValueError(f"Wikipedia 缓存分片大小或 SHA256 不匹配：{path}")
        return path.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".parquet.part")
    url = f"{endpoint}/datasets/wikimedia/wikipedia/resolve/{revision}/{item['path']}"
    request = urllib.request.Request(url, headers=_hub_download_headers())
    digest = hashlib.sha256()
    written = 0
    partial_created = False
    try:
        with partial.open("xb") as handle:
            partial_created = True
            with urllib.request.urlopen(request, timeout=30) as response:
                if urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("拒绝 Wikipedia 分片 HTTPS 降级")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != item["size_bytes"]:
                    raise ValueError("Wikipedia 分片 Content-Length 与声明大小不匹配")
                while True:
                    chunk = response.read(min(1024 * 1024, item["size_bytes"] - written + 1))
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > item["size_bytes"]:
                        raise ValueError("Wikipedia 分片下载超过声明大小/下载预算")
                    digest.update(chunk)
                    handle.write(chunk)
        if written != item["size_bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("Wikipedia 分片实际大小或 SHA256 不匹配")
        partial.replace(path)
    except BaseException:
        if partial_created and partial.exists():
            partial.unlink()
        raise
    return path.resolve()


def wikipedia_records(source: dict[str, Any], cache_dir: Path) -> Iterator[dict[str, Any]]:
    revision = source.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Wikipedia revision 必须固定为完整 40 位 commit SHA，不能使用 main")
    language = source.get("language")
    snapshot = source.get("snapshot")
    if language not in {"en", "zh"} or not re.fullmatch(r"\d{8}", str(snapshot)):
        raise ValueError("Wikipedia source 需要 language=en/zh 及 YYYYMMDD snapshot")
    explicit_files = _explicit_wikipedia_files(source) if "data_files" in source else None
    endpoint = _wikipedia_endpoint() if explicit_files is not None else None
    # 必须早于 datasets/huggingface_hub 导入，避免新缓存落到用户目录默认盘。
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("远程获取需要 datasets；请按项目依赖在远程环境中安装") from exc
    # cache_dir 和 HF_HOME 都在项目指定盘/远程工作目录，不依赖用户目录默认缓存。
    if explicit_files is not None:
        # 传入本地绝对路径，避免 Datasets 将 HTTPS /resolve/ 重写为 hf:// 再做全仓库元数据访问。
        local_files = [str(cache_wikipedia_file(item, revision, cache_dir, endpoint)) for item in explicit_files]
        dataset = load_dataset("parquet", data_files={"train": local_files}, split="train",
                               streaming=True, cache_dir=str(cache_dir / "datasets"))
    else:
        dataset = load_dataset("wikimedia/wikipedia", f"{snapshot}.{language}", split="train",
                               revision=revision, streaming=True, cache_dir=str(cache_dir / "datasets"))
    for row in dataset:
        yield {"source_id": f"wikipedia:{snapshot}:{language}:{row['id']}", "language": language,
               "text": row["text"], "url": row.get("url"), "title": row.get("title"),
               "source_revision": revision}


def download_verified(source: dict[str, Any], cache_dir: Path) -> Path:
    expected = source.get("expected_sha256")
    if isinstance(expected, str) and expected.startswith("env:"):
        env_name = expected[4:]
        expected = os.environ.get(env_name)
        if expected is None:
            raise ValueError(f"指定的校验和环境变量未设置：{env_name}")
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected)):
        raise ValueError("expected_sha256 若提供，必须是 64 位十六进制 SHA256")
    expected = expected.lower() if expected is not None else None
    url = source.get("url")
    if url != MASSIVE_RELEASE_URL:
        raise ValueError("第一版仅接受固定的 MASSIVE 1.1 官方发行 URL")
    max_download = positive_int(source, "max_download_bytes")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "massive-1.1.tar.gz"
    lock_path = cache_dir / "massive-1.1.lock.json"
    if path.exists():
        if not lock_path.exists():
            raise ValueError("存在未登记版本指纹的缓存包；请审计并移走它，不能静默信任")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        observed = sha256_file(path)
        if (path.stat().st_size > max_download or lock.get("url") != url
                or path.stat().st_size != lock.get("size_bytes") or observed != lock.get("observed_sha256")
                or (expected is not None and observed != expected)):
            raise ValueError("缓存发行包大小或 SHA256 与固定指纹不匹配；请审计并移走该文件")
        return path
    partial = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    written = 0
    partial_created = False
    request = urllib.request.Request(url, headers={"User-Agent": "llm-lab-data/1"})
    try:
        with partial.open("xb") as output:
            partial_created = True
            with urllib.request.urlopen(request, timeout=60) as response:
                resolved_url = response.geturl()
                if not resolved_url.startswith("https://"):
                    raise ValueError("拒绝官方 HTTPS 下载降级到非 HTTPS")
                length = response.headers.get("Content-Length")
                if length and int(length) > max_download:
                    raise ValueError("远程发行包 Content-Length 超出 max_download_bytes")
                while True:
                    chunk = response.read(min(1024 * 1024, max_download - written + 1))
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_download:
                        raise ValueError("发行包下载超过 max_download_bytes")
                    digest.update(chunk)
                    output.write(chunk)
        observed = digest.hexdigest()
        if expected is not None and observed != expected:
            raise ValueError("MASSIVE 发行包 SHA256 不匹配，拒绝读取")
        partial.replace(path)
        write_json(lock_path, {"schema_version": 1, "release": "MASSIVE-1.1", "url": url,
                   "resolved_url": resolved_url, "observed_sha256": observed, "size_bytes": written,
                   "expected_sha256": expected, "expected_checksum_matched": expected is not None,
                   "independent_checksum_verified": False,
                   "note": "observed_sha256 是首次观测版本指纹，不等于独立来源认证。"})
    except BaseException:
        # 仅清理当前函数自己创建且位于明确缓存目录内的单个 .part 文件。
        if partial_created and partial.exists():
            partial.unlink()
        raise
    return path


def validate_tar_member(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name:
        raise ValueError(f"拒绝不安全的归档路径：{name}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"拒绝归档符号链接/硬链接/特殊文件：{name}")
    return path


def massive_archive_records(path: Path, source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    max_uncompressed = positive_int(source, "max_uncompressed_bytes")
    max_member = positive_int(source, "max_member_bytes")
    max_members = int(source.get("max_members", 1000))
    locales = set(source.get("locales", ["en-US", "zh-CN"]))
    if not locales or not locales <= {"en-US", "zh-CN"}:
        raise ValueError("第一版 MASSIVE locales 仅允许 en-US/zh-CN")
    total = 0
    # 不使用 extractall；只把已选语言的字节流解析为 JSON，绝不信任归档路径写磁盘。
    with tarfile.open(path, "r|gz") as archive:
        for index, member in enumerate(archive):
            if index >= max_members:
                raise ValueError("归档文件数量超过 max_members")
            member_path = validate_tar_member(member)
            total += member.size
            if total > max_uncompressed or member.size > max_member:
                raise ValueError("归档解压声明大小超过有界预算")
            if not member.isfile() or member_path.name not in {f"{locale}.jsonl" for locale in locales}:
                continue
            reader = archive.extractfile(member)
            if reader is None:
                raise ValueError("无法读取归档成员")
            with reader:
                while True:
                    raw = reader.readline(8 * 1024 * 1024 + 1)
                    if not raw:
                        break
                    if len(raw) > 8 * 1024 * 1024:
                        raise ValueError("MASSIVE 单行超出 8MiB")
                    row = json.loads(raw)
                    if not isinstance(row, dict) or row.get("locale") != member_path.stem:
                        raise ValueError("归档语言/JSON 对象与声明不符")
                    yield row


def acquire(config: dict[str, Any], root: str | Path | None = None) -> Path:
    root = Path(root or Path.cwd()).resolve()
    max_docs = positive_int(config, "max_docs")
    max_bytes = positive_int(config, "max_bytes")
    sources = config.get("sources", [])
    if not sources:
        raise ValueError("sources 不能为空")
    output = config_path(config["output_dir"], root)
    cache_dir = config_path(config.get("cache_dir", ".cache/data"), root)
    make_output_dir(output, overwrite=bool(config.get("overwrite", False)))
    data_path = output / "records.jsonl"
    stats: Counter = Counter()
    source_stats = []
    with data_path.open("w", encoding="utf-8", newline="\n") as handle:
        for source_index, source in enumerate(sources):
            source_type = source.get("type")
            release_identity = None
            if source_type == "local_jsonl":
                records = read_jsonl(config_path(source["path"], root))
            elif source_type == "wikipedia":
                records = wikipedia_records(source, cache_dir)
            elif source_type == "massive_archive":
                archive_path = download_verified(source, cache_dir)
                release_identity = json.loads((cache_dir / "massive-1.1.lock.json").read_text(encoding="utf-8"))
                records = massive_archive_records(archive_path, source)
            else:
                raise ValueError(f"未知 source type：{source_type}")
            local_stats: Counter = Counter()
            # 默认给各 source 相同扫描配额，避免英文源先耗尽总预算。
            source_limit = min(int(source.get("max_docs", max(1, max_docs // len(sources)))), max_docs - stats["scanned"])
            source_bytes = min(int(source.get("max_bytes", max_bytes // len(sources))), max_bytes - stats["bytes"])
            try:
                for row in islice(records, source_limit):
                    stats["scanned"] += 1
                    local_stats["scanned"] += 1
                    line = json_line(row)
                    size = len(line.encode("utf-8"))
                    if stats["bytes"] + size > max_bytes or local_stats["bytes"] + size > source_bytes:
                        local_stats["stopped_max_bytes"] += 1
                        break
                    handle.write(line)
                    stats["bytes"] += size
                    stats["written"] += 1
                    local_stats["bytes"] += size
                    local_stats["written"] += 1
            finally:
                if hasattr(records, "close"):
                    records.close()
            source_stats.append({"source_index": source_index, "source": source, "statistics": dict(local_stats),
                                 "release_identity": release_identity})
    manifest = {"schema_version": 1, "kind": "bounded_acquisition", "path": "records.jsonl",
                "sha256": sha256_file(data_path), "statistics": dict(stats), "sources": source_stats,
                "budgets": {"max_docs": max_docs, "max_bytes": max_bytes},
                "budget_scope": "max_docs 约束扫描文档；max_bytes 约束写出的 UTF-8 JSONL。HF streaming 的网络读取按 Parquet row group，网络字节不等于输出字节。",
                "config": config}
    write_json(output / "acquisition_manifest.json", manifest)
    return data_path


def main():
    parser = argparse.ArgumentParser(description="有界获取公开数据或本地 fixture")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    print(acquire(config, root))


if __name__ == "__main__":
    main()
