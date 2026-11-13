"""小型本地协议测试：不联网、不下载 Wikipedia、不启动 GPU。"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from llm_lab.common.io import canonical_hash, read_json, write_json
from llm_lab.data import acquire_stream as stream
from llm_lab.data.common import read_jsonl


def configuration(tmp_path, files=2):
    sources = []
    for language in ("en", "zh"):
        items = [{"path": f"20231101.{language}/train-{index:05d}-of-{files:05d}.parquet",
                  "sha256": hashlib.sha256(f"{language}:{index}".encode()).hexdigest(), "size_bytes": 100}
                 for index in range(files)]
        sources.append({"type": "wikipedia", "snapshot": stream.SNAPSHOT, "language": language,
                        "revision": stream.WIKIPEDIA_REVISION, "max_download_bytes": 100 * files,
                        "data_files": items})
    return {"output_dir": str(tmp_path / "raw"), "cache_dir": str(tmp_path / "cache"),
            "max_docs": 100, "max_bytes": 1_000_000, "raw_text_bytes_per_language": 100_000,
            "min_free_bytes": 1, "sources": sources}


def stub_source(monkeypatch, *, rows=2):
    downloads = []

    def get_file(item, revision, cache, settings, progress):
        downloads.append(item["path"])
        progress(item["size_bytes"], "verified")
        return Path(item["path"])

    def records(path, source, batch_size):
        index = int(path.name.split("-")[1])
        for row in range(rows):
            yield {"source_id": f"wikipedia:20231101:{source['language']}:{index}:{row}",
                   "language": source["language"], "text": "中文" if source["language"] == "zh" else "en",
                   "source_revision": source["revision"]}

    monkeypatch.setattr(stream, "get_wikipedia_file", get_file)
    monkeypatch.setattr(stream, "parquet_records", records)
    return downloads


def test_validate_only_has_no_import_or_output_side_effects(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setattr(stream, "get_wikipedia_file", lambda *a: pytest.fail("must not download"))
    result = stream.acquire_stream(config, validate_only=True)
    assert result["valid"]
    assert not Path(config["output_dir"]).exists()
    assert not Path(config["cache_dir"]).exists()
    config["sources"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="固定"):
        stream.acquire_stream(config, validate_only=True)


def test_round_robin_commits_each_shard_with_hashes_and_row_ranges(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    downloads = stub_source(monkeypatch)
    manifest = stream.acquire_stream(config)
    state = read_json(manifest)
    assert [path.split("/")[0] for path in downloads] == ["20231101.en", "20231101.zh"] * 2
    assert state["status"] == "completed"
    assert state["totals"]["documents"] == 8
    assert [item["language"] for item in state["files"]] == ["en", "zh", "en", "zh"]
    for item in state["files"]:
        path = manifest.parent / item["path"]
        assert item["source_row_start"] == 0 and item["source_row_end_exclusive"] == 2
        assert item["source_rows_scanned"] == 2 and item["source_exhausted"]
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(list(read_jsonl(path))) == 2
    assert not list(manifest.parent.rglob("*.part"))
    assert not (manifest.parent / ".acquisition.lock").exists()


def test_utf8_language_caps_keep_complete_rows_and_are_not_jsonl_byte_caps(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config["raw_text_bytes_per_language"] = 7
    stub_source(monkeypatch, rows=8)
    manifest = stream.acquire_stream(config)
    state = read_json(manifest)
    assert state["by_language"]["en"]["text_bytes"] == 6  # 3 × 两个 ASCII 字节。
    assert state["by_language"]["zh"]["text_bytes"] == 6  # 1 × 两个三字节汉字。
    assert state["by_language"]["en"]["documents"] == 3
    assert state["by_language"]["zh"]["documents"] == 1
    assert all(item["stop_reason"] == "language_text_cap" for item in state["files"])
    assert all(item["source_rows_scanned"] == item["documents"] + 1 for item in state["files"])
    assert state["totals"]["jsonl_bytes"] > state["totals"]["text_bytes"]


def test_global_docs_and_bytes_stop_before_next_download(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config["max_docs"] = 2
    downloads = stub_source(monkeypatch, rows=8)
    state = read_json(stream.acquire_stream(config))
    assert state["totals"]["documents"] == 2 and len(downloads) == 1
    assert state["stop_reason"] == "max_docs"
    config = configuration(tmp_path / "second")
    config["max_bytes"] = 1
    state = read_json(stream.acquire_stream(config))
    assert state["totals"]["documents"] == 0 and state["totals"]["jsonl_bytes"] == 0
    assert state["stop_reason"] == "max_bytes"


def test_failure_preserves_completed_shards_and_resume_retries_only_current(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    downloads = stub_source(monkeypatch)
    reader = stream.parquet_records
    failed = False

    def interrupted(path, source, batch_size):
        nonlocal failed
        for row in reader(path, source, batch_size):
            yield row
            if source["language"] == "zh" and not failed:
                failed = True
                raise RuntimeError("fixture interruption")

    monkeypatch.setattr(stream, "parquet_records", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        stream.acquire_stream(config)
    manifest = Path(config["output_dir"]) / "acquisition_manifest.json"
    state = read_json(manifest)
    assert len(state["files"]) == 1 and state["files"][0]["language"] == "en"
    original = (manifest.parent / state["files"][0]["path"]).read_bytes()
    assert (manifest.parent / "raw/zh-00000.jsonl.part").exists()
    monkeypatch.setattr(stream, "parquet_records", reader)
    stream.acquire_stream(config, resume=True)
    state = read_json(manifest)
    assert state["status"] == "completed" and len(state["files"]) == 4
    assert (manifest.parent / state["files"][0]["path"]).read_bytes() == original
    assert downloads.count("20231101.en/train-00000-of-00002.parquet") == 1
    assert downloads.count("20231101.zh/train-00000-of-00002.parquet") == 2


def test_resume_rejects_config_change_and_committed_file_tampering(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    stub_source(monkeypatch)
    manifest = stream.acquire_stream(config)
    altered = dict(config, max_docs=200)
    with pytest.raises(ValueError, match="身份变化"):
        stream.acquire_stream(altered, resume=True)
    state = read_json(manifest)
    path = manifest.parent / state["files"][0]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        stream.acquire_stream(config, resume=True)


@pytest.mark.parametrize("renamed", [False, True])
def test_resume_finishes_pending_rename_transaction_without_redownload(tmp_path, monkeypatch, renamed):
    config = configuration(tmp_path, files=1)
    downloads = stub_source(monkeypatch)
    manifest = stream.acquire_stream(config)
    state = read_json(manifest)
    last = state["files"].pop()
    for key in ("documents", "jsonl_bytes", "text_bytes"):
        state["totals"][key] -= last[key]
        state["by_language"]["zh"][key] -= last[key]
    state["pending"] = last
    state["next_file_index"]["zh"] = 0
    state["status"] = "failed"
    if not renamed:
        final = manifest.parent / last["path"]
        final.replace(final.with_suffix(".jsonl.part"))
    write_json(manifest, state)
    downloads.clear()
    stream.acquire_stream(config, resume=True)
    state = read_json(manifest)
    assert downloads == [] and len(state["files"]) == 2 and state["pending"] is None
    assert state["totals"]["documents"] == 4


def test_disk_guard_rejects_before_downloading_or_creating_output(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setattr(stream.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(stream, "get_wikipedia_file", lambda *a: pytest.fail("must not download"))
    with pytest.raises(OSError, match="磁盘"):
        stream.acquire_stream(config)
    assert not Path(config["output_dir"]).exists()


def test_existing_writer_lock_and_unregistered_output_are_not_overwritten(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    stub_source(monkeypatch)
    manifest = stream.acquire_stream(config)
    lock = manifest.parent / ".acquisition.lock"
    lock.write_text("123")
    with pytest.raises(RuntimeError, match="writer lock"):
        stream.acquire_stream(config, resume=True)
    assert lock.read_text() == "123"
    with pytest.raises(FileExistsError, match="非空"):
        stream.acquire_stream(config)


def test_locked_resume_does_not_read_or_modify_authoritative_state(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    stub_source(monkeypatch)
    manifest = stream.acquire_stream(config)
    original = manifest.read_bytes()
    (manifest.parent / ".acquisition.lock").write_text("123")
    monkeypatch.setattr(stream, "read_json", lambda *a: pytest.fail("state must not be read before lock"))
    with pytest.raises(RuntimeError, match="writer lock"):
        stream.acquire_stream(config, resume=True)
    assert manifest.read_bytes() == original


def test_resume_rejects_corrupted_accounting_without_rewriting_manifest(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    stub_source(monkeypatch)
    manifest = stream.acquire_stream(config)
    state = read_json(manifest)
    state["totals"]["jsonl_bytes"] = 0
    write_json(manifest, state)
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="累计计数"):
        stream.acquire_stream(config, resume=True)
    assert manifest.read_bytes() == original


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None, url="https://cas-bridge.xethub.hf.co/file"):
        super().__init__(data)
        self.status, self.headers, self.url = status, headers or {}, url

    def geturl(self):
        return self.url


def download_settings(tmp_path, payload):
    config = configuration(tmp_path, files=1)
    item = config["sources"][0]["data_files"][0]
    item.update(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    config["sources"][0]["max_download_bytes"] = len(payload)
    settings = stream.validate_config(config)
    return item, settings, Path(settings["cache_dir"])


def test_official_metadata_and_direct_range_resume_verify_complete_sha(tmp_path, monkeypatch):
    payload = b"fixture original verified contents"
    item, settings, cache = download_settings(tmp_path, payload)
    partial = cache / "wikipedia" / (item["sha256"] + f".parquet.stream-{canonical_hash(settings['output_dir'])[:12]}.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:7])
    requests = []

    def head(request, timeout):
        requests.append(("metadata", request))
        raise urllib.error.HTTPError(request.full_url, 302, "redirect", {
            "Location": "https://cas-bridge.xethub.hf.co/file?secret=not-logged",
            "X-Linked-Etag": item["sha256"], "X-Linked-Size": str(len(payload))}, None)

    def direct(request, timeout):
        requests.append(("direct", request))
        assert request.headers["Range"] == "bytes=7-"
        return Response(payload[7:], status=206, headers={"Content-Range": f"bytes 7-{len(payload)-1}/{len(payload)}"})

    monkeypatch.setattr(stream, "_network_openers", lambda: (SimpleNamespace(open=head), SimpleNamespace(open=direct)))
    path = stream.get_wikipedia_file(item, stream.WIKIPEDIA_REVISION, cache, settings, lambda *a: None)
    assert path.read_bytes() == payload and not partial.exists()
    assert requests[0][1].method == "HEAD" and len(requests) == 2
    # 复用旧 helper 校验缓存，不再走网络。
    monkeypatch.setattr(stream, "_network_openers", lambda: pytest.fail("cache must be reused"))
    assert stream.get_wikipedia_file(item, stream.WIKIPEDIA_REVISION, cache, settings, lambda *a: None) == path


def test_download_failure_redacts_signed_url_and_preserves_partial(tmp_path, monkeypatch):
    payload = b"verified fixture"
    item, settings, cache = download_settings(tmp_path, payload)
    settings["download_attempts"] = 1

    def fail(*a, **kw):
        raise urllib.error.URLError("https://signed.example/?token=SECRET")

    monkeypatch.setattr(stream, "_network_openers", lambda: (SimpleNamespace(open=fail), None))
    with pytest.raises(RuntimeError) as error:
        stream.get_wikipedia_file(item, stream.WIKIPEDIA_REVISION, cache, settings, lambda *a: None)
    assert "SECRET" not in str(error.value) and error.value.__suppress_context__


def test_official_etag_mismatch_rejects_before_direct_read(tmp_path, monkeypatch):
    item, settings, cache = download_settings(tmp_path, b"fixture")

    def head(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 302, "redirect", {
            "Location": "https://cas-bridge.xethub.hf.co/file", "X-Linked-Etag": "0" * 64}, None)

    monkeypatch.setattr(stream, "_network_openers", lambda: (SimpleNamespace(open=head), None))
    with pytest.raises(ValueError, match="Etag"):
        stream.get_wikipedia_file(item, stream.WIKIPEDIA_REVISION, cache, settings, lambda *a: None)


@pytest.mark.parametrize("prefix", [b"", b"verified "])
def test_existing_partial_including_empty_file_can_resume(tmp_path, monkeypatch, prefix):
    payload = b"verified fixture"
    item, settings, cache = download_settings(tmp_path, payload)
    partial = cache / "wikipedia" / (item["sha256"] + f".parquet.stream-{canonical_hash(settings['output_dir'])[:12]}.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(prefix)
    monkeypatch.setattr(stream, "_resolve_target", lambda *a: "https://cas-bridge.xethub.hf.co/file")

    def direct(request, timeout):
        return Response(payload[len(prefix):], status=206 if prefix else 200,
                        headers={"Content-Range": f"bytes {len(prefix)}-{len(payload)-1}/{len(payload)}"})

    monkeypatch.setattr(stream, "_network_openers", lambda: (None, SimpleNamespace(open=direct)))
    path = stream.get_wikipedia_file(item, stream.WIKIPEDIA_REVISION, cache, settings, lambda *a: None)
    assert path.read_bytes() == payload


def test_ignored_range_keeps_partial_unchanged(tmp_path, monkeypatch):
    payload = b"verified fixture"
    item, settings, cache = download_settings(tmp_path, payload)
    partial = cache / "wikipedia" / (item["sha256"] + f".parquet.stream-{canonical_hash(settings['output_dir'])[:12]}.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:4])
    monkeypatch.setattr(stream, "_resolve_target", lambda *a: "https://cas-bridge.xethub.hf.co/file")
    monkeypatch.setattr(stream, "_network_openers", lambda: (
        None, SimpleNamespace(open=lambda *a, **k: Response(payload))))
    with pytest.raises(ValueError, match="Range"):
        stream.get_wikipedia_file(item, stream.WIKIPEDIA_REVISION, cache, settings, lambda *a: None)
    assert partial.read_bytes() == payload[:4]


def test_retry_after_stream_disconnect_appends_only_remaining_bytes(tmp_path, monkeypatch):
    payload = b"verified fixture"
    item, settings, cache = download_settings(tmp_path, payload)
    settings["download_attempts"] = 2
    calls = []

    class Interrupted(Response):
        def read(self, size):
            if self.tell():
                raise ConnectionError("fixture disconnected")
            return super().read(4)

    def direct(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            return Interrupted(payload)
        assert request.headers["Range"] == "bytes=4-"
        return Response(payload[4:], status=206, headers={"Content-Range": f"bytes 4-{len(payload)-1}/{len(payload)}"})

    monkeypatch.setattr(stream, "_resolve_target", lambda *a: "https://cas-bridge.xethub.hf.co/file")
    monkeypatch.setattr(stream, "_network_openers", lambda: (None, SimpleNamespace(open=direct)))
    path = stream.get_wikipedia_file(item, stream.WIKIPEDIA_REVISION, cache, settings, lambda *a: None)
    assert len(calls) == 2 and path.read_bytes() == payload


def test_real_arrow_iter_batches_preserves_original_row_schema(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = configuration(tmp_path)["sources"][0]
    path = tmp_path / "tiny.parquet"
    pq.write_table(pa.table({"id": ["1", "2", "3"], "text": ["hello", "中文", "third"],
                             "url": ["u1", "u2", "u3"], "title": ["a", "b", "c"]}), path)
    rows = list(stream.parquet_records(path, source, batch_size=1))
    assert [row["text"] for row in rows] == ["hello", "中文", "third"]
    assert rows[1]["source_id"] == "wikipedia:20231101:en:2"
    assert all(row["source_revision"] == stream.WIKIPEDIA_REVISION for row in rows)
