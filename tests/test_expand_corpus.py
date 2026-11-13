"""增量网页语料：真实 BPE、镜像排除、分片 crash/recovery，不联网/GPU。"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from llm_lab.data.common import read_jsonl, sha256_file, text_hash
from llm_lab.data.prepare import prepare_pretrain

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("expand_corpus_under_test", ROOT / "scripts/expand_corpus.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class FixtureNormalizer:
    provenance = {"mode": "t2s", "fixture": "test-only-fixed-conversion"}

    def convert(self, text, language="zh"):
        return text.replace("臺灣", "台湾").replace("資料", "资料") if language == "zh" else text


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    parent = prepare_pretrain({"inputs": [str(ROOT / "examples/pretrain_documents.jsonl")],
                              "output_dir": str(tmp_path / "parent"), "max_docs": 100, "max_bytes": 200000,
                              "max_tokens": 20000, "tokenizer": {"vocab_size": 512, "min_frequency": 1},
                              "quality": {"min_chars": 20}}, tmp_path)
    manifest = json.loads(parent.read_text(encoding="utf-8"))
    for key, name in (("audit", "document_audit.jsonl"), ("documents", "documents.jsonl")):
        manifest[key] = {"path": name, "sha256": sha256_file(parent.parent / name)}
    manifest["chinese_normalization"] = FixtureNormalizer.provenance
    parent.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(worker, "build_chinese_normalizer", lambda mode: FixtureNormalizer())
    originals = list(read_jsonl(parent.parent / "documents.jsonl"))
    holdout = next(row for row in originals if row["split"] != "train" and row["language"] == "en")
    training = next(row for row in originals if row["split"] == "train" and row["language"] == "en")
    english = "New astronomy measurements trace distant planets through changes in light and motion. Observers compare several telescopes, document their uncertainty, and publish reproducible tables for independent analysis. This new passage concerns stellar spectroscopy."
    chinese = "臺灣的科学观察小组收集植物生长的资料。他们在不同海拔记录温度和湿度，使用固定的方法测量叶片，再把独立观测结果整理成表格。参与者认真检查仪器状态，标注不确定性，并按照预先约定的分析规则比较结果。研究团队公开资料供其他学校复查，让观察过程能够被重新检验。"
    rows = {"en": [
        {"id": "wiki", "url": "https://en.wikipedia.org/wiki/Article", "text": english},
        {"id": "old_alias", "url": "https://new.example/alias", "text": training["text"]},
        {"id": "mirror", "url": "https://mirror.example/article", "text": "Unrelated framing words. " + holdout["text"] + " More wrapper words."},
        {"id": "new_en", "url": "https://science.example/stars", "text": english},
        {"id": "new_en_dup_url", "url": "http://science.example/stars#top", "text": english + " New footer."},
    ], "zh": [{"id": "new_zh", "url": "https://education.example/plants", "text": chinese}]}
    sources = []
    for lang in worker.LANGUAGES:
        path = tmp_path / f"{lang}.parquet"
        path.write_bytes(f"fixture-{lang}-not-a-real-parquet".encode())
        sources.append({"language": lang, "repository": "fixture/" + lang, "revision": "a" * 40,
                        "data_files": [{"path": path.name, "local_path": str(path), "size": path.stat().st_size,
                                        "sha256": sha256_file(path)}]})
    config = {"parent_manifest": str(parent), "output_dir": str(tmp_path / "result"), "cache_dir": str(tmp_path / "cache"),
              "sources": sources, "target_tokens_per_language": {"en": 1, "zh": 1}, "min_free_bytes": 0,
              "delete_committed_cache": False, "progress_seconds": 10000}

    def reader(path, batch_size):
        yield rows[path.stem]

    def fetch(source, item, settings, root):
        return Path(item["local_path"])

    return config, reader, fetch, manifest, rows, tmp_path


def run_fixture(fixture, **kwargs):
    config, reader, fetch, _, _, root = fixture
    return worker.run_expansion(config, root, fetch_shard=fetch, batch_reader=reader, **kwargs)


def test_base_prefix_new_eos_dedup_fragment_isolation_and_simplification(fixture):
    path = run_fixture(fixture)
    result = json.loads(path.read_text(encoding="utf-8"))
    config, _, _, parent, rows, _ = fixture
    tokenizer = worker.load_tokenizer(Path(config["parent_manifest"]).parent / "tokenizer.json")
    audit = list(read_jsonl(path.parent / "incremental_audit.jsonl"))
    statuses = {row["id"]: row["status"] for row in audit}
    assert statuses["wiki"] == "excluded_wikimedia_domain"
    assert statuses["old_alias"] == "duplicate_old_or_new"
    assert statuses["mirror"] == "heldout_fragment_match"
    assert statuses["new_en_dup_url"] == "duplicate_old_or_new"
    for lang in worker.LANGUAGES:
        base = Path(config["parent_manifest"]).parent / parent["splits"]["train"][lang]["path"]
        target = path.parent / result["splits"]["train"][lang]["path"]
        assert target.read_bytes().startswith(base.read_bytes())
        emitted = next(row for row in audit if row["language"] == lang and row["status"] == "encoded")
        assert emitted["token_offset"] == parent["splits"]["train"][lang]["token_count"]
        assert emitted["new_token_offset"] == 0
        expected = FixtureNormalizer().convert(rows[lang][-2 if lang == "en" else 0]["text"], lang)
        ids = tokenizer.encode(expected, add_special_tokens=False).ids + [parent["tokenizer"]["eos_id"]]
        assert np.fromfile(target, dtype="<u4")[emitted["token_offset"]:].tolist() == ids
        assert emitted["token_count"] == len(ids)
        assert emitted["text_sha256"] == text_hash(expected)
    assert not (path.parent / "manifest.json").exists()
    assert result["new_token_count"] > 0


@pytest.mark.parametrize("phase", ["parts_fsynced", "outputs_fsynced", "sqlite_committed"])
def test_crash_resume_does_not_duplicate_or_lose_committed_tokens(fixture, phase):
    def fail(current):
        if current == phase:
            raise RuntimeError("fixture crash")
    with pytest.raises(RuntimeError, match="fixture crash"):
        run_fixture(fixture, _failure_hook=fail)
    output = Path(fixture[0]["output_dir"])
    assert not (output / "expansion_manifest.json").exists()
    result = run_fixture(fixture, resume=True)
    audit = list(read_jsonl(result.parent / "incremental_audit.jsonl"))
    assert [row["id"] for row in audit if row["status"] == "encoded"] == ["new_en", "new_zh"]
    connection = sqlite3.connect(output / "dedup.sqlite")
    assert connection.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 2
    connection.close()
    assert not list(output.glob("shard-*.part"))


def test_completed_resume_leaves_publisher_manifest_audit_and_streams_untouched(fixture):
    result = run_fixture(fixture)
    output = result.parent
    (output / "manifest.json").write_text('{"published":true}', encoding="utf-8")
    (output / "document_audit.jsonl").write_text('publisher-owned\n', encoding="utf-8")
    paths = [output / name for name in ("manifest.json", "document_audit.jsonl", "train.en.bin", "train.zh.bin", "incremental_audit.jsonl")]
    before = [(p.stat().st_mtime_ns, sha256_file(p)) for p in paths]
    assert run_fixture(fixture, resume=True) == result
    assert [(p.stat().st_mtime_ns, sha256_file(p)) for p in paths] == before


def test_shortfall_does_not_publish_and_identity_cannot_silently_change(fixture):
    fixture[0]["target_tokens_per_language"] = {"en": 1000000, "zh": 1000000}
    with pytest.raises(ValueError, match="真实分片耗尽"):
        run_fixture(fixture)
    output = Path(fixture[0]["output_dir"])
    assert not (output / "expansion_manifest.json").exists()
    fixture[0]["target_tokens_per_language"]["en"] = 1
    with pytest.raises(ValueError, match="身份已改变"):
        run_fixture(fixture, resume=True)


def test_pinned_hash_failure_prevents_any_new_tokens(fixture):
    item = fixture[0]["sources"][0]["data_files"][0]
    Path(item["local_path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Parquet 大小/SHA256"):
        run_fixture(fixture)
    output = Path(fixture[0]["output_dir"])
    assert not (output / "expansion_manifest.json").exists()


def test_partial_base_prefix_copy_is_replaced_on_resume(fixture, monkeypatch):
    original = worker.shutil.copyfileobj

    def interrupt(reader, writer, length=0):
        writer.write(reader.read(11))
        raise RuntimeError("base copy interrupted")

    monkeypatch.setattr(worker.shutil, "copyfileobj", interrupt)
    with pytest.raises(RuntimeError, match="base copy interrupted"):
        run_fixture(fixture)
    output = Path(fixture[0]["output_dir"])
    assert (output / "base-en.part").stat().st_size == 11
    assert not (output / "train.en.bin").exists()
    monkeypatch.setattr(worker.shutil, "copyfileobj", original)
    result = run_fixture(fixture, resume=True)
    assert not (output / "base-en.part").exists()
    parent = Path(fixture[0]["parent_manifest"]).parent / "train.en.bin"
    assert (result.parent / "train.en.bin").read_bytes().startswith(parent.read_bytes())


def test_output_cannot_target_parent_or_escape_project(fixture):
    config, _, _, _, _, root = fixture
    for path in (root.parent / "outside", Path(config["parent_manifest"]).parent, root):
        invalid = dict(config, output_dir=str(path))
        with pytest.raises(ValueError):
            worker.validate_config(invalid, root)


def test_full_normalizer_provenance_change_rejected(fixture, monkeypatch):
    class Changed(FixtureNormalizer):
        provenance = {"mode": "t2s", "fixture": "changed-dictionary"}
    monkeypatch.setattr(worker, "build_chinese_normalizer", lambda mode: Changed())
    with pytest.raises(ValueError, match="完整 provenance"):
        run_fixture(fixture)


def test_rolling_fingerprints_capture_arbitrary_offsets_and_whitespace():
    original = "An independent heldout paragraph with a long sequence of precise observations and carefully documented experimental conditions."
    hashes = set(worker.window_hashes(worker.fingerprint_text(original)))
    mirror = "prefix!\n" + original.replace(" ", "\n  ").upper() + "suffix"
    assert hashes.intersection(worker.window_hashes(worker.fingerprint_text(mirror)))
    assert not hashes.intersection(worker.window_hashes(worker.fingerprint_text("Completely different educational article. " * 5)))


def test_cache_deletion_only_touches_private_regular_files(tmp_path):
    cache = tmp_path / "private"
    cache.mkdir()
    outside = tmp_path / "keep.txt"
    outside.write_text("keep", encoding="utf-8")
    assert not worker.safe_delete_cache(outside, cache)
    cached = cache / "delete.parquet"
    cached.write_bytes(b"downloaded")
    assert worker.safe_delete_cache(cached, cache)
    assert outside.read_text(encoding="utf-8") == "keep"


def test_real_parquet_reader_requires_pinned_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    path = tmp_path / "real.parquet"
    pq.write_table(pa.table({"text": ["sample"], "id": ["id"], "url": ["https://example.test"]}), path)
    assert list(worker.parquet_batches(path, 1))[0][0]["id"] == "id"
    pq.write_table(pa.table({"text": ["sample"]}), path)
    with pytest.raises(ValueError, match="text/id/url"):
        list(worker.parquet_batches(path, 1))


def test_publisher_preserves_holdouts_and_emits_training_compatible_manifest(fixture):
    spec = importlib.util.spec_from_file_location("publisher_under_test", ROOT / "scripts/publish_expanded_corpus.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    from llm_lab.engine.corpus import BinaryCorpus
    expansion = run_fixture(fixture)
    published = publisher.publish(expansion.parent)
    manifest = json.loads(published.read_text(encoding="utf-8"))
    parent = fixture[3]
    assert manifest["kind"] == "pretrain_token_streams"
    assert manifest["acceptance"]["decoded_hash_samples"] == {"en": 1, "zh": 1}
    for split in ("validation", "test"):
        for language in worker.LANGUAGES:
            assert manifest["splits"][split][language]["sha256"] == parent["splits"][split][language]["sha256"]
    corpus = BinaryCorpus(published, "train", 16)
    x, y = corpus.batch("en", 2, np.random.default_rng(42))
    assert x.shape == y.shape == (2, 16)
    assert publisher.publish(expansion.parent) == published
    before = sha256_file(published)
    assert run_fixture(fixture, resume=True) == expansion
    assert sha256_file(published) == before
