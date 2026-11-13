"""真正验证数据边界、隔离、词表拟合来源与程序偏好，不使用网络。"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest

from llm_lab.data.acquire import (
    MASSIVE_RELEASE_URL, acquire, download_verified, massive_archive_records,
    validate_tar_member, wikipedia_records,
)
from llm_lab.data.common import json_line, read_jsonl, text_hash, write_jsonl
from llm_lab.data.corpus import TokenCorpus
from llm_lab.data.massive import parse_slots, prepare_massive
from llm_lab.data.prepare import clean_documents, prepare_pretrain
from llm_lab.tokenization import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def pretrain_config(tmp_path):
    return {"inputs": [str(ROOT / "examples/pretrain_documents.jsonl")],
            "output_dir": str(tmp_path / "tokens"), "max_docs": 100, "max_bytes": 200000,
            "max_tokens": 20000, "seed": 42,
            "tokenizer": {"vocab_size": 512, "min_frequency": 1}, "quality": {"min_chars": 20}}


def massive_config(tmp_path):
    return {"inputs": [str(ROOT / "examples/massive_fixture.jsonl")],
            "output_dir": str(tmp_path / "tasks"), "max_docs": 100, "max_bytes": 100000,
            "cross_split_duplicate_policy": "keep_evaluation"}


def test_pretrain_is_train_only_and_roundtrips_unseen_unicode(tmp_path):
    path = prepare_pretrain(pretrain_config(tmp_path), tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    originals = list(read_jsonl(ROOT / "examples/pretrain_documents.jsonl"))
    expected_train_hashes = sorted(text_hash(row["text"]) for row in originals if row["split"] == "train")
    assert manifest["tokenizer_training_document_hashes"] == expected_train_hashes
    assert manifest["tokenizer"]["vocab_size"] == 512
    tokenizer = load_tokenizer(path.parent / "tokenizer.json")
    unseen = "Unseen 蝴蝶🦋 walks beyond the training corpus."
    encoded = tokenizer.encode(unseen)
    assert tokenizer.token_to_id("<|unk|>") not in encoded.ids
    assert tokenizer.decode(encoded.ids) == unseen
    # 验证六条流都有 EOS 文档终止，token 计数与二进制实际一致。
    for split, languages in manifest["splits"].items():
        corpus = TokenCorpus(path, split=split, seq_len=16)
        for language, item in languages.items():
            tokens = corpus.get_language_tokens(language)
            assert len(tokens) == item["token_count"] > 16
            assert tokens[-1] == manifest["tokenizer"]["eos_id"]
            x, y = corpus.sample_batch(language, 2, seed=4)
            assert tuple(x.shape) == tuple(y.shape) == (2, 16)
            assert (x[:, 1:] == y[:, :-1]).all()


def test_document_dedup_and_cross_split_leakage(tmp_path):
    text = "The observer carefully writes a measurement in the notebook every morning."
    rows = [{"source_id": "a", "language": "en", "split": "train", "text": text},
            {"source_id": "b", "language": "en", "split": "train", "text": "  " + text + "  "}]
    raw = tmp_path / "docs.jsonl"
    write_jsonl(raw, rows)
    config = {"inputs": [str(raw)], "max_docs": 10, "max_bytes": 10000, "quality": {"min_chars": 10}}
    docs, stats = clean_documents(config, tmp_path)
    assert len(docs) == 1 and stats["duplicates"] == 1
    rows[1]["split"] = "test"
    write_jsonl(raw, rows)
    with pytest.raises(ValueError, match="相同规范化文档跨 split"):
        clean_documents(config, tmp_path)


def test_validation_contents_cannot_change_fitted_tokenizer(tmp_path):
    first_config = pretrain_config(tmp_path)
    first_path = prepare_pretrain(first_config, tmp_path)
    first = json.loads(first_path.read_text(encoding="utf-8"))
    rows = list(read_jsonl(ROOT / "examples/pretrain_documents.jsonl"))
    for row in rows:
        if row["split"] != "train":
            row["text"] = "Entirely different heldout material contains zephyrquartz and 尚未出现在训练文档中的内容。"
            row["text"] += row["source_id"]
    altered = tmp_path / "altered.jsonl"
    write_jsonl(altered, rows)
    second_config = pretrain_config(tmp_path)
    second_config.update({"inputs": [str(altered)], "output_dir": str(tmp_path / "other_tokens")})
    second_path = prepare_pretrain(second_config, tmp_path)
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first["tokenizer"]["sha256"] == second["tokenizer"]["sha256"]


def test_unattainable_vocab_is_reported_without_silently_changing_model_size(tmp_path):
    config = pretrain_config(tmp_path)
    config["tokenizer"]["vocab_size"] = 32000
    with pytest.raises(ValueError, match="实际.*不会静默更改模型参数量"):
        prepare_pretrain(config, tmp_path)


def test_preparation_does_not_read_an_extra_document_after_scan_budget(tmp_path, monkeypatch):
    visited = []

    def rows(path):
        for index in range(100):
            visited.append(index)
            yield {"source_id": str(index), "language": "en", "text": f"A unique sufficiently long original record number {index}."}

    monkeypatch.setattr("llm_lab.data.prepare.read_jsonl", rows)
    _, stats = clean_documents({"inputs": ["unused.jsonl"], "max_docs": 2, "max_bytes": 10000}, tmp_path)
    assert visited == [0, 1] and stats["scanned"] == 2


def test_source_group_cannot_cross_languages_and_splits(tmp_path):
    rows = [{"source_id": "same", "language": "en", "split": "train", "text": "A sufficiently long English record."},
            {"source_id": "same", "language": "zh", "split": "test", "text": "这一段中文有足够的长度用于验证同源隔离。"}]
    path = tmp_path / "docs.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="同源跨 split"):
        clean_documents({"inputs": [str(path)], "max_docs": 10, "max_bytes": 10000,
                         "quality": {"min_chars": 5}}, tmp_path)


def test_pretrain_total_encoded_token_budget_and_checksum(tmp_path):
    config = pretrain_config(tmp_path)
    config.update({"max_tokens": 150, "require_all_streams": False})
    path = prepare_pretrain(config, tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["token_count"] <= 150
    nonempty = [(split, lang, item) for split, langs in manifest["splits"].items()
                for lang, item in langs.items() if item["token_count"]]
    assert nonempty
    split, _, item = nonempty[0]
    token_path = path.parent / item["path"]
    original = token_path.read_bytes()
    token_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="校验和"):
        TokenCorpus(path, split=split, seq_len=2)


def test_acquire_limits_scanned_records_without_overconsumption(tmp_path, monkeypatch):
    consumed = []

    def records(source, cache):
        for index in range(100):
            consumed.append(index)
            yield {"source_id": str(index), "language": "en", "text": "original fixture"}

    monkeypatch.setattr("llm_lab.data.acquire.wikipedia_records", records)
    result = acquire({"sources": [{"type": "wikipedia"}], "output_dir": "raw",
                      "max_docs": 2, "max_bytes": 10000}, tmp_path)
    assert consumed == [0, 1]
    assert len(list(read_jsonl(result))) == 2


def test_acquire_utf8_byte_budget(tmp_path):
    source = tmp_path / "raw.jsonl"
    row = {"source_id": "原始样本", "language": "zh", "text": "少量原创测试文本。"}
    write_jsonl(source, [row, row])
    line_bytes = len(json_line(row).encode("utf-8"))
    result = acquire({"sources": [{"type": "local_jsonl", "path": str(source)}],
                      "output_dir": "out", "max_docs": 10, "max_bytes": line_bytes}, tmp_path)
    assert result.stat().st_size == line_bytes
    assert len(list(read_jsonl(result))) == 1


def test_wikipedia_rejects_unpinned_revision_before_network(tmp_path):
    with pytest.raises(ValueError, match="完整 40 位"):
        list(wikipedia_records({"revision": "main", "snapshot": "20231101", "language": "zh"}, tmp_path))


def wikipedia_file_source(payload=b"original tiny Parquet stand-in"):
    return {"revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "snapshot": "20231101", "language": "en",
            "max_download_bytes": len(payload), "data_files": [
                {"path": "20231101.en/train-00000-of-00041.parquet",
                 "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}]}


def fake_datasets(monkeypatch, calls):
    module = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"id": "42", "text": "A tiny original record.", "url": "https://example.invalid/article", "title": "Fixture"}]

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_wikipedia_explicit_files_use_verified_local_parquet_without_repo_metadata(tmp_path, monkeypatch):
    payload = b"original tiny Parquet stand-in"
    source = wikipedia_file_source(payload)
    cached = tmp_path / "wikipedia" / (source["data_files"][0]["sha256"] + ".parquet")
    cached.parent.mkdir()
    cached.write_bytes(payload)
    calls = []
    fake_datasets(monkeypatch, calls)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("cache hit must not request network"))
    rows = list(wikipedia_records(source, tmp_path))
    assert calls == [(("parquet",), {"data_files": {"train": [str(cached.resolve())]}, "split": "train",
                                    "streaming": True, "cache_dir": str(tmp_path / "datasets")})]
    assert rows[0]["source_id"] == "wikipedia:20231101:en:42"
    assert rows[0]["source_revision"] == source["revision"]


def test_wikipedia_default_repo_path_is_unchanged(tmp_path, monkeypatch):
    source = wikipedia_file_source()
    del source["data_files"]
    calls = []
    fake_datasets(monkeypatch, calls)
    list(wikipedia_records(source, tmp_path))
    assert calls == [(("wikimedia/wikipedia", "20231101.en"), {"split": "train", "revision": source["revision"],
                           "streaming": True, "cache_dir": str(tmp_path / "datasets")})]


@pytest.mark.parametrize("bad_path", [
    "../outside.parquet", "/20231101.en/file.parquet", "20231101.zh/file.parquet", "20240101.en/file.parquet",
    "20231101.en/../file.parquet", "20231101.en/nested/file.parquet", "20231101.en\\file.parquet",
    "20231101.en/file.parquet?revision=main", "20231101.en/%2e%2e.parquet",
    "https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/file.parquet",
])
def test_wikipedia_explicit_file_rejects_other_directory_revision_and_traversal(tmp_path, bad_path):
    source = wikipedia_file_source()
    source["data_files"][0]["path"] = bad_path
    with pytest.raises(ValueError, match="相对 Parquet"):
        list(wikipedia_records(source, tmp_path))
    assert not (tmp_path / "wikipedia").exists()


@pytest.mark.parametrize("endpoint", ["http://hf-mirror.com", "https://user@hf-mirror.com", "https://hf-mirror.com?q=1",
                                       "https://hf-mirror.com/other", "https://hf-mirror.com#fragment"])
def test_wikipedia_explicit_file_endpoint_rejects_unsafe_components(tmp_path, monkeypatch, endpoint):
    source = wikipedia_file_source()
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    with pytest.raises(ValueError, match="HF_ENDPOINT"):
        list(wikipedia_records(source, tmp_path))


def test_wikipedia_explicit_file_source_cap_is_checked_before_download(tmp_path):
    source = wikipedia_file_source()
    item = dict(source["data_files"][0], path="20231101.en/train-00001-of-00041.parquet")
    source["data_files"].append(item)
    with pytest.raises(ValueError, match="总大小超过"):
        list(wikipedia_records(source, tmp_path))
    assert not (tmp_path / "wikipedia").exists()


def test_wikipedia_corrupt_cache_is_rejected_before_loader_or_network(tmp_path, monkeypatch):
    payload = b"original tiny Parquet stand-in"
    source = wikipedia_file_source(payload)
    cached = tmp_path / "wikipedia" / (source["data_files"][0]["sha256"] + ".parquet")
    cached.parent.mkdir()
    cached.write_bytes(b"!" + payload[1:])
    calls = []
    fake_datasets(monkeypatch, calls)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("corrupt cache must not be silently redownloaded"))
    with pytest.raises(ValueError, match="缓存分片大小或 SHA256 不匹配"):
        list(wikipedia_records(source, tmp_path))
    assert not calls


class FakeWikipediaResponse(io.BytesIO):
    def __init__(self, payload, *, with_length=True):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))} if with_length else {}

    def geturl(self):
        return "https://cas-bridge.xethub.hf.co/original-test-fixture"


def test_wikipedia_download_is_https_pinned_bounded_and_atomically_cached(tmp_path, monkeypatch):
    payload = b"small original HTTP fixture"
    source = wikipedia_file_source(payload)
    calls = []
    fake_datasets(monkeypatch, calls)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    monkeypatch.setattr("llm_lab.data.acquire._hub_download_headers", lambda: {"User-Agent": "huggingface_hub/offline-test"})
    requests = []

    def fake_open(request, timeout):
        requests.append((request, timeout))
        return FakeWikipediaResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    list(wikipedia_records(source, tmp_path))
    request, timeout = requests[0]
    assert request.full_url == ("https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/" + source["revision"]
                                + "/20231101.en/train-00000-of-00041.parquet")
    assert timeout == 30
    assert request.get_header("User-agent") == "huggingface_hub/offline-test"
    cached = tmp_path / "wikipedia" / (source["data_files"][0]["sha256"] + ".parquet")
    assert cached.read_bytes() == payload and not cached.with_suffix(".parquet.part").exists()
    assert calls[0][1]["data_files"] == {"train": [str(cached.resolve())]}


def test_wikipedia_oversize_download_cleans_only_owned_partial(tmp_path, monkeypatch):
    payload = b"short original payload"
    source = wikipedia_file_source(payload)
    fake_datasets(monkeypatch, [])
    monkeypatch.setattr("llm_lab.data.acquire._hub_download_headers", lambda: {})
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeWikipediaResponse(payload + b"extra", with_length=False))
    with pytest.raises(ValueError, match="超过声明大小"):
        list(wikipedia_records(source, tmp_path))
    cached = tmp_path / "wikipedia" / (source["data_files"][0]["sha256"] + ".parquet")
    assert not cached.exists() and not cached.with_suffix(".parquet.part").exists()
    partial = cached.with_suffix(".parquet.part")
    partial.write_bytes(b"preexisting partial from another run")
    with pytest.raises(FileExistsError):
        list(wikipedia_records(source, tmp_path))
    assert partial.read_bytes() == b"preexisting partial from another run"


def test_wikipedia_download_sha_mismatch_never_becomes_complete_cache(tmp_path, monkeypatch):
    source = wikipedia_file_source(b"good fixture")
    fake_datasets(monkeypatch, [])
    monkeypatch.setattr("llm_lab.data.acquire._hub_download_headers", lambda: {})
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeWikipediaResponse(b"evil fixture"))
    with pytest.raises(ValueError, match="实际大小或 SHA256 不匹配"):
        list(wikipedia_records(source, tmp_path))
    assert list((tmp_path / "wikipedia").iterdir()) == []


def test_massive_annotation_reconstruction_and_repeated_slots():
    slots = parse_slots("from Lake Town to Lake Town", "from [place_name : Lake Town] to [place_name : Lake Town]")
    assert slots == [{"slot": "place_name", "value": "Lake Town"}] * 2
    assert parse_slots("查看日程", "查看日程") == []
    with pytest.raises(ValueError, match="不能无损还原"):
        parse_slots("六点起床", "[time : 七点]起床")


def test_massive_official_splits_and_programmatic_preferences(tmp_path):
    path = prepare_massive(massive_config(tmp_path), tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["preference_provenance"]["human_preference"] is False
    ids = {}
    for split, info in manifest["splits"].items():
        sft = list(read_jsonl(path.parent / info["sft"]))
        dpo = list(read_jsonl(path.parent / info["dpo"]))
        ids[split] = {row["source_id"] for row in sft}
        assert {row["language"] for row in sft} == {"en", "zh"}
        for supervised, preference in zip(sft, dpo):
            assert set(supervised) == {"source_id", "language", "prompt", "completion"}
            assert set(preference) == {"source_id", "language", "prompt", "chosen", "rejected"}
            assert supervised["completion"] == preference["chosen"]
            gold, wrong = json.loads(preference["chosen"]), json.loads(preference["rejected"])
            assert set(wrong) == {"intent", "slots"}
            assert gold["intent"] != wrong["intent"] and gold["slots"] == wrong["slots"]
    assert ids["train"].isdisjoint(ids["validation"] | ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
    assert ids["test"] == {"massive:fixture-6", "massive:fixture-7"}


def test_massive_cross_language_partition_conflict_is_not_silently_resplit(tmp_path):
    rows = list(read_jsonl(ROOT / "examples/massive_fixture.jsonl"))
    rows[1]["partition"] = "test"
    raw = tmp_path / "bad.jsonl"
    write_jsonl(raw, rows)
    config = massive_config(tmp_path)
    config["inputs"] = [str(raw)]
    with pytest.raises(ValueError, match="跨语言源 ID 分区冲突"):
        prepare_massive(config, tmp_path)


def test_massive_cross_split_duplicate_preserves_test(tmp_path):
    rows = list(read_jsonl(ROOT / "examples/massive_fixture.jsonl"))
    rows[12]["utt"] = rows[0]["utt"]
    rows[12]["annot_utt"] = rows[0]["annot_utt"]
    raw = tmp_path / "duplicate.jsonl"
    write_jsonl(raw, rows)
    config = massive_config(tmp_path)
    config["inputs"] = [str(raw)]
    manifest_path = prepare_massive(config, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["statistics"]["cross_split_duplicate_dropped"] == 2
    train = list(read_jsonl(manifest_path.parent / "sft/train.jsonl"))
    assert not any(row["source_id"] == "massive:fixture-0" for row in train)


@pytest.mark.parametrize("name", ["../escape.jsonl", "/absolute.jsonl", "C:/escape.jsonl", "a/../../bad", "..\\bad"])
def test_archive_paths_never_escape(name):
    with pytest.raises(ValueError, match="不安全"):
        validate_tar_member(tarfile.TarInfo(name))


def test_archive_rejects_symlinks_and_limits_uncompressed_size(tmp_path):
    member = tarfile.TarInfo("1.1/data/en-US.jsonl")
    member.type = tarfile.SYMTYPE
    with pytest.raises(ValueError, match="符号链接"):
        validate_tar_member(member)
    archive_path = tmp_path / "tiny.tar.gz"
    payload = b"x" * 20
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("1.1/data/en-US.jsonl")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="大小超过"):
        list(massive_archive_records(archive_path, {"max_uncompressed_bytes": 10, "max_member_bytes": 100}))


class FakeResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def geturl(self):
        return MASSIVE_RELEASE_URL


def test_first_download_records_observed_fingerprint_then_cache_reuse_verifies(tmp_path, monkeypatch):
    payload = b"small original download fixture, not a real release"
    calls = []

    def fake_open(*args, **kwargs):
        calls.append(1)
        return FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    source = {"url": MASSIVE_RELEASE_URL, "expected_sha256": None, "max_download_bytes": 1000}
    path = download_verified(source, tmp_path)
    lock = json.loads((tmp_path / "massive-1.1.lock.json").read_text(encoding="utf-8"))
    assert lock["observed_sha256"] == hashlib.sha256(payload).hexdigest()
    assert lock["independent_checksum_verified"] is False
    assert download_verified(source, tmp_path) == path and len(calls) == 1
    path.write_bytes(payload[:-1] + b"!")
    with pytest.raises(ValueError, match="固定指纹不匹配"):
        download_verified(source, tmp_path)


def test_supplied_expected_digest_must_match_and_preexisting_partial_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse(b"fixture"))
    source = {"url": MASSIVE_RELEASE_URL, "expected_sha256": "0" * 64, "max_download_bytes": 1000}
    with pytest.raises(ValueError, match="SHA256 不匹配"):
        download_verified(source, tmp_path)
    assert not (tmp_path / "massive-1.1.tar.gz").exists()
    assert not (tmp_path / "massive-1.1.tar.gz.part").exists()
    partial = tmp_path / "massive-1.1.tar.gz.part"
    partial.write_bytes(b"preexisting unmanaged partial")
    with pytest.raises(FileExistsError):
        download_verified(source, tmp_path)
    assert partial.read_bytes() == b"preexisting unmanaged partial"
