"""本地原创 fixture 验证流式制备的容量、冻结隔离和失败发布边界。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from llm_lab.data.common import assigned_split, read_jsonl, sha256_file, write_jsonl
from llm_lab.data.prepare import prepare_pretrain
from llm_lab.data.streaming_prepare import prepare_streaming, resolve_inputs
from llm_lab.tokenization import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = {"train": .34, "validation": .33, "test": .33}


def source_for(split, tag):
    for number in range(1000):
        candidate = f"stream-test:{tag}:{number}"
        if assigned_split(candidate, 42, FRACTIONS) == split:
            return candidate
    raise AssertionError("无法生成 fixture 源 ID")


def rows_for_streaming():
    rows = list(read_jsonl(ROOT / "examples/pretrain_documents.jsonl"))
    for number, row in enumerate(rows):
        row["source_id"] = source_for(row["split"], f"fixture-{number}")
    return rows


def config_for(tmp_path, rows=None, name="result"):
    raw = tmp_path / f"{name}.jsonl"
    write_jsonl(raw, rows if rows is not None else rows_for_streaming())
    return {"inputs": [str(raw)], "output_dir": str(tmp_path / name), "seed": 42,
            "split_fractions": FRACTIONS, "max_docs": 1000, "max_input_bytes": 1000000,
            "max_clean_bytes": 1000000, "min_free_disk_bytes": 0, "min_stream_tokens": 17,
            "max_train_tokens_per_language": {"en": 100000, "zh": 100000},
            "tokenizer": {"vocab_size": 512, "min_frequency": 1, "sample_max_bytes": 100000},
            "quality": {"min_chars": 20}}


@pytest.fixture
def frozen(tmp_path):
    rows = list(read_jsonl(ROOT / "examples/pretrain_documents.jsonl"))
    old = prepare_pretrain({"inputs": [str(ROOT / "examples/pretrain_documents.jsonl")],
                            "output_dir": str(tmp_path / "old"), "max_docs": 100, "max_bytes": 200000,
                            "max_tokens": 20000, "tokenizer": {"vocab_size": 512, "min_frequency": 1},
                            "quality": {"min_chars": 20}}, tmp_path)
    return old, rows


def frozen_config(tmp_path, frozen, rows=None, name="expanded"):
    old, originals = frozen
    config = config_for(tmp_path, originals if rows is None else rows, name)
    config["frozen_holdout_manifest"] = str(old)
    config["tokenizer"] = {"path": str(old.parent / "tokenizer.json"), "vocab_size": 512}
    return config


def test_streams_have_exact_sizes_hashes_and_document_eos(tmp_path):
    config = config_for(tmp_path)
    path = prepare_streaming(config, tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(path.parent / "tokenizer.json")
    for split, languages in manifest["splits"].items():
        for language, item in languages.items():
            expected = []
            for row in rows_for_streaming():
                if row["split"] == split and row["language"] == language:
                    expected += tokenizer.encode(row["text"], add_special_tokens=False).ids + [manifest["tokenizer"]["eos_id"]]
            binary = path.parent / item["path"]
            assert np.fromfile(binary, dtype="<u4").tolist() == expected
            assert item["token_count"] == len(expected) > 16
            assert item["sha256"] == sha256_file(binary)
    assert manifest["input_provenance"][0]["sha256"] == sha256_file(Path(config["inputs"][0]))
    assert manifest["tokenizer"]["provenance"]["mode"] == "bounded_train_only_sample"


def test_heldout_edits_never_fit_tokenizer(tmp_path):
    rows = rows_for_streaming()
    first = prepare_streaming(config_for(tmp_path, rows, "first"), tmp_path)
    for row in rows:
        if row["split"] != "train":
            row["text"] = "Holdoutonlyzephyr unique material 尚未出现的测试词语。" + row["source_id"]
    second = prepare_streaming(config_for(tmp_path, rows, "second"), tmp_path)
    assert sha256_file(first.parent / "tokenizer.json") == sha256_file(second.parent / "tokenizer.json")
    samples = list(read_jsonl(first.parent / "tokenizer_sample.en.jsonl")) + list(read_jsonl(first.parent / "tokenizer_sample.zh.jsonl"))
    assert all(row["split"] == "train" for row in samples)


def test_freeze_copies_holdouts_and_blocks_changed_id_and_normalized_alias(tmp_path, frozen):
    old, rows = frozen
    old_test = next(row for row in rows if row["split"] == "test")
    changed = dict(old_test, text="A modified version of the same heldout source must remain protected forever.")
    alias = source_for("train", "old-alias")
    extras = [{"source_id": alias, "language": "en", "text": "Unique sibling text must also be blocked when its Chinese sibling aliases holdout."},
              {"source_id": alias, "language": "zh", "text": "  " + old_test["text"] + "  "}, changed]
    config = frozen_config(tmp_path, frozen, rows + extras)
    new = prepare_streaming(config, tmp_path)
    for split in ("validation", "test"):
        for language in ("en", "zh"):
            assert (new.parent / f"{split}.{language}.bin").read_bytes() == (old.parent / f"{split}.{language}.bin").read_bytes()
            assert not (new.parent / f"{split}.{language}.bin").is_symlink()
    audit = list(read_jsonl(new.parent / "document_audit.jsonl"))
    aliases = [row for row in audit if row["source_id"] == alias]
    assert len(aliases) == 2 and all(row["split"] == "test" for row in aliases)
    assert all(row["status"] == "excluded_protected_source_group" for row in aliases)
    # 隔离账本可以继续冻结；不能因为曾经被排除的 train 别名导致下一轮失效。
    second_config = frozen_config(tmp_path, (new, rows), rows, name="next_expansion")
    prepare_streaming(second_config, tmp_path)


@pytest.mark.parametrize("reverse", [False, True])
def test_cross_split_drop_is_source_group_and_order_independent(tmp_path, frozen, reverse):
    old, rows = frozen
    common = "An original conflict sentence whose duplicate in validation excludes the entire training source group."
    sibling = "A second original sibling sentence propagates the same protection to an earlier duplicate alias."
    a, b, c = source_for("train", "a"), source_for("validation", "b"), source_for("train", "c")
    extras = [{"source_id": a, "language": "en", "text": common},
              {"source_id": a, "language": "zh", "text": sibling},
              {"source_id": c, "language": "zh", "text": sibling},
              {"source_id": b, "language": "en", "text": common}]
    if reverse:
        extras.reverse()
    config = frozen_config(tmp_path, frozen, rows + extras)
    config["cross_split_duplicate_policy"] = "drop_conflicting_train"
    path = prepare_streaming(config, tmp_path)
    audit = list(read_jsonl(path.parent / "document_audit.jsonl"))
    assert all(row["split"] == "validation" for row in audit if row["source_id"] in (a, c))
    # 原先 train 不全部排除；它们的 token 与旧 train 完全相同。
    for language in ("en", "zh"):
        assert (path.parent / f"train.{language}.bin").read_bytes() == (old.parent / f"train.{language}.bin").read_bytes()


def test_default_cross_split_duplicate_fails_without_publishing(tmp_path):
    rows = rows_for_streaming()
    rows.append(dict(rows[0], source_id=source_for("test", "bad-duplicate"), split="test"))
    config = config_for(tmp_path, rows)
    with pytest.raises(ValueError, match="规范化文档跨 split"):
        prepare_streaming(config, tmp_path)
    assert not (Path(config["output_dir"]) / "manifest.json").exists()


def test_explicit_train_cannot_override_stable_partition(tmp_path):
    rows = rows_for_streaming()
    rows[0]["source_id"] = source_for("test", "override")
    with pytest.raises(ValueError, match="不能覆盖分区"):
        prepare_streaming(config_for(tmp_path, rows), tmp_path)


@pytest.mark.parametrize("kind", ["missing", "sha"])
def test_fixed_tokenizer_failure_never_retrains(tmp_path, frozen, kind):
    config = frozen_config(tmp_path, frozen)
    if kind == "missing":
        config["tokenizer"]["path"] = str(tmp_path / "absent.json")
    else:
        config["tokenizer"]["sha256"] = "0" * 64
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_streaming(config, tmp_path)
    assert not (Path(config["output_dir"]) / "manifest.json").exists()
    assert not (Path(config["output_dir"]) / "tokenizer_sample.en.jsonl").exists()


def test_insufficient_language_fails_and_refuses_implicit_resume(tmp_path, frozen):
    config = frozen_config(tmp_path, frozen)
    config["min_train_tokens_per_language"] = {"en": 99999, "zh": 17}
    with pytest.raises(ValueError, match="容量验收未通过：train/en"):
        prepare_streaming(config, tmp_path)
    output = Path(config["output_dir"])
    assert not (output / "manifest.json").exists()
    state = json.loads((output / "preparation_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed" and state["phase"] == "capacity_validation"
    assert 17 < state["token_counts"]["train"]["en"] < 99999
    with pytest.raises(FileExistsError, match="不支持.*恢复"):
        prepare_streaming(config, tmp_path)


def test_low_disk_prevents_any_publication(tmp_path, monkeypatch):
    monkeypatch.setattr("llm_lab.data.streaming_prepare.shutil.disk_usage", lambda _: type("Usage", (), {"free": 1})())
    config = config_for(tmp_path)
    with pytest.raises(OSError, match="安全水位"):
        prepare_streaming(config, tmp_path)
    assert not Path(config["output_dir"]).exists()


def test_sorted_globs_and_max_docs_do_not_read_extra_record(tmp_path, monkeypatch):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")
    assert resolve_inputs({"input_globs": [str(tmp_path / "*.jsonl")]}, tmp_path) == [first, second]
    from llm_lab.data import streaming_prepare as module
    index = module._open_index(tmp_path / "index.sqlite")
    visited = []
    def generated(_):
        for number in range(3):
            visited.append(number)
            yield {"source_id": source_for("train", str(number)), "language": "en", "text": "A unique complete document for the bounded scanner. " + str(number)}
    monkeypatch.setattr(module, "read_jsonl", generated)
    with (tmp_path / "audit.jsonl").open("w", encoding="utf-8") as audit:
        module._index_inputs({"max_docs": 2, "max_clean_bytes": 10000, "split_fractions": FRACTIONS}, [first, second], tmp_path,
                             index, audit, module.DiskGuard(tmp_path, 0))
    index.close()
    assert visited == [0, 1]


def test_acquisition_identity_and_raw_fingerprints_are_bound(tmp_path):
    revision = "a" * 40
    rows = rows_for_streaming()
    for row in rows:
        row["source_revision"] = revision
    config = config_for(tmp_path, rows)
    raw = Path(config["inputs"][0])
    upstream = tmp_path / "acquisition_manifest.json"
    acquisition = {"kind": "streaming_wikipedia_acquisition", "status": "completed", "revision": revision,
                   "repository": "wikimedia/wikipedia", "snapshot": "20231101",
                   "files": [{"path": raw.name, "sha256": sha256_file(raw), "size_bytes": raw.stat().st_size,
                              "source_revision": revision, "source": {"sha256": "b" * 64, "size_bytes": 123}}]}
    upstream.write_text(json.dumps(acquisition), encoding="utf-8")
    config["acquisition_manifest"] = str(upstream)
    path = prepare_streaming(config, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["acquisition_provenance"]["sha256"] == sha256_file(upstream)
    other = copy.deepcopy(config)
    other["output_dir"] = str(tmp_path / "mismatch")
    rows[0]["source_revision"] = "c" * 40
    write_jsonl(raw, rows)
    acquisition["files"][0].update(sha256=sha256_file(raw), size_bytes=raw.stat().st_size)
    upstream.write_text(json.dumps(acquisition), encoding="utf-8")
    with pytest.raises(ValueError, match="source_revision"):
        prepare_streaming(other, tmp_path)


def test_training_token_cap_never_truncates_document_or_eos(tmp_path, frozen):
    old, rows = frozen
    tokenizer = load_tokenizer(old.parent / "tokenizer.json")
    english = [row for row in rows if row["split"] == "train" and row["language"] == "en"]
    cap = len(tokenizer.encode(english[0]["text"]).ids) + 4
    config = frozen_config(tmp_path, frozen)
    config["max_train_tokens_per_language"]["en"] = cap
    path = prepare_streaming(config, tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = []
    for row in english:
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids + [manifest["tokenizer"]["eos_id"]]
        if len(expected) + len(ids) <= cap:
            expected += ids
    actual = np.fromfile(path.parent / "train.en.bin", dtype="<u4").tolist()
    assert actual == expected and len(actual) <= cap
    assert actual[-1] == manifest["tokenizer"]["eos_id"]
    assert manifest["statistics"]["encoding"]["over_token_cap_train_en"] > 0


TRADITIONAL = str.maketrans({"学": "學", "书": "書", "观": "觀", "图": "圖", "测": "測", "试": "試",
                           "电": "電", "脑": "腦", "与": "與", "台": "臺", "语": "語", "种": "種", "风": "風", "这": "這"})
SIMPLIFIED = str.maketrans({value: key for key, value in {chr(key): value for key, value in TRADITIONAL.items()}.items()})


@pytest.fixture
def test_normalizer(monkeypatch):
    # 这里只隔离外部 C++ 依赖来验证流水线顺序；真实 OpenCC 由独立模块测试及远程核验。
    class FixtureNormalizer:
        def __init__(self, mode):
            self.provenance = {"mode": mode, "implementation": "fixture_character_map"}
        def convert(self, text, language="zh"):
            return text.translate(SIMPLIFIED) if language == "zh" and self.provenance["mode"] == "t2s" else text
    monkeypatch.setattr("llm_lab.data.text_normalization.build_chinese_normalizer", FixtureNormalizer)
    return FixtureNormalizer("t2s")


def legacy_traditional(tmp_path):
    rows = list(read_jsonl(ROOT / "examples/pretrain_documents.jsonl"))
    for row in rows:
        if row["language"] == "zh":
            row["text"] = row["text"].translate(TRADITIONAL)
    raw = tmp_path / "legacy_traditional.jsonl"
    write_jsonl(raw, rows)
    old = prepare_pretrain({"inputs": [str(raw)], "output_dir": str(tmp_path / "old_traditional"),
                            "max_docs": 100, "max_bytes": 200000, "max_tokens": 20000,
                            "tokenizer": {"vocab_size": 512, "min_frequency": 1}, "quality": {"min_chars": 20}}, tmp_path)
    return old, rows, raw


def test_t2s_precedes_hash_dedup_and_rebuilds_all_chinese_holdouts(tmp_path, test_normalizer):
    old, rows, raw = legacy_traditional(tmp_path)
    heldout = next(row for row in rows if row["split"] == "test" and row["language"] == "zh")
    alias = {"source_id": source_for("train", "simplified-heldout-alias"), "language": "zh",
             "text": test_normalizer.convert(heldout["text"])}
    fresh = {"source_id": source_for("train", "fresh-simplified"), "language": "zh",
             "text": "學校圖書館與觀測站共同記錄氣象資料，學生整理實驗結果，並比較不同時間的測量。這些原創文字用來確認繁簡去重的先後順序。"}
    duplicate = dict(fresh, source_id=source_for("train", "fresh-traditional-alias"),
                     text=test_normalizer.convert(fresh["text"]))
    config = frozen_config(tmp_path, (old, rows), rows + [alias, fresh, duplicate])
    config.update(chinese_normalization="t2s", frozen_holdout_text_inputs=[str(raw)])
    path = prepare_streaming(config, tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(path.parent / "tokenizer.json")
    assert manifest["tokenizer"]["sha256"] == sha256_file(old.parent / "tokenizer.json")
    for row in read_jsonl(path.parent / "cleaned.jsonl"):
        if row["language"] == "zh":
            assert row["text"] == test_normalizer.convert(row["text"])
    for split in ("validation", "test"):
        assert (path.parent / f"{split}.en.bin").read_bytes() == (old.parent / f"{split}.en.bin").read_bytes()
        expected = []
        for row in read_jsonl(old.parent / "documents.jsonl"):
            if row["split"] == split and row["language"] == "zh":
                expected += tokenizer.encode(test_normalizer.convert(row["text"]), add_special_tokens=False).ids + [manifest["tokenizer"]["eos_id"]]
        assert np.fromfile(path.parent / f"{split}.zh.bin", dtype="<u4").tolist() == expected
        assert (path.parent / f"{split}.zh.bin").read_bytes() != (old.parent / f"{split}.zh.bin").read_bytes()
    audit = list(read_jsonl(path.parent / "document_audit.jsonl"))
    assert all(row["split"] == "test" for row in audit if row["source_id"] == alias["source_id"])
    assert manifest["statistics"]["cleaning"]["duplicate_content"] == 1
    assert manifest["rebuilt_zh_holdout_parent"]["parent_token_streams"]["test"]["sha256"] == sha256_file(old.parent / "test.zh.bin")
    next_config = frozen_config(tmp_path, (path, rows), rows, "next_t2s")
    next_config["chinese_normalization"] = "t2s"
    next_path = prepare_streaming(next_config, tmp_path)
    assert (next_path.parent / "test.zh.bin").read_bytes() == (path.parent / "test.zh.bin").read_bytes()


def test_t2s_requires_text_for_audit_only_holdout_and_protects_its_alias(tmp_path, test_normalizer):
    old, rows, raw = legacy_traditional(tmp_path)
    hidden = {"source_id": "legacy:unemitted:test", "language": "zh", "split": "test",
              "text": "這是一段當年因為預算沒有寫入二進制的測試資料，學校圖書館仍然需要保護這份記錄。"}
    from llm_lab.data.common import json_line, text_hash
    with (old.parent / "document_audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json_line({key: hidden[key] for key in ("source_id", "language", "split")} | {"text_sha256": text_hash(hidden["text"])}))
    alias = dict(hidden, source_id=source_for("train", "hidden-alias"), split="train", text=test_normalizer.convert(hidden["text"]))
    config = frozen_config(tmp_path, (old, rows), rows + [alias], "missing_heldout_text")
    config["chinese_normalization"] = "t2s"
    with pytest.raises(ValueError, match="缺少原始文本"):
        prepare_streaming(config, tmp_path)
    extra = tmp_path / "hidden_original.jsonl"
    write_jsonl(extra, [hidden])
    config.update(output_dir=str(tmp_path / "protected_hidden"), frozen_holdout_text_inputs=[str(raw), str(extra)])
    path = prepare_streaming(config, tmp_path)
    assert all(row["split"] == "test" for row in read_jsonl(path.parent / "document_audit.jsonl") if row["source_id"] == alias["source_id"])


def test_t2s_frozen_validation_test_collision_fails(tmp_path, test_normalizer):
    old, rows, _ = legacy_traditional(tmp_path)
    held = [row for row in rows if row["language"] == "zh" and row["split"] != "train"]
    held[-1]["text"] = test_normalizer.convert(held[0]["text"])
    raw = tmp_path / "collision_original.jsonl"
    write_jsonl(raw, rows)
    collision = prepare_pretrain({"inputs": [str(raw)], "output_dir": str(tmp_path / "collision_old"),
                                  "max_docs": 100, "max_bytes": 200000, "max_tokens": 20000,
                                  "tokenizer": {"vocab_size": 512, "min_frequency": 1}, "quality": {"min_chars": 20}}, tmp_path)
    config = frozen_config(tmp_path, (collision, rows), rows)
    config.update(chinese_normalization="t2s", frozen_holdout_text_inputs=[str(raw)])
    with pytest.raises(ValueError, match="冻结 validation/test 出现相同文档"):
        prepare_streaming(config, tmp_path)


def test_t2s_rejects_tampered_old_emitted_text(tmp_path, test_normalizer):
    old, rows, raw = legacy_traditional(tmp_path)
    documents = list(read_jsonl(old.parent / "documents.jsonl"))
    document = next(row for row in documents if row["split"] == "test" and row["language"] == "zh")
    document["text"] += "這是被改動的測試文字。"
    write_jsonl(old.parent / "documents.jsonl", documents)
    config = frozen_config(tmp_path, (old, rows), rows)
    config.update(chinese_normalization="t2s", frozen_holdout_text_inputs=[str(raw)])
    with pytest.raises(ValueError, match="未匹配原审计 hash"):
        prepare_streaming(config, tmp_path)
    assert not (Path(config["output_dir"]) / "manifest.json").exists()


def test_fixed_point_normalizer_applies_before_new_hashes_and_frozen_reencoding(tmp_path, monkeypatch):
    from llm_lab.data.text_normalization import ChineseNormalizer
    rows = list(read_jsonl(ROOT / "examples/pretrain_documents.jsonl"))
    heldout = next(row for row in rows if row["split"] == "test" and row["language"] == "zh")
    heldout["text"] += "古文記錄中的浮於潛，需要先轉換字形才能再次命中詞組。"
    raw = tmp_path / "staged_original.jsonl"
    write_jsonl(raw, rows)
    old = prepare_pretrain({"inputs": [str(raw)], "output_dir": str(tmp_path / "staged_old"),
                            "max_docs": 100, "max_bytes": 200000, "max_tokens": 20000,
                            "tokenizer": {"vocab_size": 512, "min_frequency": 1}, "quality": {"min_chars": 20}}, tmp_path)
    class StagedConverter:
        def convert(self, text):
            return text.replace("浮於潜", "浮于潜").replace("潛", "潜")
    def build(mode):
        return ChineseNormalizer(mode, StagedConverter(), {"mode": mode, "normalization_algorithm": "bounded_fixed_point", "max_passes": 8})
    monkeypatch.setattr("llm_lab.data.text_normalization.build_chinese_normalizer", build)
    fresh = {"source_id": source_for("train", "staged-new"), "language": "zh",
             "text": "新訓練文檔包含浮於潛這個詞組，用來確認多輪收斂在 hash 與去重之前完成。"}
    config = frozen_config(tmp_path, (old, rows), rows + [fresh])
    config.update(chinese_normalization="t2s", frozen_holdout_text_inputs=[str(raw)])
    path = prepare_streaming(config, tmp_path)
    from llm_lab.data.common import text_hash
    expected_text = fresh["text"].replace("浮於潛", "浮于潜")
    new = next(row for row in read_jsonl(path.parent / "cleaned.jsonl") if row["source_id"] == fresh["source_id"])
    assert new["text"] == expected_text and new["text_sha256"] == text_hash(expected_text)
    emitted = next(row for row in read_jsonl(path.parent / "documents.jsonl") if row["source_id"] == heldout["source_id"])
    assert "浮于潜" in emitted["text"] and "浮於" not in emitted["text"]
    assert json.loads(path.read_text(encoding="utf-8"))["chinese_normalization"]["normalization_algorithm"] == "bounded_fixed_point"


def test_encoding_batch_one_and_sixty_four_produce_identical_artifacts(tmp_path):
    rows = []
    for repeat in range(6):
        for number, original in enumerate(rows_for_streaming()):
            rows.append(dict(original, source_id=source_for(original["split"], f"batch-{repeat}-{number}"),
                             text=original["text"] + f" Unique document marker {repeat}/{number}."))
    manifests = []
    for batch_size in (1, 64):
        config = config_for(tmp_path, rows, f"batch_{batch_size}")
        config.update(encoding_batch_size=batch_size, encoding_batch_max_bytes=4 * 1024 * 1024)
        config["max_train_tokens_per_language"] = {"en": 1500, "zh": 1500}
        path = prepare_streaming(config, tmp_path)
        manifests.append((path, json.loads(path.read_text(encoding="utf-8"))))
    first_path, first = manifests[0]
    second_path, second = manifests[1]
    assert first["tokenizer"]["sha256"] == second["tokenizer"]["sha256"]
    assert first["splits"] == second["splits"]
    assert first["statistics"] == second["statistics"]
    for split, languages in first["splits"].items():
        for language in languages:
            name = f"{split}.{language}.bin"
            assert (first_path.parent / name).read_bytes() == (second_path.parent / name).read_bytes()
    assert (first_path.parent / "documents.jsonl").read_bytes() == (second_path.parent / "documents.jsonl").read_bytes()


def run_recorded_encoding(tmp_path, records, **options):
    """用确定的 Token 序列检查批边界、预筛及落盘语义，不依赖 BPE 学习结果。"""
    from types import SimpleNamespace
    from llm_lab.data import streaming_prepare as module
    from llm_lab.data.common import text_hash
    class RecordingTokenizer:
        def __init__(self):
            self.calls = []
            self.ids = {text: ids for _, _, text, ids, _ in records}
        def encode_batch(self, texts, add_special_tokens):
            assert add_special_tokens is False
            self.calls.append(list(texts))
            return [SimpleNamespace(ids=list(self.ids[text])) for text in texts]
    tokenizer = RecordingTokenizer()
    index = module._open_index(tmp_path / "index.sqlite")
    rows = []
    for number, (split, language, text, _, active) in enumerate(records):
        source_id = f"encoding-unit:{number}"
        digest = text_hash(text)
        rows.append({"source_id": source_id, "split": split, "language": language,
                     "text": text, "text_sha256": digest})
        index.execute("INSERT INTO sources(source_id,split,blocked) VALUES (?,?,?)", (source_id, split, int(not active)))
        index.execute("INSERT INTO contents VALUES (?,?,?)", (digest, split, source_id))
    write_jsonl(tmp_path / "cleaned.jsonl", rows)
    config = {"max_train_tokens_per_language": {"en": 1000, "zh": 1000}, "min_stream_tokens": 1, **options}
    try:
        result = module._encode(config, tmp_path, tokenizer, {"eos_id": 2}, index, None, module.DiskGuard(tmp_path, 0))
    finally:
        index.close()
    return result, tokenizer.calls


@pytest.mark.parametrize("batch_size,max_bytes,expected", [
    (4, 5, [["éé"], ["中", "zz"], ["界界"], ["x", "y", "z"]]),
    (2, 100, [["éé", "中"], ["zz", "界界"], ["x", "y"], ["z"]]),
])
def test_encoding_count_utf8_byte_bounds_oversize_singleton_and_final_flush(tmp_path, batch_size, max_bytes, expected):
    records = [("train", "en", "éé", [10], True),
               ("train", "en", "inactive", [99], False),
               ("train", "zh", "中", [11], True),
               ("train", "en", "zz", [12], True),
               ("validation", "en", "界界", [13], True),
               ("validation", "zh", "x", [14], True),
               ("test", "en", "y", [15], True),
               ("test", "zh", "z", [16], True)]
    (splits, statistics), calls = run_recorded_encoding(tmp_path, records, encoding_batch_size=batch_size,
                                                       encoding_batch_max_bytes=max_bytes)
    assert calls == expected
    assert statistics["excluded_after_group_audit"] == 1
    assert np.fromfile(tmp_path / "train.en.bin", dtype="<u4").tolist() == [10, 2, 12, 2]
    assert all(item["token_count"] > 0 for languages in splits.values() for item in languages.values())


def test_batch_cap_skips_whole_long_document_then_accepts_later_short_document(tmp_path):
    records = [("train", "en", "first", [11, 12], True),
               ("train", "en", "too long", [21] * 8, True),
               ("train", "en", "last", [13], True),
               ("train", "zh", "中文", [14], True),
               ("validation", "en", "validation en", [15], True),
               ("validation", "zh", "validation zh", [16], True),
               ("test", "en", "test en", [17], True),
               ("test", "zh", "test zh", [18], True)]
    (splits, statistics), calls = run_recorded_encoding(tmp_path, records,
        max_train_tokens_per_language={"en": 5, "zh": 1000})
    assert len(calls) == 1
    assert np.fromfile(tmp_path / "train.en.bin", dtype="<u4").tolist() == [11, 12, 2, 13, 2]
    assert splits["train"]["en"]["token_count"] == 5
    assert statistics["documents_train_en"] == 2 and statistics["over_token_cap_train_en"] == 1


@pytest.mark.parametrize("field,value", [("encoding_batch_size", 0), ("encoding_batch_size", True),
    ("encoding_batch_size", 1.5), ("encoding_batch_size", 1025), ("encoding_batch_max_bytes", 0),
    ("encoding_batch_max_bytes", False), ("encoding_batch_max_bytes", 4.5)])
def test_invalid_encoding_batch_configuration_fails_before_indexing(tmp_path, monkeypatch, field, value):
    from llm_lab.data import streaming_prepare as module
    def unexpected_index(_):
        raise AssertionError("非法批配置不应开始索引")
    monkeypatch.setattr(module, "_open_index", unexpected_index)
    config = config_for(tmp_path)
    config[field] = value
    with pytest.raises(ValueError, match=field):
        prepare_streaming(config, tmp_path)
    assert not Path(config["output_dir"]).exists()
