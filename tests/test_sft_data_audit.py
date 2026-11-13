"""真实 BPE 的小型结构验收夹具；不下载模型、不运行 GPU。"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_sft_data import (  # noqa: E402
    EMPTY_SYSTEM_PREFIX, PREFIX, audit_dataset, canonical_hash, dataset_identity, main, sha256_file,
)
from llm_lab.data.text_normalization import build_chinese_normalizer  # noqa: E402
from llm_lab.tokenization.bpe import train_bpe  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_row(tokenizer, identifier, split, language, *, system=""):
    prompt = "Please preserve sample " + identifier + "."
    response = "OK"
    prefix = (PREFIX if system else EMPTY_SYSTEM_PREFIX).format(system=system, prompt=prompt)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    answer_ids = tokenizer.encode(response, add_special_tokens=False).ids + [2]
    texts = {"system": system, "prompt": prompt, "response": response, "input_context": ""}
    source = {"dataset": "original-fixture", "revision": "a" * 40, "file": "fixture.jsonl", "row_index": 0,
              "source_subset": "toy", "license": "original-fixture-only"}
    source.update({key: hashlib.sha256(b"fixture").hexdigest() for key in (
        "raw_row_sha256", "original_prompt_sha256", "original_response_sha256", "original_context_sha256")})
    source["normalized_text_sha256"] = canonical_hash(texts)
    return {"schema_version": 1, "id": identifier, "split": split, "language": language, "task": "fixture",
            **texts, "input_ids": prefix_ids + answer_ids, "loss_mask": [0] * len(prefix_ids) + [1] * len(answer_ids),
            "prompt_token_count": len(prefix_ids), "answer_token_count": len(answer_ids),
            "group_id": "group-" + identifier, "source": source}


def repin(folder, manifest, rows):
    for split, items in rows.items():
        path = folder / (split + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items), encoding="utf-8")
        manifest["splits"][split] = {"path": path.name, "sha256": sha256_file(path), "examples": len(items)}
    manifest["data_identity"] = dataset_identity(manifest)
    write_json(folder / "manifest.json", manifest)


@pytest.fixture
def corpus(tmp_path):
    tokenizer = train_bpe(["System User Assistant Please preserve sample train validation test zh en OK", "fixture"],
                          tmp_path / "tokenizer.json", vocab_size=512, min_frequency=1)
    info = {"path": "tokenizer.json", "sha256": sha256_file(tmp_path / "tokenizer.json"),
            "eos_id": 2, "pad_id": 0, "unk_id": 1, "vocab_size": tokenizer.get_vocab_size()}
    manifest = {"schema_version": 1, "kind": "sft_tokenized_conversations", "fixture": True,
                "splits": {}, "tokenizer": info,
                "tokenization": {"prefix_template": PREFIX, "empty_system_prefix_template": EMPTY_SYSTEM_PREFIX,
                                 "encoding": "prefix_and_response_separately", "max_sequence_tokens": 1024},
                "chinese_normalization": build_chinese_normalizer("none").provenance}
    rows = {split: [make_row(tokenizer, split + "-" + lang, split, lang)
                    for lang in ("zh", "en")] for split in ("train", "validation", "test")}
    rows["pilot"] = [{**copy.deepcopy(row), "split": "pilot"} for row in rows["train"]]
    repin(tmp_path, manifest, rows)
    return tmp_path, manifest, rows, tokenizer


def test_independent_bpe_audit_passes_fixture_without_model(corpus):
    folder, _, _, _ = corpus
    report = audit_dataset(folder)
    assert report["status"] == "passed"
    assert report["fixture"] is True and report["structurally_ready"] is False
    assert report["semantic_quality_verified"] is False
    assert report["splits"]["train"]["zh_answer_token_fraction"] == .5
    assert report["checks"]["retokenization"] is True


def test_rejects_mask_prompt_boundary_corruption(corpus):
    folder, manifest, rows, _ = corpus
    rows["validation"][0]["loss_mask"][0] = 1
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="mask 边界"):
        audit_dataset(folder)


def test_rejects_mask_length_corruption(corpus):
    folder, manifest, rows, _ = corpus
    rows["test"][0]["loss_mask"].pop()
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="mask 长度"):
        audit_dataset(folder)


def test_rejects_token_ids_not_matching_bpe(corpus):
    folder, manifest, rows, _ = corpus
    row = rows["validation"][0]
    row["input_ids"][0] = (row["input_ids"][0] + 5) % manifest["tokenizer"]["vocab_size"]
    if row["input_ids"][0] < 3:
        row["input_ids"][0] = 5
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="重编码"):
        audit_dataset(folder)


def test_rejects_cross_split_group_leakage(corpus):
    folder, manifest, rows, _ = corpus
    rows["validation"][0]["group_id"] = rows["train"][0]["group_id"]
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="跨分区泄漏: group_id"):
        audit_dataset(folder)


def test_rejects_cross_split_prompt_even_with_new_id_and_group(corpus):
    folder, manifest, rows, tokenizer = corpus
    old = rows["train"][0]
    replacement = make_row(tokenizer, old["id"], "validation", "zh")
    replacement["id"] = "new-id"
    replacement["group_id"] = "new-group"
    rows["validation"][0] = replacement
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="跨分区泄漏: prompt"):
        audit_dataset(folder)


def test_rejects_cross_split_nonempty_context(corpus):
    folder, manifest, rows, _ = corpus
    for split in ("train", "validation"):
        row = rows[split][0]
        row["input_context"] = "Shared source material."
        row["source"]["normalized_text_sha256"] = canonical_hash({key: row[key] for key in (
            "system", "prompt", "response", "input_context")})
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="跨分区泄漏: input_context"):
        audit_dataset(folder)


def test_rejects_pilot_rows_not_in_train(corpus):
    folder, manifest, rows, _ = corpus
    rows["pilot"][0]["id"] = "not-in-training"
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="pilot 包含 train 外"):
        audit_dataset(folder)


def test_rejects_changed_pilot_content_under_training_id(corpus):
    folder, manifest, rows, _ = corpus
    rows["pilot"][0]["task"] = "changed-task"
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="pilot 包含 train 外"):
        audit_dataset(folder)


def test_rejects_file_hash_failure_before_row_read(corpus):
    folder, _, _, _ = corpus
    path = folder / "validation.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="文件 SHA256"):
        audit_dataset(folder)


def test_identity_excludes_acceptance_but_binds_template(corpus):
    folder, manifest, _, _ = corpus
    original = manifest["data_identity"]
    manifest["acceptance"] = {"status": "passed", "report": "acceptance.json"}
    assert dataset_identity(manifest) == original
    manifest["tokenization"]["max_sequence_tokens"] = 512
    write_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="data_identity"):
        audit_dataset(folder)


def test_formal_data_cannot_disable_chinese_conversion(corpus):
    folder, manifest, _, _ = corpus
    manifest["fixture"] = False
    write_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="正式 SFT 必须采用 t2s"):
        audit_dataset(folder)


def test_rejects_language_token_imbalance(corpus):
    folder, manifest, rows, _ = corpus
    rows["validation"][0]["language"] = "en"
    repin(folder, manifest, rows)
    with pytest.raises(ValueError, match="中英回答 Token 占比"):
        audit_dataset(folder)


def test_rejects_manifest_path_escape(corpus):
    folder, manifest, _, _ = corpus
    manifest["splits"]["validation"]["path"] = "../outside.jsonl"
    manifest["data_identity"] = dataset_identity(manifest)
    write_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="路径越界"):
        audit_dataset(folder)


def test_cli_report_cannot_overwrite_dataset_file(corpus):
    folder, _, _, _ = corpus
    before = (folder / "train.jsonl").read_bytes()
    with pytest.raises(ValueError, match="报告不能覆盖"):
        main(["--data-dir", str(folder), "--output", str(folder / "train.jsonl")])
    assert (folder / "train.jsonl").read_bytes() == before
