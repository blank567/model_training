"""SFT 运行数据：监督边界、逐轮覆盖和验收身份的实际 BPE 测试。"""
from copy import deepcopy
import json
from pathlib import Path
import random

import pytest
import torch

from llm_lab.tokenization.bpe import train_bpe
from llm_sft.data import SFTCorpus, canonical_hash, deterministic_epoch_batches, sha256_file, IDENTITY_FIELDS


CHECKS = {key: True for key in ("file_hashes", "retokenization", "answer_mask_and_eos", "normalized_text_identity",
          "declared_group_and_exact_split_isolation", "pilot_subset", "language_token_balance", "explicit_quality_rules")}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def repin(directory, manifest, rows, *, report_override=None):
    for split, selected in rows.items():
        path = directory / (split + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
        manifest["splits"][split] = {"path": path.name, "sha256": sha256_file(path), "examples": len(selected)}
    manifest["data_identity"] = canonical_hash({key: manifest[key] for key in IDENTITY_FIELDS})
    report = {"schema_version": 1, "status": "passed", "data_identity": manifest["data_identity"],
              "tokenizer_sha256": manifest["tokenizer"]["sha256"], "fixture": manifest["fixture"],
              "structurally_ready": not manifest["fixture"], "checks": deepcopy(CHECKS),
              "splits": deepcopy(manifest["splits"])}
    if report_override:
        report.update(report_override)
    save(directory / "acceptance.json", report)
    manifest["acceptance"] = {"status": "passed", "report": "acceptance.json", "sha256": sha256_file(directory / "acceptance.json")}
    save(directory / "manifest.json", manifest)


@pytest.fixture
def corpus(tmp_path):
    tokenizer = train_bpe(["User Assistant Please answer this much longer prompt about a fox. An animal.", "请回答一个问题。你好。"],
                          tmp_path / "tokenizer.json", vocab_size=360, min_frequency=1)
    def row(identifier, split, language, prompt, answer):
        prefix = tokenizer.encode("User:\n" + prompt + "\n\nAssistant:\n", add_special_tokens=False).ids
        target = tokenizer.encode(answer, add_special_tokens=False).ids + [2]
        return {"schema_version": 1, "id": identifier, "split": split, "language": language, "prompt": prompt,
                "response": answer, "system": "", "input_context": "", "task": "fixture", "group_id": identifier,
                "input_ids": prefix + target, "loss_mask": [0]*len(prefix) + [1]*len(target),
                "prompt_token_count": len(prefix), "answer_token_count": len(target)}
    rows = {split: [row(split+"zh", split, "zh", "请回答一个问题。", "你好。"),
                    row(split+"en", split, "en", "Please answer this much longer prompt about a fox.", "An animal.")]
            for split in ("train", "validation", "test")}
    rows["pilot"] = [{**deepcopy(row), "split": "pilot"} for row in rows["train"]]
    manifest = {"schema_version": 1, "kind": "sft_tokenized_conversations", "fixture": True, "splits": {},
                "tokenizer": {"path": "tokenizer.json", "sha256": sha256_file(tmp_path/"tokenizer.json"),
                              "pad_id": 0, "unk_id": 1, "eos_id": 2, "vocab_size": tokenizer.get_vocab_size()},
                "tokenization": {"encoding": "prefix_and_response_separately", "max_sequence_tokens": 1024},
                "chinese_normalization": {"mode": "none"}}
    repin(tmp_path, manifest, rows)
    return tmp_path, manifest, rows, tokenizer


def test_answer_first_token_and_eos_are_supervised_once(corpus):
    directory, _, rows, tokenizer = corpus
    dataset = SFTCorpus(directory/"manifest.json", "train")
    batch = dataset.collate([0, 1])
    assert batch["x"].shape == batch["y"].shape
    for index, row in enumerate(rows["train"]):
        before, length = row["prompt_token_count"], len(row["input_ids"])-1
        assert torch.equal(batch["x"][index, :length], torch.tensor(row["input_ids"][:-1]))
        assert (batch["y"][index, :before-1] == -100).all()
        assert batch["y"][index, before-1].item() == row["input_ids"][before]
        assert batch["y"][index, length-1].item() == 2
        assert (batch["y"][index, length:] == -100).all()
        assert (batch["x"][index, length:] == 0).all()
    assert batch["supervised_tokens"] == sum(row["answer_token_count"] for row in rows["train"])
    assert batch["input_tokens"] == sum(len(row["input_ids"])-1 for row in rows["train"])
    assert batch["sequence_tokens"] == batch["input_tokens"] + 2
    assert dataset.language_indices == {"zh": [0], "en": [1]}


def test_accumulation_counts_only_real_answer_tokens(corpus):
    directory, _, _, _ = corpus
    dataset = SFTCorpus(directory/"manifest.json", "train")
    pooled = dataset.collate([0, 1])
    separate = [dataset.collate([index]) for index in (0, 1)]
    assert sum(batch["supervised_tokens"] for batch in separate) == pooled["supervised_tokens"]
    assert sum(batch["input_tokens"] for batch in separate) == pooled["input_tokens"]


def test_epoch_plan_covers_tail_and_can_restore_update_cursor():
    before = random.getstate()
    plan = deterministic_epoch_batches(19, 3, 2, 42, 0)
    assert random.getstate() == before
    assert sorted(i for group in plan for micro in group for i in micro) == list(range(19))
    assert len(plan) == 4 and [len(batch) for batch in plan[-1]] == [1]
    assert deterministic_epoch_batches(19, 3, 2, 42, 0)[2:] == plan[2:]
    assert deterministic_epoch_batches(19, 3, 2, 42, 1) != plan
    assert deterministic_epoch_batches(0, 3, 2, 42, 0) == []


@pytest.mark.parametrize("values", [(5,0,1,42,0),(5,1,0,42,0),(5,1,1,-1,0),(5,1,1,42,-1),(True,1,1,42,0)])
def test_invalid_plan_values_fail(values):
    with pytest.raises(ValueError): deterministic_epoch_batches(*values)


def test_changed_data_bytes_are_not_accepted(corpus):
    directory, _, _, _ = corpus
    with (directory/"train.jsonl").open("a",encoding="utf-8") as stream: stream.write("\n")
    with pytest.raises(ValueError, match="SHA256"): SFTCorpus(directory/"manifest.json", "train")


def test_stale_report_identity_fails_even_with_updated_report_hash(corpus):
    directory, manifest, rows, _ = corpus
    repin(directory, manifest, rows, report_override={"data_identity": "0"*64})
    with pytest.raises(ValueError, match="数据身份"): SFTCorpus(directory/"manifest.json", "train")


def test_incomplete_acceptance_checks_are_rejected(corpus):
    directory, manifest, rows, _ = corpus
    repin(directory, manifest, rows, report_override={"checks": {}})
    with pytest.raises(ValueError, match="检查项"): SFTCorpus(directory/"manifest.json", "train")


def test_deprecated_dataset_is_not_a_training_input(corpus):
    directory, _, _, _ = corpus
    save(directory/"SUPERSEDED.json", {"replacement": "new"})
    with pytest.raises(ValueError, match="废弃"): SFTCorpus(directory/"manifest.json", "train")


def test_test_split_requires_explicit_evaluation_opt_in(corpus):
    directory, _, _, _ = corpus
    with pytest.raises(ValueError, match="allow_test"): SFTCorpus(directory/"manifest.json", "test")
    assert len(SFTCorpus(directory/"manifest.json", "test", allow_test=True)) == 2


def test_repinning_does_not_hide_bad_mask_boundary(corpus):
    directory, manifest, rows, _ = corpus
    rows["train"][0]["loss_mask"][0] = 1
    repin(directory, manifest, rows)
    with pytest.raises(ValueError, match="监督边界"): SFTCorpus(directory/"manifest.json", "train")


def test_fixture_cannot_be_sent_to_cuda(corpus):
    directory, _, _, _ = corpus
    dataset = SFTCorpus(directory/"manifest.json", "train")
    with pytest.raises(ValueError, match="CPU"): dataset.collate([0], device="cuda")


def test_production_cannot_skip_acceptance(corpus):
    directory, manifest, rows, _ = corpus
    manifest["fixture"] = False
    manifest["chinese_normalization"]["mode"] = "t2s"
    repin(directory, manifest, rows)
    with pytest.raises(ValueError, match="不能跳过"): SFTCorpus(directory/"manifest.json", "train", require_acceptance=False)


def test_tokenizer_change_is_detected_before_batches(corpus):
    directory, _, _, _ = corpus
    with (directory/"tokenizer.json").open("a",encoding="utf-8") as stream: stream.write(" ")
    with pytest.raises(ValueError, match="tokenizer SHA256"): SFTCorpus(directory/"manifest.json", "train")
