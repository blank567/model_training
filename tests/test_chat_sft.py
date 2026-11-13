"""SFT/Base 生成必须共用模板、保留完整问题并核验训练身份。"""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("chat_sft", SCRIPTS / "chat_sft.py")
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)

from llm_lab.common.io import canonical_hash, sha256_file, write_json
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_lab.tokenization.bpe import train_bpe


@pytest.fixture
def data(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    tokenizer = train_bpe(["System: Guide\nUser: Hi\nAssistant: Answer A B 中文回答。"] * 8,
                          folder / "tokenizer.json", vocab_size=300, min_frequency=1)
    tokenization = {"prefix_template": chat.PREFIX, "empty_system_prefix_template": chat.EMPTY_PREFIX,
                    "encoding": "prefix_and_response_separately", "max_sequence_tokens": 256}
    manifest = {"schema_version": 1, "kind": "sft_tokenized_conversations", "fixture": True,
                "splits": {}, "tokenizer": {"path": "tokenizer.json", "sha256": sha256_file(folder / "tokenizer.json"),
                    "vocab_size": tokenizer.get_vocab_size(), "pad_id": 0, "unk_id": 1, "eos_id": 2},
                "tokenization": tokenization, "chinese_normalization": {"mode": "none"},
                "acceptance": {"status": "passed"}}
    manifest["data_identity"] = canonical_hash({key: manifest[key] for key in chat.IDENTITY_FIELDS})
    write_json(folder / "manifest.json", manifest)
    return folder, tokenizer, manifest


@pytest.fixture
def checkpoints(data, tmp_path):
    folder, tokenizer, manifest = data
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), dim=16, n_layers=1, n_heads=2,
                         n_kv_heads=1, ffn_dim=32, max_seq_len=256, dropout=.1,
                         activation_checkpointing=False)
    model = Decoder(config)
    identity = {"manifest_sha256": sha256_file(folder / "manifest.json"),
                "data_identity": manifest["data_identity"], "tokenizer_sha256": manifest["tokenizer"]["sha256"],
                "base_checkpoint_sha256": "a" * 64}
    payload = {"schema_version": 1, "stage": "sft", "checkpoint_kind": "model_only", "step": 7,
               "config": {"project_root": str(tmp_path), "data_manifest": str(folder / "manifest.json")},
               "model_config": asdict(config), "identity": identity, "model": model.state_dict()}
    sft = tmp_path / "best.pt"
    torch.save(payload, sft)
    base_folder = tmp_path / "pretraining"
    base_folder.mkdir()
    (base_folder / "tokenizer.json").write_bytes((folder / "tokenizer.json").read_bytes())
    base_manifest = {"schema_version": 1, "kind": "pretrain_token_streams", "tokenizer": manifest["tokenizer"]}
    write_json(base_folder / "manifest.json", base_manifest)
    base_payload = {"schema_version": 1, "step": 100,
                    "config": {"project_root": str(tmp_path), "data_manifest": str(base_folder / "manifest.json"),
                               "model": asdict(config)},
                    "identity": {"manifest_sha256": sha256_file(base_folder / "manifest.json"),
                                 "tokenizer_sha256": manifest["tokenizer"]["sha256"]}, "model": model.state_dict()}
    base = tmp_path / "base.pt"
    torch.save(base_payload, base)
    return sft, base, payload


class FakeDecoder(torch.nn.Module):
    def __init__(self, vocab, sequence, context=256, prefer_forbidden=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_seq_len=context)
        self.vocab, self.sequence, self.calls = vocab, sequence, 0
        self.prefer_forbidden = prefer_forbidden

    def forward(self, x):
        scores = torch.full((1, x.shape[1], self.vocab), -50.)
        if self.prefer_forbidden:
            scores[0, -1, 0] = 100.
            scores[0, -1, 1] = 90.
        scores[0, -1, self.sequence[min(self.calls, len(self.sequence) - 1)]] = 50.
        self.calls += 1
        return SimpleNamespace(logits=scores)


def test_eos_stops_and_role_prefix_is_not_visible(data):
    _, tokenizer, manifest = data
    model = FakeDecoder(tokenizer.get_vocab_size(), [2])
    result = chat.generate(model, tokenizer, "Hi", tokenization=manifest["tokenization"], max_new_tokens=3, greedy=True)
    assert result["answer"] == ""
    assert result["stop_reason"] == "eos" and result["generated_ids"] == [2]
    assert model.calls == 1 and model.training
    assert result["prefix"] == "User:\nHi\n\nAssistant:\n"


def test_pad_unk_are_never_generated_and_answer_only_is_decoded(data):
    _, tokenizer, manifest = data
    identifier = tokenizer.encode("A", add_special_tokens=False).ids[0]
    model = FakeDecoder(tokenizer.get_vocab_size(), [identifier, 2], prefer_forbidden=True)
    result = chat.generate(model, tokenizer, "Hi", tokenization=manifest["tokenization"], system="Guide", max_new_tokens=3, greedy=True)
    assert result["generated_ids"] == [identifier, 2]
    assert result["answer"] == "A"
    assert "User:" not in result["answer"] and "<|eos|>" not in result["answer"]
    assert result["prefix"] == "System:\nGuide\n\nUser:\nHi\n\nAssistant:\n"


def test_context_overrun_fails_before_forward_instead_of_truncating(data):
    _, tokenizer, manifest = data
    model = FakeDecoder(tokenizer.get_vocab_size(), [2], context=12)
    with pytest.raises(ValueError, match="exceeds model context"):
        chat.generate(model, tokenizer, "A complete question", tokenization=manifest["tokenization"], max_new_tokens=10)
    assert model.calls == 0


def test_max_new_tokens_and_local_sampling_rng(data):
    _, tokenizer, manifest = data
    identifier = tokenizer.encode("A", add_special_tokens=False).ids[0]
    model = FakeDecoder(tokenizer.get_vocab_size(), [identifier])
    before = torch.get_rng_state().clone()
    result = chat.generate(model, tokenizer, "Hi", tokenization=manifest["tokenization"], max_new_tokens=2, seed=9)
    assert result["generated_token_count"] == 2 and result["stop_reason"] == "max_new_tokens"
    assert torch.equal(before, torch.get_rng_state())


def test_embedded_special_token_in_question_is_rejected(data):
    _, tokenizer, manifest = data
    model = FakeDecoder(tokenizer.get_vocab_size(), [2])
    with pytest.raises(ValueError, match="embedded special"):
        chat.generate(model, tokenizer, "Hi <|eos|>", tokenization=manifest["tokenization"], max_new_tokens=2)


def test_actual_tiny_sft_and_base_use_same_template(data, checkpoints):
    folder, _, _ = data
    sft, base, _ = checkpoints
    trained, tokenizer, sft_meta = chat.load_model(sft, threads=1)
    baseline, base_tokenizer, base_meta = chat.load_model(base, data_manifest=folder / "manifest.json", threads=1)
    assert sft_meta["checkpoint_step"] == 7 and base_meta["stage"] == "base_comparison"
    assert sft_meta["tokenization"] == base_meta["tokenization"]
    a = chat.generate(trained, tokenizer, "Hi", tokenization=sft_meta["tokenization"], max_new_tokens=0)
    b = chat.generate(baseline, base_tokenizer, "Hi", tokenization=base_meta["tokenization"], max_new_tokens=0)
    assert a["prefix"] == b["prefix"] and a["input_ids"] == b["input_ids"]
    assert sft_meta["checkpoint_sha256"] == sha256_file(sft)


def test_base_requires_explicit_sft_manifest(checkpoints):
    _, base, _ = checkpoints
    with pytest.raises(ValueError, match="requires --data-manifest"):
        chat.load_model(base)


def test_sft_manifest_byte_change_is_rejected(data, checkpoints):
    folder, _, _ = data
    sft, _, _ = checkpoints
    path = folder / "manifest.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        chat.load_model(sft)


def test_tokenizer_byte_change_is_rejected(data, checkpoints):
    folder, _, _ = data
    sft, _, _ = checkpoints
    path = folder / "tokenizer.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Tokenizer SHA256"):
        chat.load_model(sft)


def test_sft_cannot_silently_replace_bound_manifest(tmp_path, data, checkpoints):
    folder, _, _ = data
    sft, _, _ = checkpoints
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes((folder / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="cannot replace"):
        chat.load_model(sft, data_manifest=replacement)


def test_invalid_data_identity_is_rejected_even_if_checkpoint_hash_is_updated(data, checkpoints):
    folder, _, manifest = data
    sft, _, payload = checkpoints
    manifest["data_identity"] = "0" * 64
    write_json(folder / "manifest.json", manifest)
    payload["identity"]["manifest_sha256"] = sha256_file(folder / "manifest.json")
    torch.save(payload, sft)
    with pytest.raises(ValueError, match="data_identity"):
        chat.load_model(sft)


def test_superseded_dataset_is_rejected(data, checkpoints):
    folder, _, _ = data
    sft, _, _ = checkpoints
    write_json(folder / "SUPERSEDED.json", {"use_for_training": False})
    with pytest.raises(ValueError, match="superseded"):
        chat.load_model(sft)


def test_checkpoint_changed_during_load_is_rejected(checkpoints, monkeypatch):
    sft, _, _ = checkpoints
    original = chat.load_checkpoint
    def replace_while_loading(path):
        payload = original(path)
        payload["step"] = 8
        torch.save(payload, path)
        return payload
    monkeypatch.setattr(chat, "load_checkpoint", replace_while_loading)
    with pytest.raises(ValueError, match="changed while loading"):
        chat.load_model(sft)


def test_report_cannot_overwrite_or_escape_project(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    target = tmp_path / "answer.json"
    chat._write_report(target, {"answer": "Hi"})
    assert json.loads(target.read_text(encoding="utf-8"))["answer"] == "Hi"
    with pytest.raises(ValueError, match="new file"):
        chat._write_report(target, {"answer": "overwrite"})
    with pytest.raises(ValueError, match="inside"):
        chat._write_report(tmp_path.parent / "outside.json", {})


def test_cli_prints_answer_without_prefix(data, monkeypatch, capsys):
    _, tokenizer, manifest = data
    identifier = tokenizer.encode("A", add_special_tokens=False).ids[0]
    fake = FakeDecoder(tokenizer.get_vocab_size(), [identifier, 2])
    monkeypatch.setattr(chat, "load_model", lambda *a, **k: (fake, tokenizer, {"tokenization": manifest["tokenization"]}))
    chat.main(["--checkpoint", "ignored", "--prompt", "Hi", "--greedy", "--max-new-tokens", "3"])
    assert capsys.readouterr().out == "A\n"
