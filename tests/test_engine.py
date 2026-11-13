"""在极小随机 Token 上验证状态协议，不开展正式本地训练。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import sha256_file, write_json
from llm_lab.config import ExperimentConfig, ModelConfig, TrainConfig, load_config
from llm_lab.engine.checkpoint import load_checkpoint, save_checkpoint
from llm_lab.engine.corpus import BinaryCorpus
from llm_lab.engine.pretrain import optimizer_update, run_pretrain


def make_manifest(root: Path) -> Path:
    root.mkdir()
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    splits = {}
    for offset, split in enumerate(("train", "validation", "test")):
        splits[split] = {}
        for language in ("en", "zh"):
            name = f"{split}.{language}.bin"
            array = ((np.arange(256) + offset) % 32).astype("<u4")
            array.tofile(root / name)
            splits[split][language] = {"path": name, "token_count": len(array), "dtype": "uint32", "sha256": sha256_file(root / name)}
    path = root / "manifest.json"
    write_json(path, {"schema_version": 1, "tokenizer": {"path": "tokenizer.json", "sha256": sha256_file(tokenizer), "vocab_size": 32, "eos_id": 0}, "splits": splits})
    return path


def configuration(tmp_path, manifest, name):
    return ExperimentConfig(data_manifest=str(manifest), output_dir=str(tmp_path / name),
        model=ModelConfig(vocab_size=32, dim=16, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=32,
                          max_seq_len=8, dropout=0.15, activation_checkpointing=True),
        train=TrainConfig(device="cpu", dtype="fp32", seed=17, seq_len=6, micro_batch_size=1,
                          gradient_accumulation_steps=2, max_steps=4, warmup_steps=1,
                          eval_interval=2, eval_batches=1, save_interval=2, log_interval=2, tensorboard=False),
        budget={"hourly_rate_cny": 0, "run_limit_cny": 1, "project_limit_cny": 500,
                "reserve_cny": 100, "max_runtime_seconds": 120, "ledger_path": str(tmp_path / f"{name}_ledger.json")})


def test_corpus_targets_are_next_tokens(tmp_path):
    manifest = make_manifest(tmp_path / "corpus")
    corpus = BinaryCorpus(manifest, "train", 6)
    x, y = corpus.batch("en", 3, np.random.default_rng(3))
    assert x.shape == y.shape == (3, 6)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal((x + 1) % 32, y)


def test_resume_matches_uninterrupted_with_dropout_and_validation(tmp_path):
    manifest = make_manifest(tmp_path / "corpus")
    full = configuration(tmp_path, manifest, "full")
    interrupted = configuration(tmp_path, manifest, "interrupted")
    run_pretrain(full)
    partial = run_pretrain(interrupted, stop_after_steps=2)
    assert partial["status"] == "simulated_interruption"
    run_pretrain(interrupted, resume=Path(interrupted.output_dir) / "last.pt")
    left = load_checkpoint(Path(full.output_dir) / "last.pt")
    right = load_checkpoint(Path(interrupted.output_dir) / "last.pt")
    assert left["tokens_seen"] == right["tokens_seen"] == 48
    assert left["scheduler"] == right["scheduler"]
    assert left["sampler_rng"] == right["sampler_rng"]
    for name, value in left["model"].items():
        assert torch.equal(value, right["model"][name]), name
    for key, state in left["optimizer"]["state"].items():
        for item, value in state.items():
            assert torch.equal(value, right["optimizer"]["state"][key][item])


def test_wandb_enabled_resume_preserves_training_state_and_records_metrics(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setattr("llm_lab.common.tracking.configure_tracking_environment", lambda root: None)
    manifest = make_manifest(tmp_path / "corpus")
    full = configuration(tmp_path, manifest, "wandb_full")
    partial = configuration(tmp_path, manifest, "wandb_resumed")
    for config in (full, partial):
        config.train.wandb = True
        config.train.wandb_mode = "offline"
    run_pretrain(full)
    run_pretrain(partial, stop_after_steps=2)
    run_pretrain(partial, resume=Path(partial.output_dir) / "last.pt")
    left = load_checkpoint(Path(full.output_dir) / "last.pt")
    right = load_checkpoint(Path(partial.output_dir) / "last.pt")
    assert all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"])
    assert left["scheduler"] == right["scheduler"]
    assert fake_wandb.calls[1][0]["group"] == fake_wandb.calls[2][0]["group"]
    assert fake_wandb.calls[1][0]["id"] != fake_wandb.calls[2][0]["id"]
    assert all(run.finishes == [{"exit_code": 0}] for _, run in fake_wandb.calls)
    assert all("train_loss" in values for _, run in fake_wandb.calls for values, _ in run.logs)


def test_resume_rejects_modified_corpus(tmp_path):
    manifest = make_manifest(tmp_path / "corpus")
    config = configuration(tmp_path, manifest, "run")
    run_pretrain(config, stop_after_steps=1)
    path = manifest.parent / "train.en.bin"
    data = np.fromfile(path, dtype="<u4")
    data[0] = (data[0] + 1) % 32
    data.tofile(path)
    with pytest.raises(ValueError, match="checksum"):
        run_pretrain(config, resume=Path(config.output_dir) / "last.pt")


def test_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "invalid.json"
    write_json(path, {"data_manifest": "x", "output_dir": "x", "train": {"learning_rtae": 0.1}})
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_budget_records_and_stops_at_limit(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("llm_lab.common.budget.time.monotonic", lambda: clock[0])
    guard = BudgetGuard({"hourly_rate_cny": 10, "run_limit_cny": 1, "max_runtime_seconds": 10000}, "a", tmp_path / "ledger.json")
    clock[0] = 360
    assert guard.should_stop()
    assert guard.update()["estimated_compute_cny"] == 1
    with pytest.raises(ValueError, match="exhausted"):
        BudgetGuard({"hourly_rate_cny": 10, "run_limit_cny": 1}, "a", tmp_path / "ledger.json")


def test_even_accumulation_enforces_bilingual_balance(tmp_path):
    config = configuration(tmp_path, "unused", "run")
    config.train.gradient_accumulation_steps = 3
    with pytest.raises(ValueError, match="even"):
        config.validate()


def test_loss_scaling_overflow_skips_update_and_backs_off():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 120)
    before = model.weight.detach().clone()
    initial_scale = scaler.get_scale()
    finite_loss = model(torch.tensor([[1e4]])).sum()
    assert torch.isfinite(finite_loss)
    scaler.scale(finite_loss).backward()
    updated, _ = optimizer_update(model, optimizer, scaler, 1.0)
    assert not updated
    assert torch.equal(before, model.weight)
    assert scaler.get_scale() == initial_scale / 2
    scaler.scale(model(torch.tensor([[0.01]])).sum()).backward()
    updated, _ = optimizer_update(model, optimizer, scaler, 1.0)
    assert updated
    assert not torch.equal(before, model.weight)


def test_nonfinite_unscaled_gradients_fail_fast():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(FloatingPointError):
        optimizer_update(model, optimizer, scaler, 1.0)


def test_paid_runs_cannot_fork_project_budget_ledger(tmp_path):
    config = configuration(tmp_path, "unused", "run")
    config.train.device = "cuda"
    config.budget["ledger_path"] = str(tmp_path / "hidden_ledger.json")
    with pytest.raises(ValueError, match="share"):
        config.validate()


def test_eval_rejects_replaced_dataset_manifest(tmp_path):
    from llm_lab.cli import main
    from llm_lab.common.io import read_json
    manifest = make_manifest(tmp_path / "corpus")
    config = configuration(tmp_path, manifest, "run")
    checkpoint = tmp_path / "trusted.pt"
    save_checkpoint(checkpoint, {"schema_version": 1, "config": config.to_dict(),
                    "identity": {"manifest_sha256": sha256_file(manifest)}})
    changed = read_json(manifest)
    changed["new_version"] = True
    write_json(manifest, changed)
    with pytest.raises(ValueError, match="manifest differs"):
        main(["generate", "--checkpoint", str(checkpoint), "--device", "cpu", "--prompt", "hello"])


def test_config_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "project"
    config_path = root / "configs" / "experiment.json"
    write_json(config_path, {"project_root": "..", "data_manifest": "artifacts/data.json",
                            "output_dir": "runs/run", "budget": {"ledger_path": "runs/budget_ledger.json"}})
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert config.data_manifest == str(root / "artifacts" / "data.json")
    assert config.output_dir == str(root / "runs" / "run")
    assert config.budget["ledger_path"] == str(root / "runs" / "budget_ledger.json")

