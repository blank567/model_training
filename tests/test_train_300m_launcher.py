"""启动器的冻结、互斥、断点与身份边界；不调用 CUDA 或远程网络。"""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_300m_test", ROOT / "scripts/train_300m.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    root = tmp_path
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "configs/pretrain").mkdir(parents=True)
    corpus = root / "artifacts/pretrain/bilingual_3b"
    corpus.mkdir(parents=True)
    tokenizer = corpus / "tokenizer.json"
    tokenizer.write_text('{"fixture":true}', encoding="utf-8")
    token_sha = launcher.sha256_file(tokenizer)
    manifest = corpus / "manifest.json"
    manifest.write_text(json.dumps({"acceptance": {"status": "passed"},
        "tokenizer": {"path": "tokenizer.json", "sha256": token_sha, "vocab_size": 32000},
        "splits": {"train": {lang: {"token_count": 1550000000} for lang in ("en", "zh")}}}), encoding="utf-8")
    monkeypatch.setattr(launcher, "MANIFEST_SHA", launcher.sha256_file(manifest))
    monkeypatch.setattr(launcher, "TOKENIZER_SHA", token_sha)
    monkeypatch.setattr(launcher, "source_hash", lambda: "source-before")
    raw = json.loads((ROOT / "configs/pretrain/300m_3b.candidate.json").read_text())
    raw["project_root"] = str(root)
    raw["budget"]["hourly_rate_cny"] = 3.03
    template = root / "configs/pretrain/300m_3b.json"
    template.write_text(json.dumps(raw), encoding="utf-8")
    old = root / "runs/pretrain_100m_1b_original"
    old.mkdir(parents=True)
    (old / "last.pt").write_bytes(b"old model untouched")
    return root, template, old


def freeze(plan):
    result = launcher.prepare(plan[1], "20261025_010203")
    return Path(result["RUN_CONFIG"]), result


def simulate_remote(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(launcher, "require_linux", lambda: None)
    monkeypatch.setattr(launcher, "training_lock", lambda root: nullcontext())
    monkeypatch.setattr(launcher, "training_processes", lambda: [])
    monkeypatch.setattr(launcher, "gpu_idle", lambda: None)
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda path: SimpleNamespace(free=30 * launcher.GIB))


def test_prepare_freezes_independent_paths_and_refuses_overwrite(plan):
    frozen, result = freeze(plan)
    assert not Path(result["RUN_DIR"]).exists()
    assert frozen.parent == plan[0] / "runs/configs"
    cfg = launcher.frozen_config(frozen)
    assert cfg.train.max_steps == 183106
    assert cfg.train.wandb_run_name.endswith("20261025_010203")
    assert Path(result["RUN_LOG"]).parent == Path(result["RUN_DIR"])
    env = frozen.with_suffix(".env").read_text()
    assert "source " in env and "CUDA_VISIBLE_DEVICES=0" in env
    before = frozen.read_bytes()
    with pytest.raises(FileExistsError):
        freeze(plan)
    assert frozen.read_bytes() == before
    assert (plan[2] / "last.pt").read_bytes() == b"old model untouched"


def test_manifest_and_source_changes_reject_frozen_run(plan, monkeypatch):
    frozen, _ = freeze(plan)
    monkeypatch.setattr(launcher, "source_hash", lambda: "source-after")
    with pytest.raises(ValueError, match="源码已改变"):
        launcher.frozen_config(frozen)
    monkeypatch.setattr(launcher, "source_hash", lambda: "source-before")
    (plan[0] / "artifacts/pretrain/bilingual_3b/manifest.json").write_text('{}')
    with pytest.raises(ValueError, match="固定 SHA256"):
        launcher.frozen_config(frozen)


def test_changed_config_rejected_and_old_checkpoint_cannot_be_target(plan):
    frozen, _ = freeze(plan)
    raw = json.loads(frozen.read_text())
    raw["output_dir"] = str(plan[2])
    frozen.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="独立的 300M"):
        launcher.frozen_config(frozen)


def test_probe_keeps_full_schedule_and_resume_uses_only_own_checkpoint(plan, monkeypatch):
    frozen, result = freeze(plan)
    simulate_remote(monkeypatch)
    calls = []
    def run(config, **kwargs):
        calls.append((config.train.max_steps, kwargs))
        output = Path(config.output_dir)
        output.mkdir(exist_ok=True)
        (output / "last.pt").write_bytes(b"own checkpoint")
        return {"status": "simulated_interruption", "step": 32}
    monkeypatch.setattr(launcher, "run_engine", run)
    launcher.execute("probe", frozen)
    assert calls == [(183106, {"stop_after_steps": 32})]
    with pytest.raises(FileExistsError, match="续训使用 resume"):
        launcher.execute("start", frozen)
    launcher.execute("resume", frozen)
    assert calls[-1] == (183106, {"resume": Path(result["RUN_DIR"]) / "last.pt"})
    assert (plan[2] / "last.pt").read_bytes() == b"old model untouched"


def test_preflight_rejects_existing_training_before_gpu_query(plan, monkeypatch):
    frozen, _ = freeze(plan)
    cfg = launcher.frozen_config(frozen)
    monkeypatch.setattr(launcher, "training_processes", lambda: [{"pid": 2987, "args": "old 100M"}])
    monkeypatch.setattr(launcher, "gpu_idle", lambda: pytest.fail("must reject before GPU query"))
    with pytest.raises(RuntimeError, match="已有训练进程"):
        launcher.preflight(cfg, "start")


def test_probe_steps_cli_keeps_full_training_plan(plan, monkeypatch):
    frozen, _ = freeze(plan)
    simulate_remote(monkeypatch)
    calls = []
    monkeypatch.setattr(launcher, "run_engine", lambda cfg, **kwargs: calls.append((cfg.train.max_steps, kwargs)) or {"status": "simulated_interruption"})
    launcher.main(["probe", "--config", str(frozen), "--steps", "7"])
    assert calls == [(183106, {"stop_after_steps": 7})]
    for invalid in (0, -1, 183106, True):
        with pytest.raises(ValueError, match="probe steps"):
            launcher.execute("probe", frozen, probe_steps=invalid)


def test_check_existing_checkpoint_uses_resume_disk_requirement(plan, monkeypatch):
    frozen, result = freeze(plan)
    simulate_remote(monkeypatch)
    monkeypatch.setattr(launcher, "GIB", 1)
    output = Path(result["RUN_DIR"])
    output.mkdir()
    (output / "last.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda path: SimpleNamespace(free=12))
    assert launcher.execute("check", frozen)["required_free_gib"] == 10


def test_exhausted_recorded_project_budget_rejected_before_engine(plan, monkeypatch):
    frozen, _ = freeze(plan)
    simulate_remote(monkeypatch)
    ledger = plan[0] / "runs/budget_ledger.json"
    ledger.write_text(json.dumps({"runs": {"past": {"estimated_compute_cny": 400}}}))
    monkeypatch.setattr(launcher, "run_engine", lambda *a, **kw: pytest.fail("No budget must prevent engine start"))
    with pytest.raises(RuntimeError, match="预算已耗尽"):
        launcher.execute("probe", frozen)


def test_disk_reserve_subtracts_only_unique_existing_checkpoint_inodes(tmp_path, monkeypatch):
    # 按小单位检查相同算术与 inode 分支，避免 Windows truncate 实际写入多 GiB。
    monkeypatch.setattr(launcher, "GIB", 1)
    assert launcher.required_free_bytes(tmp_path) == 20 * launcher.GIB
    a = tmp_path / "last.pt"
    with a.open("wb") as handle:
        handle.truncate(2 * launcher.GIB)
    assert launcher.required_free_bytes(tmp_path, True) == 18 * launcher.GIB
    b = tmp_path / "best.pt"
    os.link(a, b)
    assert launcher.required_free_bytes(tmp_path, True) == 18 * launcher.GIB
    b.unlink()
    with b.open("wb") as handle:
        handle.truncate(20 * launcher.GIB)
    assert launcher.required_free_bytes(tmp_path, True) == 8 * launcher.GIB


@pytest.mark.parametrize("code,output", [(1, ""), (0, "[Not Supported]"), (0, "2987\n")])
def test_gpu_query_failure_or_occupied_device_is_not_treated_as_idle(monkeypatch, code, output):
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=code, stdout=output))
    with pytest.raises(RuntimeError):
        launcher.gpu_idle()


def test_failed_engine_records_failure_and_releases_guard(plan, monkeypatch):
    frozen, _ = freeze(plan)
    simulate_remote(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError("fixture OOM")
    monkeypatch.setattr(launcher, "run_engine", fail)
    with pytest.raises(RuntimeError, match="fixture OOM"):
        launcher.execute("probe", frozen)
    assert json.loads(frozen.with_suffix(".state.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("micro,accum,recompute,expected_steps,expected_tokens", [
    (4, 4, True, 183106, 16384),
    (8, 2, False, 183106, 16384),
    (8, 4, False, 91553, 32768),
    (16, 2, False, 91553, 32768),
    (16, 2, True, 91553, 32768),
])
def test_plan_uses_token_budget_for_supported_microbatch_combinations(plan, monkeypatch, micro, accum, recompute, expected_steps, expected_tokens):
    root, template, _ = plan
    raw = json.loads(template.read_text())
    raw["model"]["activation_checkpointing"] = recompute
    raw["train"].update(micro_batch_size=micro, gradient_accumulation_steps=accum, max_steps=expected_steps)
    template.write_text(json.dumps(raw))
    frozen, result = freeze(plan)
    identity = json.loads(frozen.with_suffix(".identity.json").read_text())
    assert identity["tokens_per_update"] == expected_tokens
    assert identity["planned_tokens"] == 3_000_008_704
    assert identity["parameter_count"] == 303_350_784
    simulate_remote(monkeypatch)
    checked = launcher.execute("check", frozen)
    assert checked["tokens_per_update"] == expected_tokens
    assert checked["planned_tokens"] == 3_000_008_704
    assert checked["parameter_count"] == 303_350_784
    assert not Path(result["RUN_DIR"]).exists()


@pytest.mark.parametrize("micro,accum,steps,sequence,error", [
    (8, 4, 183106, 1024, "必须为 91553"),
    (16, 2, 91552, 1024, "必须为 91553"),
    (4, 2, 183106, 1024, "16384 或 32768"),
    (32, 2, 45777, 1024, "16384 或 32768"),
    (8, 4, 183106, 512, "1024 上下文"),
    (32, 1, 91553, 1024, "Bilingual accumulation"),
])
def test_plan_rejects_wrong_update_budget_or_double_training_tokens(plan, micro, accum, steps, sequence, error):
    cfg = launcher.load_config(plan[1])
    cfg.train.micro_batch_size = micro
    cfg.train.gradient_accumulation_steps = accum
    cfg.train.max_steps = steps
    cfg.train.seq_len = sequence
    with pytest.raises(ValueError, match=error):
        launcher.validate_plan(cfg)


def test_plan_rejects_untied_embedding_parameter_change(plan):
    cfg = launcher.load_config(plan[1])
    cfg.model.tie_embeddings = False
    with pytest.raises(ValueError, match="303,350,784"):
        launcher.validate_plan(cfg)


def test_old_identity_without_extended_token_fields_remains_compatible(plan):
    frozen, _ = freeze(plan)
    path = frozen.with_suffix(".identity.json")
    identity = json.loads(path.read_text())
    for key in ("tokens_per_update", "planned_tokens", "parameter_count"):
        identity.pop(key)
    path.write_text(json.dumps(identity))
    assert launcher.frozen_config(frozen).train.max_steps == 183106


def test_extended_identity_token_fields_cannot_disagree_with_frozen_config(plan):
    frozen, _ = freeze(plan)
    path = frozen.with_suffix(".identity.json")
    identity = json.loads(path.read_text())
    identity["tokens_per_update"] = 32768
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="计划身份不符"):
        launcher.frozen_config(frozen)


def test_cleared_run_cannot_restart_but_status_stays_readable(plan, monkeypatch):
    frozen, result = freeze(plan)
    frozen.with_suffix(".state.json").write_text(json.dumps({"status": "cleared", "removed_output": result["RUN_DIR"]}))
    simulate_remote(monkeypatch)
    monkeypatch.setattr(launcher, "run_engine", lambda *a, **kw: pytest.fail("Cleared run must never enter engine"))
    for action in ("probe", "start", "resume", "check"):
        with pytest.raises(ValueError, match="已清空"):
            launcher.execute(action, frozen)
    current = launcher.status(frozen)
    assert current["launcher"]["status"] == "cleared"
    assert current["last_checkpoint_exists"] is False
    assert (plan[2] / "last.pt").read_bytes() == b"old model untouched"


@pytest.mark.skipif(os.name != "posix", reason="fcntl lock is Linux/POSIX only")
def test_project_lock_is_exclusive_until_training_context_exits(tmp_path):
    with launcher.training_lock(tmp_path):
        with pytest.raises(RuntimeError, match="项目 GPU 锁"):
            with launcher.training_lock(tmp_path):
                pytest.fail("duplicate lock acquired")
    with launcher.training_lock(tmp_path):
        pass
