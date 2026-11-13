"""受限 GPU 测速协议的 CPU 夹具；没有真实 GPU、网络或大权重文件。"""
import importlib.util
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_300m_test", ROOT / "scripts/benchmark_300m.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "source_hash", lambda: "fixture-source")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    folder = tmp_path / "configs/pretrain"
    folder.mkdir(parents=True)
    raw = json.loads((ROOT / "configs/pretrain/300m_3b.json").read_text())
    raw["project_root"] = str(tmp_path)
    path = folder / "300m.json"
    path.write_text(json.dumps(raw))
    return tmp_path, path


def test_plan_is_cpu_only_and_candidates_keep_same_token_budget(project):
    root, template = project
    path = benchmark.create_plan(template, "20261025_153000")
    plan = benchmark.verified_plan(path)
    assert plan["gpu_training_started"] is False
    assert plan["total_timeout_seconds"] == 360
    assert plan["steps"] == 24
    for row in plan["trials"]:
        config = benchmark.load_config(row["config"])
        assert config.train.micro_batch_size * config.train.gradient_accumulation_steps * 1024 == 32768
        assert config.train.max_steps == 91553
        assert config.train.warmup_steps == 1500
        assert config.train.eval_batches * config.train.micro_batch_size * 1024 == 16384
        assert config.budget["run_limit_cny"] == 1
        assert config.budget["ledger_path"] == str(root / "runs/budget_ledger.json")
        assert not config.train.wandb and not config.train.tensorboard
        assert not Path(config.output_dir).exists()
    with pytest.raises(FileExistsError):
        benchmark.create_plan(template, "20261025_153000")


def test_plan_tampering_and_output_escape_are_rejected(project):
    _, template = project
    path = benchmark.create_plan(template, "20261025_153001")
    plan = json.loads(path.read_text())
    plan["trials"][0]["output_dir"] = str(project[0] / "runs/pretrain_100m")
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="越界"):
        benchmark.verified_plan(path)


def test_config_hash_change_and_script_change_are_rejected(project, monkeypatch):
    _, template = project
    path = benchmark.create_plan(template, "20261025_153002")
    plan = json.loads(path.read_text())
    cfg = Path(plan["trials"][0]["config"])
    cfg.write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        benchmark.verified_plan(path)
    path2 = benchmark.create_plan(template, "20261025_153003")
    monkeypatch.setattr(benchmark, "source_hash", lambda: "changed-source")
    with pytest.raises(ValueError, match="源码身份"):
        benchmark.verified_plan(path2)


def test_run_without_execute_does_not_even_query_gpu(monkeypatch):
    monkeypatch.setattr(benchmark, "require_linux", lambda: pytest.fail("must not access GPU runtime"))
    result = benchmark.run_plan("not-present.json")
    assert result["status"] == "not_started"


def test_metrics_exclude_initial_steps_and_use_incremental_timing():
    rows = [{"step": step, "tokens_seen": step * 32768, "elapsed_seconds_this_session": 100 + step * .5}
            for step in range(2, 26, 2)]
    rows[0]["elapsed_seconds_this_session"] = 100
    stable = benchmark.summarize_metrics(rows)
    assert stable["first_step"] == 6 and stable["last_step"] == 24
    assert stable["tokens_per_second"] == 65536
    with pytest.raises(ValueError, match="记录不足"):
        benchmark.summarize_metrics(rows[:2])


def test_recommendation_requires_safe_headroom_and_checkpoint():
    gib = benchmark.GIB
    def row(name, speed, peak, saved=True, status="passed"):
        return {"name": name, "status": status, "checkpoint_saved": saved, "gpu_total_bytes": 48*gib,
                "peak_reserved_bytes": peak*gib, "stable": {"tokens_per_second": speed}}
    results = [row("fast_unsafe", 90000, 44), row("slower", 30000, 20),
               row("best_safe", 50000, 35), row("no_checkpoint", 100000, 30, False),
               row("oom", 200000, 20, True, "oom")]
    assert benchmark.choose_best(results)["name"] == "best_safe"
    assert benchmark.choose_best([results[0]]) is None


def test_cleanup_deletes_only_controlled_checkpoint_files(tmp_path):
    trial = tmp_path / "trials/example"
    trial.mkdir(parents=True)
    for name in benchmark.WEIGHTS:
        (trial/name).write_bytes(b"tiny weight")
    (trial/"summary.json").write_text("{}")
    original = tmp_path / "old_model"
    original.mkdir()
    (original/"last.pt").write_bytes(b"preserve")
    assert len(benchmark.cleanup_weights(trial, tmp_path)) == 4
    assert (trial/"summary.json").exists()
    with pytest.raises(ValueError, match="越界"):
        benchmark.cleanup_weights(original, tmp_path)
    assert (original/"last.pt").read_bytes() == b"preserve"


def test_worker_cannot_be_started_accidentally_without_controller(project, monkeypatch):
    monkeypatch.setattr(benchmark, "require_linux", lambda: None)
    monkeypatch.delenv("LLM_LAB_BENCHMARK_PLAN", raising=False)
    with pytest.raises(RuntimeError, match="持锁"):
        benchmark.worker("unused.json", "batch8_acc4")


def simulated_runtime(project, monkeypatch, interrupt=False):
    root, _ = project
    state = {"locked": False, "calls": 0}
    monkeypatch.setattr(benchmark, "require_linux", lambda: None)
    monkeypatch.setattr(benchmark, "training_processes", lambda: [])
    monkeypatch.setattr(benchmark, "gpu_idle", lambda: None)
    monkeypatch.setattr(benchmark, "corpus_identity", lambda cfg: None)
    monkeypatch.setattr(benchmark.shutil, "disk_usage", lambda path: SimpleNamespace(free=30 * benchmark.GIB))
    @contextmanager
    def lock(path):
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False
    monkeypatch.setattr(benchmark, "training_lock", lock)
    class Child:
        def __init__(self, args, **kwargs):
            assert state["locked"]
            plan = json.loads(Path(args[args.index("--plan") + 1]).read_text())
            name = args[args.index("--trial") + 1]
            item = next(r for r in plan["trials"] if r["name"] == name)
            output = Path(item["output_dir"]); output.mkdir(parents=True)
            (output/"last.pt").write_bytes(b"tiny checkpoint")
            (output/"last.pt.tmp").write_bytes(b"tiny partial")
            (output/"summary.json").write_text("{}")
            self.returncode = None
            state["calls"] += 1
            status = "oom" if state["calls"] == 1 else "passed"
            row = {"name": name, "status": status, "checkpoint_saved": status == "passed",
                   "stable": {"tokens_per_second": 50000 / state["calls"]},
                   "peak_reserved_bytes": 30 * benchmark.GIB, "gpu_total_bytes": 48 * benchmark.GIB}
            benchmark.write_json(item["report"], row)
        def wait(self, timeout=None):
            assert state["locked"]
            if interrupt and self.returncode is None:
                raise KeyboardInterrupt()
            self.returncode = self.returncode if self.returncode is not None else 0
            return self.returncode
        def poll(self):
            return self.returncode
        def send_signal(self, value):
            self.returncode = -2
        def kill(self):
            self.returncode = -9
    monkeypatch.setattr(benchmark.subprocess, "Popen", Child)
    return state


def test_serial_execution_keeps_lock_handles_oom_and_cleans_only_own_weights(project, monkeypatch):
    path = benchmark.create_plan(project[1], "20261025_153005")
    state = simulated_runtime(project, monkeypatch)
    result = benchmark.run_plan(path, True)
    assert not state["locked"] and state["calls"] == 3
    assert result["results"][0]["status"] == "oom"
    assert result["recommendation"]["name"] == "batch16_acc2"
    assert result["official_long_run_started"] is False
    directory = Path(json.loads(path.read_text())["directory"])
    assert not list((directory/"trials").rglob("*.pt"))
    assert not list((directory/"trials").rglob("*.pt.tmp"))
    assert len(list((directory/"trials").rglob("summary.json"))) == 3
    with pytest.raises(FileExistsError):
        benchmark.run_plan(path, True)


def test_controller_interrupt_stops_its_child_before_cleanup_and_releases_lock(project, monkeypatch):
    path = benchmark.create_plan(project[1], "20261025_153006")
    state = simulated_runtime(project, monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_plan(path, True)
    assert not state["locked"]
    directory = Path(json.loads(path.read_text())["directory"])
    assert not list((directory/"trials").rglob("*.pt"))
    assert not list((directory/"trials").rglob("*.pt.tmp"))


def test_accumulation_profile_matches_total_warmup_eval_and_logging_token_budgets(project):
    path = benchmark.create_plan(project[1], "20261025_153010", profile="accumulation64k")
    plan = benchmark.verified_plan(path)
    assert plan["profile"] == "accumulation64k"
    assert [row["name"] for row in plan["trials"]] == ["batch8_acc4", "batch8_acc8"]
    configs = [benchmark.load_config(row["config"]) for row in plan["trials"]]
    for row, config, steps, batch, first, maximum, warmup, interval, logging in zip(
            plan["trials"], configs, (24, 12), (32768, 65536), (6, 3),
            (91553, 45777), (1500, 750), (500, 250), (2, 1)):
        assert row["steps"] == steps and row["stable_first_step"] == first
        assert row["tokens_per_update"] == batch
        assert row["trial_token_budget"] == steps * batch == 786432
        assert (steps-first) * batch == 589824
        assert first * batch == 196608
        assert config.train.max_steps == maximum
        assert config.train.warmup_steps == warmup
        assert config.train.eval_interval == config.train.save_interval == interval
        assert config.train.log_interval == logging
        assert steps // logging == 12
        assert warmup * batch == 49152000
        assert interval * batch == 16384000
        assert config.train.eval_batches == 2
        assert config.train.micro_batch_size == 8 and not config.model.activation_checkpointing
    assert configs[1].train.max_steps * 65536 == 3000041472


def test_profile_stable_windows_measure_same_effective_tokens(project):
    path = benchmark.create_plan(project[1], "20261025_153011", profile="accumulation64k")
    plan = benchmark.verified_plan(path)
    for row in plan["trials"]:
        config = benchmark.load_config(row["config"])
        metrics = [{"step": step, "tokens_seen": step * row["tokens_per_update"],
                    "elapsed_seconds_this_session": 100 + step * row["tokens_per_update"] / 65536}
                   for step in range(config.train.log_interval, row["steps"]+1, config.train.log_interval)]
        stable = benchmark.summarize_metrics(metrics, row["stable_first_step"])
        assert len(metrics) == 12
        assert stable["tokens"] == 589824 and stable["tokens_per_second"] == 65536


def test_legacy_plan_without_profile_or_per_trial_fields_remains_readable(project):
    path = benchmark.create_plan(project[1], "20261025_153012")
    old = json.loads(path.read_text())
    old.pop("profile")
    for row in old["trials"]:
        for field in ("steps", "stable_first_step", "tokens_per_update", "trial_token_budget"):
            row.pop(field)
    path.write_text(json.dumps(old))
    assert len(benchmark.verified_plan(path)["trials"]) == 3


def test_accumulation_profile_rejects_unequal_probe_steps_or_modified_trial(project):
    with pytest.raises(ValueError, match="固定 24/12"):
        benchmark.create_plan(project[1], "20261025_153013", steps=16, profile="accumulation64k")
    path = benchmark.create_plan(project[1], "20261025_153014", profile="accumulation64k")
    plan = json.loads(path.read_text())
    plan["trials"][1]["steps"] = 24
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Token 预算"):
        benchmark.verified_plan(path)


def test_worker_uses_per_trial_steps_and_window_without_running_gpu(project, monkeypatch):
    path = benchmark.create_plan(project[1], "20261025_153015", profile="accumulation64k")
    monkeypatch.setattr(benchmark, "require_linux", lambda: None)
    monkeypatch.setenv("LLM_LAB_BENCHMARK_PLAN", str(path.resolve()))
    monkeypatch.setenv("LLM_LAB_BENCHMARK_PARENT", str(os.getppid()))
    calls = []
    def fake_engine(config, stop_after_steps):
        calls.append((config.train.max_steps, stop_after_steps))
        output = Path(config.output_dir); output.mkdir(parents=True)
        rows = [{"step": step, "tokens_seen": step * 65536,
                 "elapsed_seconds_this_session": 100 + step}
                for step in range(1, stop_after_steps+1)]
        (output/"metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        (output/"last.pt").write_bytes(b"tiny fixture checkpoint")
        return {"step": stop_after_steps, "status": "simulated_interruption"}
    monkeypatch.setattr("llm_lab.engine.pretrain.run_pretrain", fake_engine)
    result = benchmark.worker(path, "batch8_acc8")
    assert calls == [(45777, 12)]
    assert result["status"] == "passed" and result["checkpoint_saved"]
    assert result["stable"]["first_step"] == 3
    assert result["stable"]["last_step"] == 12
    assert result["stable"]["tokens"] == 589824


@pytest.mark.parametrize("steps", [0, 11, 33, True])
def test_probe_length_is_bounded_before_any_plan_creation(project, steps):
    with pytest.raises(ValueError, match="12..32"):
        benchmark.create_plan(project[1], "20261025_153004", steps)
