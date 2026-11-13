"""检查可选监控不会丢失同一步日志、混淆恢复身份或推进训练 RNG。"""
import importlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from llm_lab.common.tracking import numeric_metrics, start_tracker, validate_tracking


def settings(mode="offline"):
    return {"wandb": True, "wandb_mode": mode, "wandb_project": "fixture-project"}


@pytest.mark.parametrize("field,value", [("wandb", "true"), ("wandb_mode", "ofline"), ("wandb_project", ""),
                                          ("wandb_entity", "team/project"), ("wandb_run_name", " ")])
def test_tracking_rejects_invalid_config(field, value):
    with pytest.raises(ValueError):
        validate_tracking({field: value})


def test_disabled_tracking_does_not_import_sdk_or_create_files(tmp_path, monkeypatch):
    original = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name: pytest.fail("disabled tracking imported SDK") if name == "wandb" else original(name))
    tracker = start_tracker({}, tmp_path / "run", tmp_path, job_type="test")
    tracker.log({"loss": 1.0}, 1)
    tracker.finish()
    assert not (tmp_path / "run").exists()


def test_online_resume_reuses_id_and_logs_multiple_events_at_same_optimizer_step(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setattr("llm_lab.common.tracking.configure_tracking_environment", lambda root: None)
    output = tmp_path / "run"
    tracker = start_tracker(settings("online"), output, tmp_path, job_type="sft_full")
    first, run = fake_wandb.calls[-1]
    tracker.log({"loss": 1.0}, 4)
    tracker.log({"eval_loss": 1.1}, 4)
    tracker.finish()
    resumed = start_tracker(settings("online"), output, tmp_path, job_type="sft_full", resume=True)
    second, _ = fake_wandb.calls[-1]
    assert first["resume"] == "never" and second["resume"] == "must"
    assert first["id"] == second["id"] and first["group"] == second["group"]
    assert second["entity"] == "fixture-owner"
    assert len(run.logs) == 2 and all(event["optimizer_step"] == 4 for event, _ in run.logs)
    assert all(options == {"commit": True} for _, options in run.logs)
    assert first["settings"].disable_code and first["settings"].disable_git and first["save_code"] is False
    resumed.finish()


def test_offline_resume_uses_separate_segments_in_same_experiment_group(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setattr("llm_lab.common.tracking.configure_tracking_environment", lambda root: None)
    first = start_tracker(settings(), tmp_path / "run", tmp_path, job_type="pretrain")
    first.finish()
    second = start_tracker(settings(), tmp_path / "run", tmp_path, job_type="pretrain", resume=True)
    initial, resumed = (call[0] for call in fake_wandb.calls)
    assert "resume" not in initial and "resume" not in resumed
    assert initial["id"] != resumed["id"] and initial["group"] == resumed["group"]
    state = json.loads((tmp_path / "run/wandb_run.json").read_text())
    assert state["segments"] == [initial["id"], resumed["id"]]
    second.finish()


def test_resume_refuses_missing_or_mismatched_tracking_identity(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setattr("llm_lab.common.tracking.configure_tracking_environment", lambda root: None)
    with pytest.raises(ValueError, match="identity"):
        start_tracker(settings(), tmp_path / "run", tmp_path, job_type="pretrain", resume=True)
    tracker = start_tracker(settings(), tmp_path / "run", tmp_path, job_type="pretrain")
    tracker.finish()
    with pytest.raises(ValueError, match="identity"):
        start_tracker(settings("online"), tmp_path / "run", tmp_path, job_type="pretrain", resume=True)


def test_only_numeric_metrics_are_sent():
    assert numeric_metrics({"loss": 1.0, "nested": {"score": 2}, "prompt": "private text", "path": "/data", "flag": True,
                            "bad": float("nan"), "infinite": float("inf"), "samples": ["private"]}) == {"loss": 1.0, "nested/score": 2}


def test_sdk_random_calls_leave_training_rng_unchanged(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setattr("llm_lab.common.tracking.configure_tracking_environment", lambda root: None)
    original = fake_wandb.init

    def random_calls():
        return random.random(), np.random.rand(), torch.rand(1)

    def init(**kwargs):
        random_calls()
        run = original(**kwargs)
        run.log = lambda *args, **options: random_calls()
        run.finish = lambda **options: random_calls()
        return run

    fake_wandb.init = init
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    python, numpy, tensor = random.getstate(), np.random.get_state(), torch.get_rng_state()
    tracker = start_tracker(settings(), tmp_path / "run", tmp_path, job_type="test")
    tracker.log({"loss": 2.0}, 1)
    tracker.finish()
    assert random.getstate() == python
    current = np.random.get_state()
    assert np.array_equal(current[1], numpy[1]) and current[2:] == numpy[2:]
    assert torch.equal(torch.get_rng_state(), tensor)


def test_logging_failure_keeps_training_alive_and_finish_is_idempotent(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setattr("llm_lab.common.tracking.configure_tracking_environment", lambda root: None)
    tracker = start_tracker(settings(), tmp_path / "run", tmp_path, job_type="test")
    run = tracker.run

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable SDK transport")

    run.log = fail
    with pytest.warns(RuntimeWarning, match="continues"):
        tracker.log({"loss": 1.0}, 1)
    tracker.log({"loss": 0.5}, 2)
    tracker.finish()
    tracker.finish()
    assert run.finishes == [{"exit_code": 1}]
    assert json.loads((tmp_path / "run/wandb_error.json").read_text())["status"] == "logging_failed"
