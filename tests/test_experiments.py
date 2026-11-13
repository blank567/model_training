from pathlib import Path

import pytest

from llm_lab.common.io import read_json, write_json
from llm_lab.experiments import build_plan


def test_all_trials_process_equal_tokens(tmp_path):
    spec = read_json("configs/experiments/pretrain_ablation.json")
    spec["output_root"] = str(tmp_path / "trials")
    path = tmp_path / "spec.json"
    write_json(path, spec)
    plan = build_plan(path, hourly_rate_cny=2.0)
    assert len(plan) == 5
    assert {trial["optimizer_updates"] * trial["tokens_per_update"] for trial in plan} == {262144}
    assert read_json(Path(plan[0]["config"]))["budget"]["hourly_rate_cny"] == 2.0
    assert not (tmp_path / "trials" / "baseline" / "last.pt").exists()


def test_fractional_update_budget_is_rejected(tmp_path):
    spec = read_json("configs/experiments/pretrain_ablation.json")
    spec.update(output_root=str(tmp_path / "trials"), tokens_per_trial=123)
    path = tmp_path / "spec.json"
    write_json(path, spec)
    with pytest.raises(ValueError, match="divisible"):
        build_plan(path)

