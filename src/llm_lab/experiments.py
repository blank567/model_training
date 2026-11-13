"""生成等 Token 的预训练对照计划；默认只落配置，不启动训练。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .common.io import read_json, write_json
from .config import load_config


def build_plan(path: str | Path, hourly_rate_cny: float | None = None) -> list[dict]:
    path = Path(path).resolve()
    spec = read_json(path)
    root = next((parent for parent in path.parents if (parent / "pyproject.toml").is_file()), Path.cwd()).resolve()
    baseline = load_config(root / spec["base_config"]).to_dict()
    output_root = (root / spec["output_root"]).resolve()
    tokens = int(spec["tokens_per_trial"])
    result = []
    for trial in spec["trials"]:
        name = trial["name"]
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in name):
            raise ValueError("Trial names must be lowercase safe directory names")
        config = copy.deepcopy(baseline)
        for section, changes in trial.get("overrides", {}).items():
            if section not in ("model", "train"):
                raise ValueError("Only model/train trial overrides are allowed")
            config[section].update(changes)
        per_update = config["train"]["seq_len"] * config["train"]["micro_batch_size"] * config["train"]["gradient_accumulation_steps"]
        if tokens <= 0 or tokens % per_update:
            raise ValueError(f"{name}: tokens_per_trial must be divisible by tokens_per_update={per_update}")
        updates = tokens // per_update
        config["train"].update(max_steps=updates, warmup_steps=max(1, updates // 10),
                               eval_interval=max(1, updates // 2), save_interval=max(1, updates // 2),
                               log_interval=max(1, updates // 10))
        config["output_dir"] = str(output_root / name)
        if hourly_rate_cny is not None:
            config["budget"]["hourly_rate_cny"] = hourly_rate_cny
        destination = output_root / "configs" / f"{name}.json"
        write_json(destination, config)
        load_config(destination)  # 统一经过严格校验，不能用覆盖项绕过 schema。
        result.append({"name": name, "config": str(destination), "tokens_per_trial": tokens,
                       "tokens_per_update": per_update, "optimizer_updates": updates,
                       "selection_split": "validation", "status": "planned_not_executed"})
    if len({trial["name"] for trial in result}) != len(result):
        raise ValueError("Trial names must be unique")
    write_json(output_root / "plan.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hourly-rate-cny", type=float)
    parser.add_argument("--execute", action="store_true", help="Run the planned GPU experiments sequentially")
    args = parser.parse_args(argv)
    plan = build_plan(args.config, args.hourly_rate_cny)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.execute:
        from .engine.pretrain import run_pretrain
        for trial in plan:
            result = run_pretrain(load_config(trial["config"]))
            if result["status"] != "completed":
                print("Trial stopped before its Token budget; remaining trials are not executed.", flush=True)
                break


if __name__ == "__main__":
    main()

