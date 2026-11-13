"""300M 单卡受限批次测速；plan 不训练，run --execute 才执行独立短测。

默认候选每步均为 32768 Token；accumulation64k 按相同 Token 总量比较 8×4/8×8。
测速权重在子进程退出后删除，仅保留配置和报告；
吞吐与显存用于选择运行配置，不用于判断模型质量，也不启动正式长跑。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from llm_lab.common.io import read_json, sha256_file, write_json
from llm_lab.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_300m import corpus_identity, gpu_idle, require_linux, source_hash, training_lock, training_processes

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
VARIANTS = (("batch8_acc4", 8, 4, False), ("batch16_acc2", 16, 2, False),
            ("batch16_acc2_checkpoint", 16, 2, True))
PROFILES = ("balanced32k", "accumulation64k")
WEIGHTS = ("last.pt", "best.pt", "last.pt.tmp", "best.pt.tmp")


def inside(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError("测速路径越界或为符号链接")
    return path.resolve()


def trial_specs(profile, steps):
    if profile not in PROFILES:
        raise ValueError("未知测速 profile")
    if type(steps) is not int or not 12 <= steps <= 32:
        raise ValueError("短测 steps 必须在 12..32，不能替代正式训练")
    if profile == "accumulation64k":
        if steps != 24:
            raise ValueError("accumulation64k 固定 24/12 步，保持两组 Token 预算相等")
        variants = (("batch8_acc4", 8, 4, False, 24, 6),
                    ("batch8_acc8", 8, 8, False, 12, 3))
    else:
        variants = [(*row, steps, 6) for row in VARIANTS]
    return [{"name": name, "micro": micro, "accum": accum, "checkpoint": checkpoint,
             "steps": count, "stable_first_step": first,
             "max_steps": 45777 if accum == 8 else 91553,
             "warmup_steps": 750 if accum == 8 else 1500,
             "eval_interval": 250 if accum == 8 else 500,
             "log_interval": 1 if profile == "accumulation64k" and accum == 8 else 2}
            for name, micro, accum, checkpoint, count, first in variants]


def create_plan(template, stamp=None, steps=24, profile="balanced32k"):
    config = load_config(template)
    root = Path(config.project_root).resolve()
    if root != ROOT.resolve():
        raise ValueError("测速必须使用当前项目根目录")
    specs = trial_specs(profile, steps)
    stamp = stamp or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", stamp):
        raise ValueError("时间戳必须为 YYYYmmdd_HHMMSS")
    directory = inside(root / "runs" / f"benchmark_300m_{stamp}", root / "runs")
    if directory.exists():
        raise FileExistsError("同名测速目录已存在；新测速使用新时间戳")
    config.model.validate()
    if (config.model.dim, config.model.n_layers, config.model.n_heads, config.model.n_kv_heads,
        config.model.ffn_dim, config.model.vocab_size) != (1024, 24, 16, 4, 2816, 32000):
        raise ValueError("只测已核验的 303,350,784 参数结构")
    if config.train.device != "cuda" or config.train.dtype != "bf16" or config.train.seq_len != 1024:
        raise ValueError("测速要求 CUDA/BF16 和 1024 上下文")
    rate = config.budget.get("hourly_rate_cny")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate <= 10:
        raise ValueError("填写有限、有效的实例单价后再制定测速计划")
    directory.mkdir(parents=True)
    (directory / "configs").mkdir()
    trials = []
    for spec in specs:
        name, micro, accum, checkpoint = (spec[key] for key in ("name", "micro", "accum", "checkpoint"))
        raw = config.to_dict()
        raw["output_dir"] = str(directory / "trials" / name)
        raw["model"]["activation_checkpointing"] = checkpoint
        raw["train"].update(micro_batch_size=micro, gradient_accumulation_steps=accum,
                            max_steps=spec["max_steps"], warmup_steps=spec["warmup_steps"],
                            eval_interval=spec["eval_interval"], eval_batches=16384 // (micro * 1024),
                            save_interval=spec["eval_interval"], log_interval=spec["log_interval"],
                            wandb=False, tensorboard=False)
        raw["budget"].update(run_limit_cny=1, project_limit_cny=500, reserve_cny=100,
                             max_runtime_seconds=150, ledger_path=str(root / "runs/budget_ledger.json"))
        path = directory / "configs" / f"{name}.json"
        write_json(path, raw)
        trials.append({"name": name, "config": str(path), "config_sha256": sha256_file(path),
                       "output_dir": raw["output_dir"], "report": str(directory / f"{name}.json"),
                       "steps": spec["steps"], "stable_first_step": spec["stable_first_step"],
                       "tokens_per_update": micro * accum * 1024,
                       "trial_token_budget": spec["steps"] * micro * accum * 1024})
    plan = {"schema_version": 1, "kind": "bounded_300m_batch_benchmark", "project_root": str(root),
            "directory": str(directory), "steps": steps, "profile": profile,
            "tokens_per_update": 32768 if profile == "balanced32k" else None,
            "planned_tokens": 3000008704 if profile == "balanced32k" else None,
            "per_trial_timeout_seconds": 180,
            "total_timeout_seconds": 360, "min_free_disk_bytes": 12 * GIB,
            "source_sha256": source_hash(), "benchmark_sha256": sha256_file(Path(__file__)),
            "trials": trials, "gpu_training_started": False}
    write_json(directory / "plan.json", plan)
    return directory / "plan.json"


def verified_plan(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("计划不能是符号链接")
    plan = read_json(path)
    root = Path(plan["project_root"]).resolve()
    directory = inside(plan["directory"], root / "runs")
    if root != ROOT.resolve() or not directory.name.startswith("benchmark_300m_") or path.resolve() != directory / "plan.json":
        raise ValueError("无效测速计划根目录")
    if plan.get("schema_version") != 1 or plan.get("kind") != "bounded_300m_batch_benchmark":
        raise ValueError("无效测速计划协议")
    if plan["benchmark_sha256"] != sha256_file(Path(__file__)) or plan["source_sha256"] != source_hash():
        raise ValueError("测速脚本或训练源码身份变化，请重新制定计划")
    specs = trial_specs(plan.get("profile", "balanced32k"), plan["steps"])
    if plan["total_timeout_seconds"] != 360 or plan["per_trial_timeout_seconds"] != 180:
        raise ValueError("测速上限被修改")
    if plan["min_free_disk_bytes"] != 12 * GIB or len(plan["trials"]) != len(specs):
        raise ValueError("测速余量或候选数量被修改")
    for item, spec in zip(plan["trials"], specs):
        name, micro, accum, checkpoint = (spec[key] for key in ("name", "micro", "accum", "checkpoint"))
        config_path = inside(item["config"], directory / "configs")
        output = inside(item["output_dir"], directory / "trials")
        report = inside(item["report"], directory)
        if item["name"] != name or output.name != name or report != directory / f"{name}.json":
            raise ValueError("候选输出范围不符")
        if sha256_file(config_path) != item["config_sha256"]:
            raise ValueError("候选配置 SHA256 已改变")
        config = load_config(config_path)
        train, model, budget = config.train, config.model, config.budget
        if (train.micro_batch_size, train.gradient_accumulation_steps, model.activation_checkpointing) != (micro, accum, checkpoint):
            raise ValueError("候选批次与重算开关不符")
        if (item.get("steps", plan["steps"]) != spec["steps"]
                or item.get("stable_first_step", 6) != spec["stable_first_step"]
                or item.get("tokens_per_update", micro * accum * 1024) != micro * accum * 1024
                or item.get("trial_token_budget", spec["steps"] * micro * accum * 1024) != spec["steps"] * micro * accum * 1024):
            raise ValueError("候选短测步数或 Token 预算被修改")
        if Path(config.output_dir) != output or train.max_steps != spec["max_steps"] or train.warmup_steps != spec["warmup_steps"]:
            raise ValueError("正式学习率计划或候选输出不符")
        if ((train.device, train.dtype, train.seq_len, train.eval_interval, train.save_interval,
             train.eval_batches, train.log_interval) != ("cuda", "bf16", 1024, spec["eval_interval"],
                                                        spec["eval_interval"], 16384 // (micro * 1024), spec["log_interval"])
                or train.wandb or train.tensorboard or budget["run_limit_cny"] != 1
                or budget["project_limit_cny"] != 500 or budget["reserve_cny"] != 100
                or budget["max_runtime_seconds"] != 150):
            raise ValueError("测速日志/预算保护被更改")
    return plan


def summarize_metrics(rows, first_step=6):
    window = [row for row in rows if row["step"] >= first_step]
    if len(window) < 2:
        raise ValueError("稳态窗口记录不足")
    first, last = window[0], window[-1]
    elapsed = last["elapsed_seconds_this_session"] - first["elapsed_seconds_this_session"]
    tokens = last["tokens_seen"] - first["tokens_seen"]
    if elapsed <= 0 or tokens <= 0:
        raise ValueError("稳态时间或 Token 增量非法")
    return {"first_step": first["step"], "last_step": last["step"],
            "seconds": elapsed, "tokens": tokens, "tokens_per_second": tokens / elapsed}


def choose_best(results):
    eligible = []
    for row in results:
        total, peak = row.get("gpu_total_bytes", 0), row.get("peak_reserved_bytes", 0)
        margin = max(total * .15, 6 * GIB)
        if (row.get("status") == "passed" and row.get("checkpoint_saved")
                and 0 < peak <= total - margin and row.get("stable", {}).get("tokens_per_second", 0) > 0):
            eligible.append(row)
    if not eligible:
        return None
    best = max(eligible, key=lambda row: row["stable"]["tokens_per_second"])
    return {"name": best["name"], "tokens_per_second": best["stable"]["tokens_per_second"],
            "peak_reserved_gib": best["peak_reserved_bytes"] / GIB,
            "headroom_gib": (best["gpu_total_bytes"] - best["peak_reserved_bytes"]) / GIB,
            "rule": "highest measured throughput with >=15% and >=6GiB VRAM headroom"}


def cleanup_weights(output, plan_directory):
    directory = Path(plan_directory).resolve()
    output = inside(output, directory / "trials")
    removed = []
    if not output.exists():
        return removed
    for name in WEIGHTS:
        path = output / name
        if path.is_symlink():
            raise ValueError("拒绝删除符号链接测速权重")
        if path.exists():
            inside(path, directory / "trials")
            if not path.is_file():
                raise ValueError("测速权重必须为普通文件")
            removed.append({"name": name, "bytes": path.stat().st_size})
            path.unlink()
    return removed


def worker(plan_path, trial_name):
    require_linux()
    if os.environ.get("LLM_LAB_BENCHMARK_PLAN") != str(Path(plan_path).resolve()) or os.environ.get("LLM_LAB_BENCHMARK_PARENT") != str(os.getppid()):
        raise RuntimeError("内部 worker 只能由持锁测速控制器调用")
    plan = verified_plan(plan_path)
    item = next(row for row in plan["trials"] if row["name"] == trial_name)
    config = load_config(item["config"])
    steps, first_step = item.get("steps", plan["steps"]), item.get("stable_first_step", 6)
    tokens_per_update = config.train.micro_batch_size * config.train.gradient_accumulation_steps * config.train.seq_len
    report = {"name": trial_name, "status": "failed", "checkpoint_saved": False,
              "profile": plan.get("profile", "balanced32k"), "steps": steps,
              "stable_first_step": first_step, "tokens_per_update": tokens_per_update,
              "trial_token_budget": steps * tokens_per_update}
    import torch
    started = time.monotonic()
    try:
        from llm_lab.engine.pretrain import run_pretrain
        summary = run_pretrain(config, stop_after_steps=steps)
        output = Path(config.output_dir)
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        if summary["step"] != steps or rows[-1]["step"] != steps:
            raise ValueError("短测未完成指定步数，不能推荐不完整结果")
        report.update(status="passed", summary=summary, stable=summarize_metrics(rows, first_step),
                      checkpoint_saved=(output / "last.pt").is_file() and (output / "last.pt").stat().st_size > 0)
    except BaseException as error:
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        report.update(status="oom" if oom else "failed", error_type=type(error).__name__, error=str(error)[:2000])
    finally:
        if torch.cuda.is_initialized():
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(item["report"], report)
    return report


def stop_child(child):
    if child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def run_plan(plan_path, execute=False):
    if not execute:
        return {"status": "not_started", "plan": str(Path(plan_path).resolve()),
                "message": "使用 run --plan PLAN --execute 才执行 GPU 短测"}
    require_linux()
    plan = verified_plan(plan_path)
    root, directory = Path(plan["project_root"]), Path(plan["directory"])
    result_path = directory / "results.json"
    if result_path.exists() or any(Path(row["output_dir"]).exists() for row in plan["trials"]):
        raise FileExistsError("测速已经执行或留下中间结果；请用新计划保留原证据")
    with training_lock(root):
        if training_processes():
            raise RuntimeError("已有模型训练，拒绝并行测速")
        gpu_idle()
        corpus_identity(load_config(plan["trials"][0]["config"]))
        started = time.monotonic()
        results = []
        status = "completed"
        for item in plan["trials"]:
            remaining = plan["total_timeout_seconds"] - (time.monotonic() - started)
            if remaining < 20:
                status = "time_limit"; break
            if shutil.disk_usage(root).free < plan["min_free_disk_bytes"] + 4 * GIB:
                status = "disk_limit"; break
            gpu_idle()
            environment = os.environ.copy()
            environment.update(LLM_LAB_BENCHMARK_PLAN=str(Path(plan_path).resolve()),
                               LLM_LAB_BENCHMARK_PARENT=str(os.getpid()),
                               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", CUDA_VISIBLE_DEVICES="0")
            args = [sys.executable, "-u", str(Path(__file__).resolve()), "_worker",
                    "--plan", str(Path(plan_path).resolve()), "--trial", item["name"]]
            child = None; timed_out = False; removed = []
            try:
                with (directory / f"{item['name']}.log").open("xb") as log:
                    child = subprocess.Popen(args, cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        child.wait(timeout=min(plan["per_trial_timeout_seconds"], remaining - 15))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        stop_child(child)
            finally:
                if child is not None:
                    stop_child(child)
                removed = cleanup_weights(item["output_dir"], directory)
            report = read_json(item["report"]) if Path(item["report"]).exists() else {"name": item["name"], "status": "worker_failed"}
            if timed_out:
                report["status"] = "time_limit"
            elif child.returncode != 0 and report.get("status") == "passed":
                report["status"] = "worker_failed"
            report["returncode"] = child.returncode
            report["removed_weights"] = removed
            results.append(report)
            payload = {"status": "running", "results": results, "elapsed_seconds": time.monotonic() - started,
                       "profile": plan.get("profile", "balanced32k"),
                       "limit": "bounded throughput/memory benchmark; not validation quality or full-run ETA"}
            write_json(result_path, payload)
            print(json.dumps({"finished": item["name"], "status": report["status"],
                              "stable": report.get("stable"), "peak_reserved_bytes": report.get("peak_reserved_bytes")}), flush=True)
        final = {"status": status, "results": results, "recommendation": choose_best(results),
                 "profile": plan.get("profile", "balanced32k"),
                 "elapsed_seconds": time.monotonic() - started, "official_long_run_started": False,
                 "note": ("same 32768 tokens/update; memory safety and throughput only; first validation cycle not measured"
                          if plan.get("profile", "balanced32k") == "balanced32k" else
                          "equal 786432-token trials and 589824-token stable windows; global batch differs and changes optimization trajectory; no automatic formal-run change")}
        write_json(result_path, final)
        return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--template", default="configs/pretrain/300m_3b.json")
    plan.add_argument("--stamp")
    plan.add_argument("--steps", type=int, default=24)
    plan.add_argument("--profile", choices=PROFILES, default="balanced32k")
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--execute", action="store_true")
    internal = commands.add_parser("_worker", help=argparse.SUPPRESS)
    internal.add_argument("--plan", required=True)
    internal.add_argument("--trial", required=True, choices=sorted({item[0] for item in VARIANTS} | {"batch8_acc8"}))
    args = parser.parse_args(argv)
    if args.action == "plan":
        result = {"plan": str(create_plan(args.template, args.stamp, args.steps, args.profile)), "gpu_training_started": False}
    elif args.action == "run":
        result = run_plan(args.plan, args.execute)
    else:
        result = worker(args.plan, args.trial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "_worker" and result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
