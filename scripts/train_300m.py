"""300M 独立预训练启动器：冻结配置、单进程租用 GPU、保持 100M 源码身份。

prepare/status 只读写小型配置与报告；GPU 训练由用户显式执行 probe/start/resume。
核心模型、优化器和断点逻辑复用 llm_lab，不在 scripts 中复制训练循环。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from llm_lab.common.io import read_json, sha256_file, write_json
from llm_lab.config import load_config

MANIFEST_SHA = "f1dbbba35db3f414a36359f96c3a8a6a3b4bba1fb94b09284a5a0fef75c6c7b1"
TOKENIZER_SHA = "6040c90c045836982a47bc0544942f122d0446bf9ed4d2ac1929befd0d4b86b3"
GIB = 1024**3
ACTIONS = {"probe", "start", "resume"}
TARGET_TOKENS = 3_000_000_000
PARAMETER_COUNT = 303_350_784
ALLOWED_UPDATE_TOKENS = {16_384, 32_768}


def source_hash():
    from llm_lab.engine.pretrain import source_fingerprint
    return source_fingerprint()


def run_engine(config, **options):
    from llm_lab.engine.pretrain import run_pretrain
    return run_pretrain(config, **options)


def under(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"路径越出项目或是符号链接：{path}")
    return path.resolve()


def paths(config_path):
    config_path = Path(config_path).resolve()
    return {"config": config_path, "identity": config_path.with_suffix(".identity.json"),
            "state": config_path.with_suffix(".state.json"), "env": config_path.with_suffix(".env")}


def corpus_identity(config):
    root = Path(config.project_root)
    manifest_path = under(config.data_manifest, root)
    if sha256_file(manifest_path) != MANIFEST_SHA:
        raise ValueError("必须使用已验收、固定 SHA256 的 bilingual_3b manifest")
    manifest = read_json(manifest_path)
    token = manifest["tokenizer"]
    tokenizer_path = under(manifest_path.parent / token["path"], manifest_path.parent)
    if manifest.get("acceptance", {}).get("status") != "passed" or token["vocab_size"] != 32000:
        raise ValueError("语料验收状态或词表不符")
    if token["sha256"] != TOKENIZER_SHA or sha256_file(tokenizer_path) != TOKENIZER_SHA:
        raise ValueError("Tokenizer 身份不符")
    if any(manifest["splits"]["train"][lang]["token_count"] < 1_500_000_000 for lang in ("en", "zh")):
        raise ValueError("中英训练池容量不足")
    return manifest


def plan_details(config):
    """由真实 micro batch/累积数推导处理量，避免把更新次数误当 Token 预算。"""
    train = config.train
    tokens_per_update = train.micro_batch_size * train.gradient_accumulation_steps * train.seq_len
    return {"tokens_per_update": tokens_per_update,
            "planned_tokens": train.max_steps * tokens_per_update,
            "parameter_count": PARAMETER_COUNT}


def validate_plan(config):
    config.validate()
    model, train, budget = config.model, config.train, config.budget
    if (model.dim, model.n_layers, model.n_heads, model.n_kv_heads, model.ffn_dim, model.vocab_size) != (1024, 24, 16, 4, 2816, 32000) or not model.tie_embeddings:
        raise ValueError("此启动器只接受已核验的 303,350,784 参数结构")
    if train.device != "cuda" or train.dtype != "bf16" or train.seq_len != 1024:
        raise ValueError("300M 计划必须为 CUDA/BF16 和 1024 上下文")
    details = plan_details(config)
    tokens_per_update = details["tokens_per_update"]
    if tokens_per_update not in ALLOWED_UPDATE_TOKENS:
        raise ValueError("每次更新必须处理 16384 或 32768 Token")
    expected_steps = (TARGET_TOKENS + tokens_per_update - 1) // tokens_per_update
    if train.max_steps != expected_steps or not TARGET_TOKENS <= details["planned_tokens"] < TARGET_TOKENS + tokens_per_update:
        raise ValueError(f"{tokens_per_update} Token/更新的 3B 计划必须为 {expected_steps} 个完整更新")
    rate = budget.get("hourly_rate_cny")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("填写实际 hourly_rate_cny 后才可冻结 GPU 计划")
    if budget.get("project_limit_cny") != 500 or budget.get("reserve_cny") != 100:
        raise ValueError("保留项目 500 元总额与 100 元余量")


def prepare(template, stamp=None):
    config = load_config(template)
    root = Path(config.project_root).resolve()
    validate_plan(config)
    corpus_identity(config)
    stamp = stamp or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", stamp):
        raise ValueError("时间戳必须为 YYYYmmdd_HHMMSS")
    config.output_dir = str(root / "runs" / f"pretrain_300m_3b_{stamp}")
    config.train.wandb_run_name = f"pretrain-300m-3b-zh-hans-en-{stamp}"
    output = under(config.output_dir, root)
    if output.exists():
        raise FileExistsError("新实验输出目录已存在；请换时间戳，旧断点使用 resume")
    frozen = root / "runs/configs" / f"pretrain_300m_3b_{stamp}.json"
    related = paths(frozen)
    if any(p.exists() or p.is_symlink() for p in related.values()):
        raise FileExistsError("同名冻结文件已存在，拒绝覆盖")
    frozen.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    # 独占创建拒绝同时间戳竞争；失败留下可见证据，不覆盖任何旧文件。
    with frozen.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    identity = {"config_sha256": sha256_file(frozen), "manifest_sha256": MANIFEST_SHA,
                "tokenizer_sha256": TOKENIZER_SHA, "source_sha256": source_hash(),
                "output_dir": str(output), "stamp": stamp, **plan_details(config)}
    with related["identity"].open("x", encoding="utf-8") as handle:
        json.dump(identity, handle, ensure_ascii=False, indent=2); handle.write("\n")
    values = {"RUN_STAMP": stamp, "RUN_DIR": str(output), "RUN_CONFIG": str(frozen),
              "RUN_LOG": str(output / f"console_{stamp}.log")}
    text = "#!/usr/bin/env bash\n" + f"cd {shlex.quote(str(root))}\nsource {shlex.quote(str(root / 'scripts/env_remote.sh'))}\n"
    text += "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0\n"
    text += "".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())
    with related["env"].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return {**values, "identity": str(related["identity"]), "gpu_training_started": False}


def frozen_config(config_path):
    related = paths(config_path)
    config = load_config(related["config"])
    root = Path(config.project_root)
    # 清空过的时间戳只保留历史记录；禁止用旧输出名重新训练并混用旧预算/W&B。
    if related["state"].is_file() and read_json(related["state"]).get("status") == "cleared":
        raise ValueError("该实验输出已清空，请准备新的时间戳，不能重新启动或恢复")
    identity = read_json(related["identity"])
    expected = root / "runs" / f"pretrain_300m_3b_{identity['stamp']}"
    if under(config.output_dir, root) != expected or identity["output_dir"] != str(expected):
        raise ValueError("冻结输出必须是独立的 300M 时间戳目录")
    if sha256_file(related["config"]) != identity["config_sha256"] or source_hash() != identity["source_sha256"]:
        raise ValueError("冻结配置或训练源码已改变，拒绝混入当前实验")
    if identity["manifest_sha256"] != MANIFEST_SHA or identity["tokenizer_sha256"] != TOKENIZER_SHA:
        raise ValueError("冻结数据身份不符")
    validate_plan(config)
    # 旧冻结身份没有这些扩展字段时仍兼容；新身份存在时必须与配置一致。
    for key, expected_value in plan_details(config).items():
        if key in identity and identity[key] != expected_value:
            raise ValueError("冻结训练计划身份不符")
    corpus_identity(config)
    return config


def require_linux():
    if sys.platform != "linux":
        raise RuntimeError("check/probe/start/resume 只能在远程 Linux GPU 实例执行")


def training_processes(proc=Path("/proc")):
    excluded, current = {os.getpid()}, os.getppid()
    while current > 1 and current not in excluded:
        excluded.add(current)
        try:
            row = next(s for s in (proc / str(current) / "status").read_text().splitlines() if s.startswith("PPid:"))
            current = int(row.split()[1])
        except (OSError, StopIteration, ValueError):
            break
    found = []
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) in excluded:
            continue
        try:
            argv = (directory / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if ("llm_lab.cli" in argv and "pretrain" in argv) or (any(Path(a).name == "train_300m.py" for a in argv) and ACTIONS.intersection(argv)):
            found.append({"pid": int(directory.name), "args": " ".join(argv).strip()})
    return found


def gpu_idle():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("无法查询 GPU 计算进程，请先检查 nvidia-smi")
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not row.isdecimal() for row in rows):
        raise RuntimeError("GPU 查询未返回可识别 PID，不能确认空闲")
    if rows:
        raise RuntimeError("GPU 已有计算进程，先结束另一轮训练：" + ", ".join(rows))


@contextmanager
def training_lock(root):
    import fcntl
    lock = under(Path(root) / "runs/.gpu_training.lock", root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("另一个训练启动器持有项目 GPU 锁") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    # 不删除锁文件，避免另一个进程持有旧 inode 时出现第二把锁。


def required_free_bytes(output, resume=False):
    reserved = 0
    seen = set()
    if resume:
        for name in ("last.pt", "best.pt"):
            path = Path(output) / name
            if path.is_file() and not path.is_symlink():
                stat = path.stat(); key = (stat.st_dev, stat.st_ino)
                if key not in seen:
                    reserved += stat.st_size; seen.add(key)
    return max(8 * GIB, 20 * GIB - reserved)


def preflight(config, action):
    busy = training_processes()
    if busy:
        raise RuntimeError("已有训练进程，拒绝并行：" + json.dumps(busy, ensure_ascii=False))
    gpu_idle()
    output = Path(config.output_dir)
    resume = action == "resume" or (action == "check" and (output / "last.pt").is_file())
    if resume and (not (output / "last.pt").is_file() or (output / "last.pt").is_symlink()):
        raise ValueError("resume 只允许本实验原目录中的 last.pt")
    if action in {"probe", "start"} and output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空；续训使用 resume，改变配置则另建实验")
    need = required_free_bytes(output, resume)
    free = shutil.disk_usage(config.project_root).free
    if free < need:
        raise RuntimeError(f"磁盘余量不足：需要 {need / GIB:.2f} GiB，实际 {free / GIB:.2f} GiB")
    ledger_path = Path(config.budget["ledger_path"])
    ledger = read_json(ledger_path) if ledger_path.exists() else {"runs": {}}
    recorded = sum(float(row["estimated_compute_cny"]) for row in ledger["runs"].values())
    run_recorded = float(ledger["runs"].get(str(output.resolve()), {}).get("estimated_compute_cny", 0))
    available = config.budget["project_limit_cny"] - config.budget["reserve_cny"] - recorded
    if available <= 0 or run_recorded >= config.budget["run_limit_cny"]:
        raise RuntimeError("记录的实验或项目训练预算已耗尽")
    return {"free_gib": free / GIB, "required_free_gib": need / GIB,
            "checkpoint_main_tensor_gib": PARAMETER_COUNT * 12 / GIB,
            **plan_details(config),
            "budget": config.budget, "recorded_project_compute_cny": recorded,
            "available_recorded_compute_cny": available, "gpu_training_started": False}


def execute(action, config_path, *, probe_steps=32):
    require_linux()
    config = frozen_config(config_path)
    if action == "probe" and (type(probe_steps) is not int or not 1 <= probe_steps < config.train.max_steps):
        raise ValueError("probe steps 必须是小于正式 max_steps 的正整数，正式学习率计划不变")
    with training_lock(config.project_root):
        checks = preflight(config, action)
        if action == "check":
            return checks
        state_path = paths(config_path)["state"]
        state = {"status": "running", "action": action, "pid": os.getpid(),
                 "config": str(Path(config_path).resolve()), "started_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(state_path, state)
        try:
            options = {"stop_after_steps": probe_steps} if action == "probe" else {}
            if action == "resume":
                options["resume"] = Path(config.output_dir) / "last.pt"
            result = run_engine(config, **options)
            write_json(state_path, {**state, "status": "exited", "summary": result})
            return result
        except BaseException as error:
            write_json(state_path, {**state, "status": "failed", "error_type": type(error).__name__})
            raise


def status(config_path):
    related = paths(config_path)
    config = load_config(related["config"])
    output = Path(config.output_dir)
    result = {"output_dir": str(output), "last_checkpoint_exists": (output / "last.pt").is_file()}
    for name, path in (("launcher", related["state"]), ("summary", output / "summary.json"), ("wandb", output / "wandb_run.json")):
        if path.is_file():
            result[name] = read_json(path)
    if result.get("launcher", {}).get("status") == "running":
        pid = result["launcher"]["pid"]
        result["pid_alive"] = None
        if sys.platform == "linux":
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
                result["pid_alive"] = any(Path(a).name == "train_300m.py" for a in argv) and bool(ACTIONS.intersection(argv))
            except OSError:
                result["pid_alive"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--template", default="configs/pretrain/300m_3b.json")
    prepare_command.add_argument("--stamp")
    for command in ("check", "probe", "start", "resume", "status"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True)
        if command == "probe":
            sub.add_argument("--steps", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.template, args.stamp) if args.action == "prepare" else status(args.config) if args.action == "status" else execute(args.action, args.config, probe_steps=getattr(args, "steps", 32))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"错误：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
