"""自研 300M 全参数 SFT：冻结实验、单 GPU 锁、显式启动与单轴计划。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from llm_lab.common.io import canonical_hash, read_json, sha256_file, write_json
from llm_sft.config import GIB, PARAMETER_COUNT, plan_details, read_config, runtime_fingerprint, under

ACTIONS = {"probe", "start", "resume"}
AXES = {"lr": [1e-5, 2e-5, 5e-5], "batch": [8, 16, 32], "epochs": [.5, 1., 2.],
        "warmup": [.01, .03, .05], "weight_decay": [0., .01, .1]}


def _number_token(value, *, scientific=False):
    """保留配置数字的十进制含义，生成适合目录的稳定数字标签。"""
    number = Decimal(str(value)).normalize()
    if scientific or number and (abs(number) < Decimal('0.0001') or abs(number) >= Decimal('1000000')):
        mantissa, exponent = format(number, 'e').split('e')
        return f"{mantissa.replace('.', 'p')}e{int(exponent)}"
    return format(number, 'f').replace('.', 'p')


def experiment_name(config, stamp):
    """由实际参数统一派生目录、配置、日志和W&B名称，时间戳保留独立重复实验。"""
    if not re.fullmatch(r"\d{8}_\d{6}", stamp):
        raise ValueError("时间戳必须为 YYYYmmdd_HHMMSS")
    train = config['train']
    micro, accum = train['micro_batch_size'], train['gradient_accumulation_steps']
    amount = 'ep' + _number_token(train['epochs'])
    if train['stop_epochs'] is not None:
        amount += '_stop' + _number_token(train['stop_epochs'])
    return (f"sft_300m_full_lr{_number_token(train['learning_rate'], scientific=True)}"
            f"_mb{micro}_ga{accum}_b{micro * accum}_{amount}"
            f"_wu{_number_token(train['warmup_ratio'])}_wd{_number_token(train['weight_decay'])}"
            f"_seed{train['seed']}_{stamp}")


def paths(config_path):
    path = Path(config_path).resolve()
    return {"config": path, "identity": path.with_suffix(".identity.json"),
            "env": path.with_suffix(".env"), "state": path.with_suffix(".state.json")}


def _file(root, name):
    path = under(name, root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def dataset_identity(config):
    """只核文件和元数据，不读取测试题或实例化模型。"""
    path = _file(config["project_root"], config["data_manifest"])
    if (path.parent / "SUPERSEDED.json").exists():
        raise ValueError("该 SFT 数据已废弃，使用 latest_sft.json 指向的正式发布")
    manifest = read_json(path)
    keys = ("schema_version", "kind", "splits", "tokenizer", "tokenization", "chinese_normalization")
    if manifest.get("kind") != "sft_tokenized_conversations" or canonical_hash({key: manifest[key] for key in keys}) != manifest.get("data_identity"):
        raise ValueError("SFT manifest 数据身份不符")
    if bool(manifest.get("fixture")) != config["fixture"]:
        raise ValueError("正式/fixture 实验与数据类型必须一致")
    spec = manifest.get("acceptance", {})
    if spec.get("status") != "passed":
        raise ValueError("SFT 数据尚未通过独立验收")
    report_path = _file(path.parent, spec.get("report", "acceptance.json"))
    if sha256_file(report_path) != spec.get("sha256"):
        raise ValueError("SFT 验收报告 SHA 不符")
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("data_identity") != manifest["data_identity"]:
        raise ValueError("SFT 验收报告绑定了其他数据")
    token = manifest["tokenizer"]
    if sha256_file(_file(path.parent, token["path"])) != token["sha256"]:
        raise ValueError("SFT tokenizer 身份不符")
    split_hashes = {}
    for split in (config["train_split"], "validation"):
        item = manifest["splits"][split]
        if type(item.get("examples")) is not int or item["examples"] <= 0:
            raise ValueError("SFT 分区样本数无效")
        actual = sha256_file(_file(path.parent, item["path"]))
        if actual != item["sha256"]:
            raise ValueError("SFT 分区文件 SHA 不符：" + split)
        split_hashes[split] = actual
    return {"manifest_sha256": sha256_file(path), "data_identity": manifest["data_identity"],
            "tokenizer_sha256": token["sha256"], "split_sha256": split_hashes,
            "train_examples": manifest["splits"][config["train_split"]]["examples"]}


def base_identity(config):
    path = _file(config["project_root"], config["base_checkpoint"])
    if not config["fixture"]:
        summary = read_json(_file(path.parent, "summary.json"))
        run = read_json(_file(path.parent, "run_manifest.json"))
        if summary.get("status") != "completed" or run.get("stage") != "pretrain":
            raise ValueError("SFT 必须从已完成预训练的 Base 开始")
        if run.get("parameters") != PARAMETER_COUNT:
            raise ValueError("生产入口只接受当前 303,350,784 参数自研 Base")
        previous = read_json(_file(path.parent, "config.json"))
        if summary.get("step") != previous.get("train", {}).get("max_steps"):
            raise ValueError("Base 预训练还没有完成冻结步数")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Base checkpoint 在核验时发生变化")
    return {"base_checkpoint_sha256": digest, "base_checkpoint_bytes": after.st_size}


def source_identity(config):
    base, data = base_identity(config), dataset_identity(config)
    runtime = runtime_fingerprint(config["project_root"])
    for key, current in (("base_checkpoint_sha256", base["base_checkpoint_sha256"]),
                         ("data_manifest_sha256", data["manifest_sha256"]),
                         ("data_identity", data["data_identity"]), ("runtime_sha256", runtime)):
        if config.get(key) is not None and config[key] != current:
            raise ValueError("冻结的 Base/数据/代码身份已变化：" + key)
    return {**base, **data, "runtime_sha256": runtime}


def prepare(template, stamp=None):
    config = read_config(template)
    root = Path(config["project_root"])
    identity = source_identity(config)
    stamp = stamp or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    name = experiment_name(config, stamp)
    output = under(root / "runs" / name, root)
    if output.exists():
        raise FileExistsError("实验目录已存在；新实验更换时间戳，原断点使用 resume")
    config["output_dir"] = str(output)
    config["train"]["wandb_run_name"] = name
    config.update(base_checkpoint_sha256=identity["base_checkpoint_sha256"],
                  data_manifest_sha256=identity["manifest_sha256"], data_identity=identity["data_identity"],
                  runtime_sha256=identity["runtime_sha256"])
    frozen = root / "runs/configs" / f"{name}.json"
    related = paths(frozen)
    if any(path.exists() or path.is_symlink() for path in related.values()):
        raise FileExistsError("冻结配置已存在，拒绝覆盖")
    frozen.parent.mkdir(parents=True, exist_ok=True)
    with frozen.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    frozen_identity = {**identity, "stamp": stamp, "name": name, "config_sha256": sha256_file(frozen),
                       "output_dir": str(output), "plan": plan_details(config, identity["train_examples"])}
    with related["identity"].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(frozen_identity, handle, ensure_ascii=False, indent=2); handle.write("\n")
    # shell 的 tee 会先打开文件，控制台日志放在独立目录，避免制造非空实验目录。
    log_directory = under(root / "runs/logs", root)
    values = {"RUN_STAMP": stamp, "RUN_NAME": name, "RUN_CONFIG": str(frozen), "RUN_DIR": str(output),
              "RUN_LOG": str(log_directory / f"{name}.log")}
    text = f"#!/usr/bin/env bash\ncd {shlex.quote(str(root))}\nsource {shlex.quote(str(root / 'scripts/env_remote.sh'))}\n"
    text += "export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8\n"
    text += f"mkdir -p {shlex.quote(str(log_directory))}\n"
    text += "".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())
    with related["env"].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return {**values, "env": str(related["env"]), "identity": str(related["identity"]),
            "plan": frozen_identity["plan"], "gpu_training_started": False}


def frozen_config(path):
    related = paths(path)
    config, identity = read_config(related["config"]), read_json(related["identity"])
    if related["state"].is_file():
        state = read_json(related["state"]).get("status")
        if state == "cleared":
            raise ValueError("该实验已清空，使用新的时间戳")
        if state == "superseded":
            raise ValueError("该实验配置已被替代，请从模板重新 prepare")
    if not identity.get('name'):
        raise ValueError("旧版实验配置未标注参数，请从模板重新 prepare 创建新实验")
    if sha256_file(related["config"]) != identity["config_sha256"]:
        raise ValueError("冻结配置已改变")
    name = experiment_name(config, identity['stamp'])
    if identity['name'] != name:
        raise ValueError("冻结实验名称与参数不符")
    expected = Path(config["project_root"]) / "runs" / name
    if (Path(config["output_dir"]) != expected or identity["output_dir"] != str(expected)
            or related['config'].name != name + '.json' or config['train']['wandb_run_name'] != name):
        raise ValueError("冻结输出、配置或W&B名称不是本实验的独立路径")
    current = source_identity(config)
    if any(current[key] != identity[key] for key in current):
        raise ValueError("冻结来源身份不符")
    if plan_details(config, current["train_examples"]) != identity["plan"]:
        raise ValueError("冻结训练计划不符")
    return config


def require_linux():
    if sys.platform != "linux":
        raise RuntimeError("GPU 启动/检查仅在远程 Linux 实例执行")


def training_processes(proc=Path("/proc")):
    excluded, ancestor = {os.getpid()}, os.getppid()
    while ancestor > 1 and ancestor not in excluded:
        excluded.add(ancestor)
        try:
            line = next(row for row in (proc / str(ancestor) / "status").read_text().splitlines() if row.startswith("PPid:"))
            ancestor = int(line.split()[1])
        except (OSError, ValueError, StopIteration):
            break
    found = []
    for path in proc.iterdir():
        if not path.name.isdecimal() or int(path.name) in excluded:
            continue
        try:
            argv = (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        known = any(Path(arg).name in {"train_sft.py", "train_300m.py"} for arg in argv) and bool(ACTIONS.intersection(argv))
        if known or "llm_lab.cli" in argv and "pretrain" in argv:
            found.append({"pid": int(path.name), "args": " ".join(argv).strip()})
    return found


def gpu_idle():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or any(not line.isdecimal() for line in rows):
        raise RuntimeError("无法确认 GPU 计算进程，请检查 nvidia-smi")
    if rows:
        raise RuntimeError("GPU 已有计算进程：" + ", ".join(rows))


def check_cuda(config):
    if config["train"]["device"] != "cuda":
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if config["train"]["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前设备不支持 BF16")


@contextmanager
def training_lock(root):
    import fcntl
    path = under(Path(root) / "runs/.gpu_training.lock", root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("另一轮训练持有项目 GPU 锁") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def required_free_bytes(config):
    # 完整last + last临时文件 + best模型及其临时文件；已有本轮文件不重复预留。
    used, seen = 0, set()
    for name in ("last.pt", "best.pt"):
        path = Path(config["output_dir"]) / name
        if path.is_file() and not path.is_symlink():
            stat = path.stat(); inode = (stat.st_dev, stat.st_ino)
            if inode not in seen:
                seen.add(inode); used += stat.st_size
    tensors = 0 if config["fixture"] else PARAMETER_COUNT * 32
    return int(config["min_free_bytes"]) + max(0, tensors - used)


def preflight(config, action):
    busy = training_processes()
    if busy:
        raise RuntimeError("已有训练进程，拒绝并行：" + json.dumps(busy, ensure_ascii=False))
    if config["train"]["device"] == "cuda":
        gpu_idle(); check_cuda(config)
    output = Path(config["output_dir"])
    if action in {"start", "probe"} and output.exists() and any(output.iterdir()):
        raise FileExistsError("输出非空，使用 resume 或新的时间戳")
    if action == "resume" and (not (output / "last.pt").is_file() or (output / "last.pt").is_symlink()):
        raise ValueError("resume 仅允许本目录的完整 last.pt，不能恢复 best.pt")
    needed = required_free_bytes(config)
    free = shutil.disk_usage(config["project_root"]).free
    if free < needed:
        raise RuntimeError(f"磁盘不足：需要 {needed/GIB:.2f}GiB，实际 {free/GIB:.2f}GiB")
    ledger_path = Path(config["budget"]["ledger_path"])
    ledger = read_json(ledger_path) if ledger_path.exists() else {"runs": {}}
    recorded = sum(float(row["estimated_compute_cny"]) for row in ledger["runs"].values())
    run_cost = float(ledger["runs"].get(str(output.resolve()), {}).get("estimated_compute_cny", 0))
    available = config["budget"]["project_limit_cny"] - config["budget"]["reserve_cny"] - recorded
    if available <= 0 or run_cost >= config["budget"]["run_limit_cny"]:
        raise RuntimeError("记录的运行/项目计算预算已耗尽")
    return {"gpu_training_started": False, "free_gib": free/GIB, "required_free_gib": needed/GIB,
            "recorded_project_compute_cny": recorded, "available_recorded_compute_cny": available,
            "plan": plan_details(config), "budget": config["budget"],
            "storage_note": "完整last用于断点恢复，best仅模型；多轮对照前逐次检查磁盘，不自动删除旧输出"}


def run_engine(config, **options):
    from llm_sft.engine import run_sft
    return run_sft(config, **options)


def execute(action, config_path, *, probe_steps=16):
    require_linux()
    config = frozen_config(config_path)
    plan = plan_details(config)
    if action == "probe" and (type(probe_steps) is not int or not 1 <= probe_steps < plan["stop_step"]):
        raise ValueError("probe steps 必须为正数且小于本实验停止步数，保持完整学习率计划")
    with training_lock(config["project_root"]):
        checks = preflight(config, action)
        if action == "check":
            return checks
        state_path = paths(config_path)["state"]
        state = {"status": "running", "action": action, "pid": os.getpid(), "name": Path(config['output_dir']).name,
                 "config": str(Path(config_path).resolve()), "started_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(state_path, state)
        try:
            options = {"stop_after_steps": probe_steps} if action == "probe" else {}
            if action == "resume":
                options["resume"] = Path(config["output_dir"]) / "last.pt"
            result = run_engine(config, **options)
            write_json(state_path, {**state, "status": "exited", "summary": result})
            return result
        except BaseException as error:
            write_json(state_path, {**state, "status": "failed", "error_type": type(error).__name__})
            raise


def status(config_path):
    related = paths(config_path); config = read_config(related["config"])
    output = Path(config["output_dir"])
    identity = read_json(related['identity']) if related['identity'].is_file() else {}
    result = {"name": identity.get('name', output.name), "legacy_naming": not bool(identity.get('name')),
              "output_dir": str(output), "last_checkpoint_exists": (output / "last.pt").is_file()}
    for name, path in (("launcher", related["state"]), ("summary", output / "summary.json"), ("wandb", output / "wandb_run.json")):
        if path.is_file():
            result[name] = read_json(path)
    if result.get("launcher", {}).get("status") == "running" and sys.platform == "linux":
        try:
            argv = Path(f"/proc/{result['launcher']['pid']}/cmdline").read_bytes().decode(errors="replace").split("\0")
            result["pid_alive"] = any(Path(arg).name == "train_sft.py" for arg in argv) and bool(ACTIONS.intersection(argv))
        except OSError:
            result["pid_alive"] = False
    return result


def plan(baseline, axis, stamp=None):
    if axis not in AXES:
        raise ValueError("只支持明确的单轴计划")
    config = read_config(baseline); root = Path(config["project_root"])
    stamp = stamp or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", stamp):
        raise ValueError("时间戳必须为 YYYYmmdd_HHMMSS")
    directory = under(root / "runs/plans" / f"sft_controls_{axis}_{stamp}", root)
    directory.mkdir(parents=True, exist_ok=False)
    candidates = []
    for index, value in enumerate(AXES[axis]):
        candidate = copy.deepcopy(config)
        candidate["runtime_sha256"] = None
        candidate["train"]["wandb_run_name"] = None
        candidate["train"]["stop_epochs"] = None
        if axis == "lr": candidate["train"]["learning_rate"] = value
        elif axis == "batch":
            if value % candidate["train"]["micro_batch_size"]:
                raise ValueError("Batch 轴要求候选可被固定 micro_batch_size 整除")
            baseline_batch = config['train']['micro_batch_size'] * config['train']['gradient_accumulation_steps']
            candidate["train"]["gradient_accumulation_steps"] = value // candidate["train"]["micro_batch_size"]
            # 按处理样本位置匹配验证与保存频率，避免小batch得到更多选模机会。
            for interval in ('eval_interval', 'save_interval'):
                candidate['train'][interval] = max(1, round(config['train'][interval] * baseline_batch / value))
        elif axis == "epochs":
            candidate["train"]["epochs"] = 2.; candidate["train"]["stop_epochs"] = value
        elif axis == "warmup": candidate["train"]["warmup_ratio"] = value
        else: candidate["train"]["weight_decay"] = value
        candidate["experiment"] = {"control_axis": axis, "control_value": value,
                                   "baseline_config_sha256": sha256_file(Path(baseline)),
                                   "note": "单轴候选；epochs轴共享2轮cosine调度，只改变停止训练量。"}
        proposed = (datetime.strptime(stamp, "%Y%m%d_%H%M%S") + timedelta(seconds=index)).strftime("%Y%m%d_%H%M%S")
        name = experiment_name(candidate, proposed)
        candidate['output_dir'] = str(under(root / 'runs' / name, root))
        path = directory / f"candidate_{index+1}.json"
        write_json(path, candidate)
        candidates.append({"axis": axis, "value": value, "template": str(path), "proposed_stamp": proposed,
                           "RUN_NAME": name, "proposed_output_dir": candidate['output_dir'],
                           "plan": plan_details(candidate),
                           "prepare_command": f".venv/bin/python scripts/train_sft.py prepare --template {shlex.quote(str(path))} --stamp {proposed}"})
    report = {"schema_version": 1, "axis": axis, "baseline": str(Path(baseline).resolve()),
              "gpu_training_started": False, "frozen": False, "candidates": candidates,
              "comparison": "同Base/数据/种子/模板；先比较学习率，选择基线后再逐轴生成。",
              "deferred": ["LoRA更新容量与学习率", "DPO", "匹配语言/长度和封存评测集的数据质量对照"]}
    write_json(directory / "plan.json", report)
    return {"plan": str(directory / "plan.json"), **report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--template", default="configs/posttrain/custom_300m_sft.json")
    prepare_command.add_argument("--stamp")
    for name in ("check", "probe", "start", "resume", "status"):
        command = commands.add_parser(name); command.add_argument("--config", required=True)
        if name == "probe": command.add_argument("--steps", type=int, default=16)
    matrix = commands.add_parser("plan")
    matrix.add_argument("--baseline", default="configs/posttrain/custom_300m_sft.json")
    matrix.add_argument("--axis", choices=sorted(AXES), default="lr")
    matrix.add_argument("--stamp")
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare": result = prepare(args.template, args.stamp)
        elif args.action == "status": result = status(args.config)
        elif args.action == "plan": result = plan(args.baseline, args.axis, args.stamp)
        else: result = execute(args.action, args.config, probe_steps=getattr(args, "steps", 16))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"错误：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
