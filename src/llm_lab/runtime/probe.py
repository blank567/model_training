"""记录真实运行环境，并只在 CUDA 可用时执行小型 GPU 矩阵乘法。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any


PACKAGES = ("torch", "transformers", "datasets", "tokenizers", "peft", "trl", "accelerate", "wandb")


def _driver_info() -> dict[str, Any]:
    """nvidia-smi 读取的是驱动信息，其 CUDA 标记不是 PyTorch wheel 版本。"""
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "detail": type(exc).__name__}
    return {
        "available": process.returncode == 0,
        "detail": process.stdout.strip() if process.returncode == 0 else process.stderr.strip(),
    }


def collect_report(device: str = "auto", matmul_size: int = 1024, iterations: int = 10) -> dict[str, Any]:
    """device=cpu 仅检查 CPU 环境；不会借机进行本机 GPU 测试或训练。"""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 必须为 auto、cpu 或 cuda")
    if matmul_size <= 0 or iterations <= 0:
        raise ValueError("matmul_size 与 iterations 必须为正整数")
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    report: dict[str, Any] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "requested_device": device,
        "cuda_available": False,
        "torch_cuda_runtime": None,
        "gpus": [],
        "gpu_test": {"status": "skipped", "reason": "尚未检查 CUDA"},
        "driver": _driver_info(),
    }
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        report["gpu_test"] = {"status": "skipped", "reason": f"PyTorch 未安装或无法导入: {exc}"}
        return report
    report["torch_cuda_runtime"] = torch.version.cuda
    report["cuda_available"] = bool(torch.cuda.is_available())
    if device == "cpu":
        report["gpu_test"] = {"status": "skipped", "reason": "显式 CPU 检查；未执行 GPU 运算"}
        return report
    if not report["cuda_available"]:
        report["gpu_test"] = {"status": "skipped", "reason": "torch.cuda.is_available() 为 False"}
        return report
    try:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                bf16 = bool(torch.cuda.is_bf16_supported())
            report["gpus"].append({
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "bf16_supported": bf16,
            })
        # 只对当前默认 CUDA 设备测一次真实工作；这不是训练吞吐基准。
        index = torch.cuda.current_device()
        dtype = torch.bfloat16 if report["gpus"][index]["bf16_supported"] else torch.float32
        selected = torch.device("cuda", index)
        with torch.inference_mode():
            left = torch.randn(matmul_size, matmul_size, device=selected, dtype=dtype)
            right = torch.randn(matmul_size, matmul_size, device=selected, dtype=dtype)
            for _ in range(3):
                result = left @ right
            torch.cuda.synchronize(selected)
            started = time.perf_counter()
            for _ in range(iterations):
                result = left @ right
            torch.cuda.synchronize(selected)
            elapsed = time.perf_counter() - started
            finite = bool(torch.isfinite(result).all().item())
        report["gpu_test"] = {
            "status": "passed" if finite else "failed",
            "device_index": index,
            "dtype": str(dtype),
            "shape": [matmul_size, matmul_size],
            "iterations": iterations,
            "elapsed_seconds": elapsed,
            "milliseconds_per_matmul": elapsed * 1000 / iterations,
            "finite": finite,
            "note": "已同步 CUDA；小型矩阵测试不能替代完整训练吞吐/显存实测",
        }
    except (RuntimeError, AssertionError) as exc:
        report["gpu_test"] = {"status": "failed", "reason": str(exc)}
    return report


def write_report(report: dict[str, Any], output: str | Path) -> Path:
    """只写调用方指定的报告，并在同目录原子替换，避免留下半份 JSON。"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matmul-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    report = collect_report(args.device, args.matmul_size, args.iterations)
    if args.output:
        print(write_report(report, args.output))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    must_pass = args.require_cuda or args.device == "cuda"
    return int(must_pass and report["gpu_test"]["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
