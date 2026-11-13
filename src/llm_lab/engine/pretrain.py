"""从零预训练：更新边界、Token 预算与恢复状态全部显式化。"""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import canonical_hash, sha256_file, write_json
from llm_lab.common.repro import environment_snapshot, restore_rng, rng_state, seed_everything
from llm_lab.common.tracking import start_tracker
from llm_lab.config import ExperimentConfig
from llm_lab.models import Decoder
from .checkpoint import load_checkpoint, save_checkpoint
from .corpus import BinaryCorpus


def autocast(config: ExperimentConfig):
    if config.train.dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if config.train.dtype == "bf16" else torch.float16
    return torch.autocast(config.train.device, dtype=dtype)


def lr_factor(step: int, warmup_steps: int, max_steps: int, minimum: float) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, max_steps - warmup_steps)))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    return canonical_hash({str(file.relative_to(root)): sha256_file(file) for file in sorted(root.rglob("*.py"))})


def optimizer_update(model, optimizer, scaler, max_grad_norm: float) -> tuple[bool, float]:
    """返回是否真正更新；FP16 溢出由 GradScaler 回退，其他精度立即报错。"""
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not torch.isfinite(grad_norm):
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite gradients; update aborted")
        scaler.step(optimizer)  # unscale 已登记 found_inf，optimizer 不会执行。
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return False, float(grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return True, float(grad_norm)


@torch.no_grad()
def evaluate_model(model: Decoder, corpus: BinaryCorpus, config: ExperimentConfig) -> dict:
    was_training = model.training
    model.eval()
    result = {}
    try:
        for index, language in enumerate(("en", "zh")):
            # 局部采样器固定验证窗口，不改变训练采样和全局随机数状态。
            sampler = np.random.default_rng(config.train.seed + 10000 + index)
            losses = []
            for _ in range(config.train.eval_batches):
                x, y = corpus.batch(language, config.train.micro_batch_size, sampler)
                with autocast(config):
                    loss = model(x.to(config.train.device), y.to(config.train.device)).loss
                losses.append(float(loss))
            average = sum(losses) / len(losses)
            if not math.isfinite(average):
                raise FloatingPointError("Non-finite evaluation loss")
            result[language] = {"loss": average, "perplexity": math.exp(min(average, 80.0)),
                                "perplexity_capped": average > 80.0,
                                "evaluated_tokens": config.train.eval_batches * config.train.micro_batch_size * config.train.seq_len}
        result["macro_loss"] = (result["en"]["loss"] + result["zh"]["loss"]) / 2
        return result
    finally:
        model.train(was_training)


def _runtime_identity(config: ExperimentConfig, train: BinaryCorpus, validation: BinaryCorpus, environment: dict) -> dict:
    payload = config.to_dict()
    # 资源限额可在恢复时缩紧，但不能修改模型、调度终点、数据或种子。
    payload.pop("budget")
    payload.pop("output_dir")
    payload.pop("project_root")
    payload["data_manifest"] = sha256_file(config.data_manifest)
    return {"config": canonical_hash(payload), "manifest_sha256": sha256_file(config.data_manifest),
            "tokenizer_sha256": train.manifest["tokenizer"]["sha256"], "train_data": train.fingerprint,
            "validation_data": validation.fingerprint, "source": source_fingerprint(),
            "environment": {key: environment[key] for key in ("python", "packages", "torch_cuda", "gpu")}}


def run_pretrain(config: ExperimentConfig, resume: str | Path | None = None,
                 stop_after_steps: int | None = None) -> dict:
    """stop_after_steps 只用于模拟中断测试，不改变学习率总计划。"""
    config.validate()
    device = config.train.device
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable: run remote probe before training")
        if config.train.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported on this GPU/runtime")
        if not config.budget.get("hourly_rate_cny") or config.budget["hourly_rate_cny"] <= 0:
            raise ValueError("Set the actual remote GPU hourly_rate_cny before paid training")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError("Output directory is not empty; choose a new run or explicitly resume")
    if resume and not Path(resume).is_file():
        raise FileNotFoundError(resume)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.train.seed, config.train.deterministic)
    environment = environment_snapshot()
    training = BinaryCorpus(config.data_manifest, "train", config.train.seq_len)
    validation = BinaryCorpus(config.data_manifest, "validation", config.train.seq_len)
    vocab_size = int(training.manifest["tokenizer"]["vocab_size"])
    if vocab_size != config.model.vocab_size:
        raise ValueError(f"Tokenizer vocab_size={vocab_size}, model vocab_size={config.model.vocab_size}; update explicit model config")
    identity = _runtime_identity(config, training, validation, environment)
    model = Decoder(config.model).to(device)
    # 矩阵衰减；norm 等一维参数不做 weight decay。
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": config.train.weight_decay},
              {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=config.train.learning_rate, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, config.train.warmup_steps,
                                                                                 config.train.max_steps, config.train.min_lr_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and config.train.dtype == "fp16")
    sampler = np.random.default_rng(config.train.seed)
    step, tokens_seen, best_loss = 0, 0, None
    previous = load_checkpoint(resume) if resume else None
    run_id = previous["run_id"] if previous else str(output.resolve())
    if previous:
        if previous["identity"] != identity:
            raise ValueError("Resume identity mismatch: config/data/code/environment changed")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        step, tokens_seen, best_loss = previous["step"], previous["tokens_seen"], previous["best_val_loss"]
        sampler.bit_generator.state = previous["sampler_rng"]
        restore_rng(previous["rng"])
    write_json(output / "config.json", config.to_dict())
    write_json(output / "environment.json", environment)
    write_json(output / "run_manifest.json", {"schema_version": 1, "stage": "pretrain", "run_id": run_id,
                                             "identity": identity, "parameters": model.parameter_count(),
                                             "sampling": "random windows with replacement; en/zh alternate per microbatch",
                                             "unique_tokens": {lang: len(stream) for lang, stream in training.arrays.items()}})
    guard = BudgetGuard(config.budget, run_id, config.budget.get("ledger_path", "runs/budget_ledger.json"))
    writer = None
    if config.train.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError("Install the tracking extra for TensorBoard or explicitly set tensorboard=false") from error
        writer = SummaryWriter(str(output / "tensorboard"), purge_step=step + 1 if resume else None)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started, initial_tokens = time.monotonic(), tokens_seen
    tokens_per_update = config.train.micro_batch_size * config.train.seq_len * config.train.gradient_accumulation_steps
    latest_metrics = None
    status = "completed"
    consecutive_overflows = 0
    stop_requested = False
    old_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Signal {signum}: stopping after the current optimizer update", flush=True)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)

    def persist() -> None:
        save_checkpoint(output / "last.pt", {"schema_version": 1, "config": config.to_dict(),
                        "identity": identity, "run_id": run_id, "environment": environment,
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "rng": rng_state(), "sampler_rng": sampler.bit_generator.state,
                        "step": step, "tokens_seen": tokens_seen, "best_val_loss": best_loss})

    tracker = None
    try:
        tracker = start_tracker(config.to_dict()["train"], output, Path(config.project_root), job_type="pretrain",
                                run_config={"model": config.to_dict()["model"], "train": config.to_dict()["train"]},
                                resume=previous is not None)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
            while step < config.train.max_steps:
                # 在 update 边界停止，避免保存未完成的梯度累积。
                if stop_requested:
                    status = "signal_stop"
                    break
                if guard.should_stop():
                    status = "budget_or_time_limit"
                    break
                if stop_after_steps is not None and step >= stop_after_steps:
                    status = "simulated_interruption"
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                learning_rate = optimizer.param_groups[0]["lr"]
                for micro in range(config.train.gradient_accumulation_steps):
                    language = ("en", "zh")[micro % 2]
                    x, y = training.batch(language, config.train.micro_batch_size, sampler)
                    with autocast(config):
                        loss = model(x.to(device), y.to(device)).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss at update {step + 1}")
                    loss_sum += float(loss.detach())
                    scaler.scale(loss / config.train.gradient_accumulation_steps).backward()
                updated, grad_norm = optimizer_update(model, optimizer, scaler, config.train.max_grad_norm)
                if not updated:
                    consecutive_overflows += 1
                    if consecutive_overflows > 32:
                        raise FloatingPointError("Repeated FP16 overflows after loss-scale backoff")
                    print(json.dumps({"event": "fp16_overflow_skipped", "step": step,
                                      "loss_scale": scaler.get_scale()}), flush=True)
                    continue
                consecutive_overflows = 0
                scheduler.step()
                step += 1
                tokens_seen += tokens_per_update
                if step % config.train.eval_interval == 0 or step == config.train.max_steps:
                    latest_metrics = evaluate_model(model, validation, config)
                    improved = best_loss is None or latest_metrics["macro_loss"] < best_loss
                    if improved:
                        best_loss = latest_metrics["macro_loss"]
                        # 保留完整恢复状态；best 只由 validation 选择。
                        persist()
                        import shutil
                        shutil.copyfile(output / "last.pt", output / "best.pt.tmp")
                        os.replace(output / "best.pt.tmp", output / "best.pt")
                else:
                    improved = False
                if step % config.train.log_interval == 0 or step == config.train.max_steps:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.monotonic() - started
                    record = {"step": step, "tokens_seen": tokens_seen, "train_loss": loss_sum / config.train.gradient_accumulation_steps,
                              "learning_rate": learning_rate, "grad_norm_before_clip": float(grad_norm),
                              "tokens_per_second_including_eval": (tokens_seen - initial_tokens) / max(elapsed, 1e-9),
                              "elapsed_seconds_this_session": elapsed,
                              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                              "budget": guard.update()}
                    if latest_metrics and (step % config.train.eval_interval == 0 or step == config.train.max_steps):
                        record["validation"] = latest_metrics
                    metrics_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    metrics_file.flush()
                    tracker.log(record, step)
                    if writer:
                        for key in ("train_loss", "learning_rate", "grad_norm_before_clip", "tokens_per_second_including_eval"):
                            writer.add_scalar(key, record[key], step)
                        if "validation" in record:
                            for language in ("en", "zh"):
                                writer.add_scalar(f"validation/{language}_loss", latest_metrics[language]["loss"], step)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                if step % config.train.save_interval == 0 or step == config.train.max_steps:
                    persist()
            persist()
    finally:
        if tracker is not None:
            tracker.finish(exit_code=int(sys.exc_info()[0] is not None))
        cost = guard.update()
        if writer:
            writer.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    summary = {"status": status, "step": step, "tokens_seen": tokens_seen,
               "best_validation_loss": best_loss, "budget": cost,
               "checkpoint": str(output / "last.pt"), "parameters": model.parameter_count()}
    write_json(output / "summary.json", summary)
    return summary

