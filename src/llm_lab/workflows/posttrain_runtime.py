"""Hugging Face 后训练执行层；所有训练框架均按需加载。"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from .posttrain_contracts import (
    PINNED_PACKAGES, ValidatedRun, canonical_hash,
    encode_example, example_token_counts, make_manifest, make_training_plan,
    read_artifact_manifest, read_json, write_json,
)

def check_dependency_versions() -> dict:
    versions = {}
    for name, expected in PINNED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"缺少 {name}=={expected}；先在远程环境安装项目 posttrain extra") from exc
        if actual != expected:
            raise RuntimeError(f"训练接口锁定 {name}=={expected}，当前为 {actual}；请使用锁定的远程环境")
        versions[name] = actual
    versions["torch"] = importlib.metadata.version("torch")
    return versions


def configure_environment(cfg: dict) -> None:
    cache = Path(cfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USE_TF"] = "0"
    if cfg["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def load_saved_model(path: Path, trainable: bool = False, *, _depth: int = 0) -> tuple[Any, Any]:
    """根据 manifest 分辨完整模型和 adapter；adapter 必须先重建其真实基础权重。"""
    if _depth > 4:
        raise ValueError("模型来源链过深或包含环")
    manifest = read_artifact_manifest(path)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = manifest["config"]
    configure_environment(cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if manifest["checkpoint_type"] == "full":
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32, trust_remote_code=False, attn_implementation="sdpa")
        for parameter in model.parameters():
            parameter.requires_grad_(trainable)
    else:
        from peft import PeftModel
        parent = manifest.get("weights_base_checkpoint")
        if parent:
            model, _ = load_saved_model(Path(parent), trainable=False, _depth=_depth + 1)
        else:
            model = AutoModelForCausalLM.from_pretrained(manifest["base_model"], revision=manifest["base_revision"],
                cache_dir=cfg["cache_dir"], torch_dtype=torch.float32, trust_remote_code=False, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(model, str(path), is_trainable=trainable, adapter_name="default")
    return model, tokenizer


def restore_dpo_adapters_inplace(model: Any, checkpoint: Path | str,
                                 sft_reference: Path | str | None = None) -> None:
    """明确恢复根目录 policy，并从原 SFT 恢复冻结 reference，保持 Parameter 对象不变。

    Transformers 4.57.1 遇到 ``reference/`` 子目录时只加载子目录，可能漏掉根目录
    的 default 权重。不能仅依赖 Trainer 默认恢复，也不能重建 adapter 后继续用旧 optimizer。
    """
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    adapters = getattr(model, "peft_config", {})
    if "default" not in adapters or (sft_reference is not None and "reference" not in adapters):
        raise ValueError("恢复 DPO 必须先构建已有 default/reference adapter，禁止在恢复时新建")
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}

    def copy_existing(path: Path | str, adapter: str) -> None:
        path = Path(path)
        if not path.is_dir() or not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError(f"缺少本地 {adapter} adapter 权重，禁止尝试远程查找: {path}")
        state = load_peft_weights(str(path), device="cpu", local_files_only=True)
        # low_cpu_mem_usage=False 使用 load_state_dict 的 copy 路径，不用 assign 替换参数。
        result = set_peft_model_state_dict(model, state, adapter_name=adapter, low_cpu_mem_usage=False)
        if getattr(result, "unexpected_keys", []) or any(
            f".{adapter}." in key for key in getattr(result, "missing_keys", [])
        ):
            raise ValueError(f"{adapter} adapter 的 checkpoint 权重不完整或不匹配")

    copy_existing(checkpoint, "default")
    if sft_reference is not None:
        # reference 的唯一来源是显式 SFT 起点，不信任训练 checkpoint 内的 reference 副本。
        copy_existing(sft_reference, "reference")
    model.set_adapter("default")
    if {name: id(parameter) for name, parameter in model.named_parameters()} != parameter_ids:
        raise RuntimeError("adapter 恢复替换了 Parameter，停止以免 optimizer 引用失效")
    if sft_reference is not None:
        reference_parameters = [p for name, p in model.named_parameters() if ".reference." in name]
        if not reference_parameters or any(p.requires_grad for p in reference_parameters):
            raise RuntimeError("SFT reference adapter 未保持冻结")


def run_training(run: ValidatedRun) -> dict:
    cfg = run.config
    if cfg["budget"]["hourly_rate_cny"] is None:
        raise ValueError("训练前必须填写实际 GPU hourly_rate_cny；未知价格不能启动付费计算")
    # BudgetGuard 使用项目总账；guard 覆盖模型加载、训练和最终保存的运行时间。
    from llm_lab.common.budget import BudgetGuard
    guard = BudgetGuard(cfg["budget"], run_id=canonical_hash(cfg["output_dir"])[:16], ledger_path=cfg["budget_ledger"])
    started = time.monotonic()
    tracker = None
    try:
        versions = check_dependency_versions()
        configure_environment(cfg)
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
        from torch.utils.data import SequentialSampler
        from peft import LoraConfig, PeftModel, get_peft_model
        from llm_lab.common.repro import environment_snapshot
        if not torch.cuda.is_available():
            raise RuntimeError("训练必须在远程 CUDA GPU 上运行；本机仅执行 --validate-only 和离线测试")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("首版只支持单进程单卡，请不要用多卡 launcher")
        if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 BF16，请在新实验配置中改用 FP32/FP16")
        set_seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if cfg["deterministic"]:
            torch.use_deterministic_algorithms(True)
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        from llm_lab.common.tracking import start_tracker
        tracker = start_tracker(cfg, output_dir, Path(cfg["project_root"]),
            job_type=f"{cfg['stage']}_{cfg['method']}", resume=run.resume_path is not None,
            run_config={key: cfg[key] for key in ("stage", "method", "base_model", "base_revision", "seed", "precision",
                "max_length", "batch_size", "gradient_accumulation_steps", "learning_rate", "train_token_budget", "lora")})
        manifest = make_manifest(run)
        manifest["versions"] = versions
        manifest["environment"] = environment_snapshot()
        model_kwargs = {"torch_dtype": torch.float32, "trust_remote_code": False, "attn_implementation": "sdpa"}
        reference = None
        dpo_extra = {}
        lora_config = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                                 target_modules=cfg["lora"]["target_modules"], bias="none", task_type="CAUSAL_LM")
        if cfg["stage"] == "sft":
            tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["base_revision"], cache_dir=cfg["cache_dir"], trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg["base_revision"], cache_dir=cfg["cache_dir"], **model_kwargs)
            if cfg["method"] == "lora":
                model = get_peft_model(model, lora_config)
        else:
            model, tokenizer = load_saved_model(Path(cfg["sft_checkpoint"]), trainable=True)
            if run.source_manifest["checkpoint_type"] == "adapter" and cfg["method"] == "lora":
                # 同一份 Base，default 为可训练 SFT adapter，reference 为冻结的同一 SFT adapter。
                # 不能仅 disable_adapter()：那会错误地把 reference 变回未 SFT 的原始 Base。
                model.load_adapter(cfg["sft_checkpoint"], adapter_name="reference", is_trainable=False)
                model.set_adapter("default")
                dpo_extra.update(model_adapter_name="default", ref_adapter_name="reference")
            else:
                if isinstance(model, PeftModel):
                    model = model.merge_and_unload()  # adapter SFT -> 全参 DPO 时先还原完整 SFT 权重。
                    for parameter in model.parameters():
                        parameter.requires_grad_(True)
                reference, _ = load_saved_model(Path(cfg["sft_checkpoint"]), trainable=False)
                reference.eval()
                if cfg["method"] == "lora":
                    model = get_peft_model(model, lora_config)
                    dpo_extra["force_use_ref_model"] = True
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        if tokenizer.eos_token_id is None:
            raise ValueError("Base tokenizer 没有 EOS，不能构造一致的监督序列")
        model.config.use_cache = False
        if isinstance(model, PeftModel) and cfg["gradient_checkpointing"]:
            model.enable_input_require_grads()
        manifest["resolved_base_revision"] = getattr(model.config, "_commit_hash", None) or cfg["base_revision"]
        manifest["template"]["eos_token_id"] = tokenizer.eos_token_id
        manifest["model_parameters"] = sum(p.numel() for p in model.parameters())
        manifest["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_encoded = [encode_example(r, tokenizer, cfg["max_length"], cfg["stage"]) for r in run.train_rows]
        val_encoded = [encode_example(r, tokenizer, cfg["max_length"], cfg["stage"]) for r in run.validation_rows]
        effective_batch = cfg["batch_size"] * cfg["gradient_accumulation_steps"]
        order, plan = make_training_plan(train_encoded, token_budget=cfg["train_token_budget"], effective_batch_size=effective_batch, seed=cfg["seed"], stage=cfg["stage"],
                                         languages=[r["language"] for r in run.train_rows])
        train_dataset = Dataset.from_list([train_encoded[i] for i in order])
        val_dataset = Dataset.from_list(val_encoded)
        manifest["training_plan"] = plan
        manifest["effective_batch_size"] = effective_batch
        manifest["precision_note"] = "参数与 AdamW 状态 FP32，forward/backward 按 precision autocast；full/LoRA 共用相同存储精度。"
        manifest["complete_checkpoint"] = False
        if run.resume_path:
            old = read_json(run.resume_path / "run_manifest.json")
            if old.get("training_plan") != plan or old.get("versions") != versions:
                raise ValueError("恢复要求相同训练 plan 与依赖版本")
            environment_fields = ("python", "platform", "packages", "torch_cuda", "cuda_available", "gpu")
            if any(old.get("environment", {}).get(key) != manifest["environment"].get(key) for key in environment_fields):
                raise ValueError("恢复要求相同 Python/CUDA/GPU/包版本环境")
        write_json(output_dir / "run_manifest.json", manifest)
        write_json(output_dir / "resolved_config.json", cfg)
        prefix_tokens, prefix_loss = [0], [0]
        for index in order:
            count, loss_count = example_token_counts(train_encoded[index], cfg["stage"])
            prefix_tokens.append(prefix_tokens[-1] + count)
            prefix_loss.append(prefix_loss[-1] + loss_count)
        if run.resume_path:
            progress = read_json(run.resume_path / "progress.json")
            consumed = progress["examples_consumed"]
            if consumed > len(order) or progress["input_tokens_seen"] != prefix_tokens[consumed] or progress["loss_tokens_seen"] != prefix_loss[consumed]:
                raise ValueError("恢复数据位置或已消费 Token 数与训练 plan 不一致")

        class BudgetStop(RuntimeError):
            pass

        class Telemetry(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                self.step_at_start = state.global_step
                self.clock = time.monotonic()
                torch.cuda.reset_peak_memory_stats()

            def on_step_end(self, args, state, control, **kwargs):
                guard.update()
                if guard.should_stop():
                    control.should_training_stop = True
                    control.should_save = True
                    control.should_evaluate = False
                return control

            def on_prediction_step(self, args, state, control, **kwargs):
                if guard.should_stop():
                    raise BudgetStop("验证期间到达估计费用/时长上限")

            def on_log(self, args, state, control, logs=None, **kwargs):
                consumed = min(state.global_step * effective_batch, len(order))
                previous = min(self.step_at_start * effective_batch, len(order))
                torch.cuda.synchronize()
                duration = max(time.monotonic() - self.clock, 1e-9)
                event = {"global_step": state.global_step, **(logs or {}), "input_tokens_seen": prefix_tokens[consumed],
                         "loss_tokens_seen": prefix_loss[consumed], "input_tokens_per_second": (prefix_tokens[consumed] - prefix_tokens[previous]) / duration,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "session_estimated_compute_cost_cny": (time.monotonic() - started) * cfg["budget"]["hourly_rate_cny"] / 3600}
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                tracker.log(event, state.global_step)

            def on_save(self, args, state, control, **kwargs):
                checkpoint = output_dir / f"checkpoint-{state.global_step}"
                consumed = min(state.global_step * effective_batch, len(order))
                write_json(checkpoint / "progress.json", {"global_step": state.global_step, "examples_consumed": consumed,
                    "input_tokens_seen": prefix_tokens[consumed], "loss_tokens_seen": prefix_loss[consumed], "plan_hash": plan["plan_hash"]})
                tokenizer.save_pretrained(str(checkpoint))
                write_json(checkpoint / "run_manifest.json", {**manifest, "complete_checkpoint": True})

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                with (output_dir / "validation_history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"global_step": state.global_step, "checkpoint": f"checkpoint-{state.global_step}",
                        "metrics": metrics or {}, "split": "validation"}, ensure_ascii=False) + "\n")

        class OrderedTrainer(Trainer):
            def _get_train_sampler(self, train_dataset=None):
                # plan 自己负责随机化；固定序列使 skip-data resume 与等 Token 对照可审计。
                return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        class CompletionTrainer(OrderedTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :].float()  # [B,T-1,V]
                targets = labels[:, 1:].contiguous()       # [B,T-1]
                denominator = num_items_in_batch if num_items_in_batch is not None else (targets != -100).sum()
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction="sum") / denominator
                return (loss, outputs) if return_outputs else loss

        def collate_sft(examples):
            width = max(len(row["input_ids"]) for row in examples)
            result = {"input_ids": [], "attention_mask": [], "labels": []}
            for row in examples:
                padding = width - len(row["input_ids"])
                result["input_ids"].append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
                result["attention_mask"].append(row["attention_mask"] + [0] * padding)
                result["labels"].append(row["labels"] + [-100] * padding)
            return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}

        arguments = dict(output_dir=str(output_dir), max_steps=plan["optimizer_steps"], num_train_epochs=1,
            per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"],
            lr_scheduler_type="cosine", warmup_ratio=cfg["warmup_ratio"], weight_decay=cfg["weight_decay"],
            max_grad_norm=cfg["max_grad_norm"], optim="adamw_torch", bf16=cfg["precision"] == "bf16", fp16=cfg["precision"] == "fp16",
            gradient_checkpointing=cfg["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps", eval_steps=cfg["eval_steps"], save_strategy="steps", save_steps=cfg["save_steps"],
            logging_steps=cfg["logging_steps"], save_total_limit=cfg["save_total_limit"],
            seed=cfg["seed"], data_seed=cfg["seed"], dataloader_num_workers=0, dataloader_drop_last=False,
            report_to=["tensorboard"] if cfg["tensorboard"] else [], logging_dir=str(output_dir / "tensorboard"),
            remove_unused_columns=False, save_only_model=False, ignore_data_skip=False, push_to_hub=False,
            full_determinism=cfg["deterministic"], tf32=False, load_best_model_at_end=False)
        if cfg["stage"] == "sft":
            trainer = CompletionTrainer(model=model, args=TrainingArguments(**arguments), train_dataset=train_dataset,
                eval_dataset=val_dataset, data_collator=collate_sft, processing_class=tokenizer, callbacks=[Telemetry()])
            # Trainer 为每个累积窗口传入所有有效 label 的总数；避免短/长 microbatch 等权误差。
            trainer.model_accepts_loss_kwargs = True
        else:
            from trl import DPOConfig, DPOTrainer

            class OrderedDPOTrainer(DPOTrainer):
                def _get_train_sampler(self, train_dataset=None):
                    return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

                def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
                    target = self.model if model is None else model
                    before = {name: id(parameter) for name, parameter in target.named_parameters()}
                    super()._load_from_checkpoint(resume_from_checkpoint, model=model)
                    if isinstance(target, PeftModel):
                        source_adapter = cfg["sft_checkpoint"] if run.source_manifest["checkpoint_type"] == "adapter" else None
                        restore_dpo_adapters_inplace(target, resume_from_checkpoint, source_adapter)
                    if {name: id(parameter) for name, parameter in target.named_parameters()} != before:
                        raise RuntimeError("Trainer 恢复替换了 Parameter，不能安全继续 optimizer 状态")

                def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
                    # TRL 0.24 默认总会 tokenize raw chosen/rejected；这里复用已精确编码的边界。
                    expected = {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
                    if set(dataset.column_names) != expected:
                        raise ValueError("DPO 必须使用本工作流已经预检过的 Token 数据")
                    return dataset

            dpo_arguments = DPOConfig(**arguments, **dpo_extra, beta=cfg["beta"], loss_type="sigmoid", max_length=cfg["max_length"],
                max_prompt_length=cfg["max_length"], max_completion_length=cfg["max_length"],
                precompute_ref_log_probs=False, reference_free=False, disable_dropout=True, generate_during_eval=False,
                dataset_num_proc=1)
            trainer = OrderedDPOTrainer(model=model, ref_model=reference, args=dpo_arguments,
                train_dataset=train_dataset, eval_dataset=val_dataset, processing_class=tokenizer, callbacks=[Telemetry()])
        if guard.should_stop():
            raise ValueError("模型加载/数据预检已到达预算上限，尚未开始训练")
        try:
            result_metrics = trainer.train(resume_from_checkpoint=str(run.resume_path) if run.resume_path else None).metrics
        except BudgetStop:
            result_metrics = {"budget_stop_during_validation": True}
        final_validation = {"skipped_due_to_budget": True}
        if not guard.should_stop():
            try:
                final_validation = trainer.evaluate()  # 不足 eval_steps 也记录最后点，预算停止则明确跳过。
            except BudgetStop:
                final_validation = {"incomplete_due_to_budget": True}
        # 最后一步即使没有整除 save_steps，也必须留下可恢复 checkpoint。
        if not (output_dir / f"checkpoint-{trainer.state.global_step}" / "run_manifest.json").exists():
            trainer._save_checkpoint(trainer.model, trial=None)
            trainer.control = trainer.callback_handler.on_save(trainer.args, trainer.state, trainer.control)
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        if isinstance(model, PeftModel):
            model.save_pretrained(str(final_dir), selected_adapters=["default"], safe_serialization=True)
        else:
            model.save_pretrained(str(final_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(final_dir))
        final_manifest = {**manifest, "complete_checkpoint": False, "global_step": trainer.state.global_step,
                          "budget_stopped_early": trainer.state.global_step < plan["optimizer_steps"]}
        write_json(final_dir / "run_manifest.json", final_manifest)
        trainer.save_state()
        write_json(output_dir / "train_result.json", {**result_metrics, "global_step": trainer.state.global_step,
            "training_plan": plan, "remote_gpu_verified": True, "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda, "final_validation": final_validation,
            "budget_stopped_early": final_manifest["budget_stopped_early"]})
        return final_manifest
    finally:
        if tracker is not None:
            tracker.finish(exit_code=int(sys.exc_info()[0] is not None))
        guard.update()


