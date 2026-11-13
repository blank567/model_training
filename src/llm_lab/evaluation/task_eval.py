"""MASSIVE JSON 生成评测，不把任务准确率解释为通用语言能力。

gold/prediction 的结构是 ``{"intent": str, "slots": [{"slot": str, "value": str}]}``。
槽位按 (名称, 原文值) 多重集严格匹配，重复槽位不会被 set 静默丢弃。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import unicodedata
from typing import Any


from .task_schema import parse_task_json, unique_json_object


def _slots(obj: dict[str, Any]) -> Counter:
    # 只统一 Unicode 编码形式；不忽略大小写、顺序内的文字或错误值。
    return Counter((unicodedata.normalize("NFC", x["slot"]), unicodedata.normalize("NFC", x["value"])) for x in obj["slots"])


def score_one(reference: str, prediction: str) -> dict[str, Any]:
    gold = parse_task_json(reference)  # gold 错误应直接终止，不能算成模型失误。
    parsed = False
    try:
        json.loads(prediction, object_pairs_hook=unique_json_object)
        parsed = True
    except (ValueError, TypeError):
        pass
    try:
        pred = parse_task_json(prediction)
    except (ValueError, TypeError):
        pred = None
    gold_slots = _slots(gold)
    pred_slots = _slots(pred) if pred is not None else Counter()
    true_positive = sum((gold_slots & pred_slots).values())
    false_positive = sum(pred_slots.values()) - true_positive
    false_negative = sum(gold_slots.values()) - true_positive
    intent_correct = pred is not None and pred["intent"] == gold["intent"]
    return {
        "json_parse_valid": parsed,
        "json_valid": pred is not None,
        "intent_correct": intent_correct,
        "slot_tp": true_positive,
        "slot_fp": false_positive,
        "slot_fn": false_negative,
        "task_success": bool(intent_correct and gold_slots == pred_slots),
    }


def aggregate_scores(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """分别统计 zh/en；pooled 是逐例汇总，macro_languages 才是中英等权主结果。"""
    result = {}
    for name in ("pooled", "zh", "en"):
        subset = records if name == "pooled" else [r for r in records if r["language"] == name]
        count = len(subset)
        tp = sum(r["slot_tp"] for r in subset)
        fp = sum(r["slot_fp"] for r in subset)
        fn = sum(r["slot_fn"] for r in subset)
        denominator = 2 * tp + fp + fn
        result[name] = {
            "examples": count,
            "intent_accuracy": sum(r["intent_correct"] for r in subset) / count if count else None,
            "slot_micro_f1": 2 * tp / denominator if denominator else (1.0 if count else None),
            "slot_tp": tp, "slot_fp": fp, "slot_fn": fn,
            "json_parse_rate": sum(r["json_parse_valid"] for r in subset) / count if count else None,
            "json_valid_rate": sum(r["json_valid"] for r in subset) / count if count else None,
            "task_success_rate": sum(r["task_success"] for r in subset) / count if count else None,
        }
    fields = ("intent_accuracy", "slot_micro_f1", "json_parse_rate", "json_valid_rate", "task_success_rate")
    complete = result["zh"]["examples"] > 0 and result["en"]["examples"] > 0
    result["macro_languages"] = {field: (result["zh"][field] + result["en"][field]) / 2 if complete else None for field in fields}
    result["macro_languages"]["languages_present"] = [language for language in ("zh", "en") if result[language]["examples"]]
    return result


def score_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    """严格按 source_id/language 对齐；不接受预测文件改写 gold。"""
    indexed = {}
    for row in predictions:
        key = (row.get("source_id"), row.get("language"))
        if key in indexed or not isinstance(row.get("prediction"), str):
            raise ValueError("预测存在重复 ID 或非字符串 prediction")
        indexed[key] = row["prediction"]
    expected = {(r["source_id"], r["language"]) for r in rows}
    if set(indexed) != expected or len(expected) != len(rows):
        raise ValueError("预测与评测数据 ID 集合不一致，或评测数据含重复 ID")
    records = []
    for row in rows:
        prediction = indexed[(row["source_id"], row["language"])]
        records.append({**row, "prediction": prediction, **score_one(row["completion"], prediction)})
    return records, aggregate_scores(records)


def check_eval_isolation(rows: list[dict], manifest: dict, split: str) -> None:
    from llm_lab.workflows.posttrain_contracts import normalized_prompt_hash
    train_ids = set(manifest.get("training_source_ids", []))
    forbidden_ids = train_ids | (set(manifest.get("validation_source_ids", [])) if split == "test" else set())
    forbidden_prompts = set(manifest.get("training_prompt_hashes", []))
    if split == "test":
        forbidden_prompts.update(manifest.get("validation_prompt_hashes", []))
    for row in rows:
        if row["source_id"] in forbidden_ids or normalized_prompt_hash(row["prompt"]) in forbidden_prompts:
            raise ValueError(f"评测 {split} 与训练/选模数据重叠: {row['source_id']}")


def generate_predictions(checkpoint: Path, rows: list[dict], manifest: dict, output: Path,
                         max_new_tokens: int, hourly_rate_cny: float | None, run_limit_cny: float) -> list[dict]:
    """GPU 生成也记入共用总账；停止时保留逐例，不伪造完整测试集分数。"""
    from llm_lab.workflows.posttrain_contracts import format_prompt, write_json
    from llm_lab.workflows.posttrain_runtime import configure_environment, load_saved_model
    from llm_lab.common.budget import BudgetGuard
    cfg = manifest["config"]
    if hourly_rate_cny is None or not math.isfinite(hourly_rate_cny) or hourly_rate_cny <= 0:
        raise ValueError("GPU 生成评测必须显式传 --hourly-rate-cny 实际单价；离线 --predictions 不需要")
    if not math.isfinite(run_limit_cny) or run_limit_cny <= 0:
        raise ValueError("run-limit-cny 必须为有限正数")
    configure_environment(cfg)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("真实模型生成评测要求远程 CUDA GPU；本地可用 --predictions 离线评分")
    settings = {**cfg["budget"], "hourly_rate_cny": hourly_rate_cny, "run_limit_cny": run_limit_cny}
    guard = BudgetGuard(settings, run_id="task_eval_" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16],
                        ledger_path=str(Path(cfg["project_root"]) / "runs" / "budget_ledger.json"))
    predictions = []
    output.mkdir(parents=True, exist_ok=True)
    completed = False
    try:
        model, tokenizer = load_saved_model(checkpoint, trainable=False)
        model.to("cuda").eval()
        dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
        torch.manual_seed(cfg["seed"])
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=cfg["precision"] != "fp32"):
            for row in rows:
                if guard.should_stop():
                    raise RuntimeError("评测达到估计费用/时长上限，保留 partial_predictions；没有完整测试指标")
                inputs = tokenizer(format_prompt(row["prompt"]), add_special_tokens=False, return_tensors="pt").to("cuda")
                if inputs["input_ids"].shape[1] + max_new_tokens > model.config.max_position_embeddings:
                    raise ValueError("生成请求超出模型上下文，禁止静默截断测试问题")
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                                           eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                answer = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                prediction = {"source_id": row["source_id"], "language": row["language"], "prediction": answer}
                predictions.append(prediction)
                with (output / "partial_predictions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        completed = True
        return predictions
    finally:
        write_json(output / "runtime_cost.json", {**guard.update(), "completed": completed,
            "examples_generated": len(predictions), "examples_total": len(rows),
            "note": "运行时间与用户单价估算；不含未知存储/传输费，也不会停止云实例计费。"})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--predictions", type=Path, help="离线预测 JSONL，每行 source_id/language/prediction")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--final", action="store_true", help="确认测试集仅用于最终评测")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--hourly-rate-cny", type=float, help="GPU 生成评测的实际单价；必须显式填写")
    parser.add_argument("--run-limit-cny", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final:
        parser.error("独立 test 只用于 final，必须显式传入 --final；调参使用 validation")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens 必须大于 0")
    from llm_lab.workflows.posttrain_contracts import read_task_rows, write_json, read_jsonl, read_artifact_manifest
    args.output = args.output.resolve()
    if os.name == "nt" and args.output.drive.upper() != "E:":
        raise ValueError("本地评测输出必须放在 E: 盘")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("评测输出已存在，请使用新的 output 目录，保留历史结果")
    rows = read_task_rows(args.data.resolve(), "sft")
    manifest = None
    if args.predictions:
        predictions = read_jsonl(args.predictions)
    else:
        manifest = read_artifact_manifest(args.checkpoint)
        check_eval_isolation(rows, manifest, args.split)
        predictions = generate_predictions(args.checkpoint.resolve(), rows, manifest, args.output,
            args.max_new_tokens, args.hourly_rate_cny, args.run_limit_cny)
    records, metrics = score_predictions(rows, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "predictions.jsonl"
    if records_path.exists() or (args.output / "metrics.json").exists():
        raise FileExistsError("评测输出已存在，请使用新的 output 目录，保留历史结果")
    records_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    write_json(args.output / "metrics.json", {
        "schema_version": 1, "split": args.split, "final": args.final, "metrics": metrics,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "model_config_hash": manifest.get("config_hash") if manifest else None,
        "isolation_checked": manifest is not None,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens} if args.checkpoint else None,
        "scope": "MASSIVE 中英 JSON 意图/槽位任务；不能解释为通用聊天或推理准确率。",
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
