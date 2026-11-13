"""将 MASSIVE 原始标注转换为指令样本及明确标注的程序偏好。"""

from __future__ import annotations

import json
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

from .common import (
    LANGUAGES, SPLITS, config_path, input_provenance, json_line, make_output_dir, normalize_text,
    positive_int, read_jsonl, sha256_file, split_name, text_hash, write_json, write_jsonl,
)


LOCALES = {"en-US": "en", "zh-CN": "zh"}
SLOT_PATTERN = re.compile(r"\[([A-Za-z0-9_]+)\s*:\s*([^\[\]]+)\]")


def parse_slots(utterance: str, annotated: str) -> list[dict[str, str]]:
    """校验去掉标记后仍是原句，保留相同槽位的多次出现。"""
    slots = []
    parts = []
    previous = 0
    for match in SLOT_PATTERN.finditer(annotated):
        parts.append(annotated[previous:match.start()])
        value = match.group(2).strip()
        if not value:
            raise ValueError("空槽位值")
        slots.append({"slot": match.group(1), "value": value})
        parts.append(value)
        previous = match.end()
    parts.append(annotated[previous:])
    reconstructed = normalize_text("".join(parts))
    if reconstructed != normalize_text(utterance):
        raise ValueError("annot_utt 不能无损还原 utt，需人工审计该标注")
    return slots


def make_completion(intent: str, slots: list[dict[str, str]]) -> str:
    return json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False, separators=(",", ":"))


def make_prompt(language: str, utterance: str, intents: list[str]) -> str:
    labels = ", ".join(intents)
    if language == "zh":
        return ("请识别用户话语的意图并抽取所有槽位。仅输出一个 JSON 对象，禁止解释、Markdown 或额外字段。"
                '格式为 {"intent":"意图标签","slots":[{"slot":"槽位标签","value":"原文值"}]}。'
                "没有槽位时 slots 为 []；重复出现的槽位需全部保留。\n"
                f"可选意图：{labels}\n用户话语：{utterance}")
    return ("Identify the intent and extract every slot from the utterance. Return only one JSON object, "
            "without explanations, Markdown, or additional fields. "
            'Use {"intent":"intent_label","slots":[{"slot":"slot_label","value":"verbatim value"}]}. '
            "Use [] when there are no slots, and preserve repeated slots.\n"
            f"Allowed intents: {labels}\nUtterance: {utterance}")


def construct_preference(row: dict[str, str], intents: list[str]) -> dict[str, str]:
    """仅用金标和一个确定性错误意图构造偏好；不把它称为真人反馈。"""
    gold = json.loads(row["completion"])
    alternatives = [intent for intent in intents if intent != gold["intent"]]
    if not alternatives:
        raise ValueError("至少需要两个训练意图类别，才能构造合法但错误的负例")
    index = int(text_hash(row["source_id"] + ":" + row["language"])[:16], 16) % len(alternatives)
    rejected = make_completion(alternatives[index], gold["slots"])
    negative = json.loads(rejected)
    if negative["intent"] == gold["intent"] or negative["slots"] != gold["slots"]:
        raise AssertionError("程序负例未形成预期的单项错误")
    return {"source_id": row["source_id"], "language": row["language"], "prompt": row["prompt"],
            "chosen": row["completion"], "rejected": rejected}


def prepare_massive(config: dict[str, Any], root: str | Path | None = None) -> Path:
    root = Path(root or Path.cwd()).resolve()
    max_docs = positive_int(config, "max_docs")
    max_bytes = positive_int(config, "max_bytes")
    selected = []
    groups = {}
    seen_ids = set()
    stats: Counter = Counter()
    for path_value in config["inputs"]:
        for original in islice(read_jsonl(config_path(path_value, root)), max_docs - stats["scanned"]):
            size = len(json_line(original).encode("utf-8"))
            if stats["bytes"] + size > max_bytes:
                stats["stopped_max_bytes"] += 1
                break
            stats["scanned"] += 1
            stats["bytes"] += size
            language = LOCALES.get(original.get("locale"))
            if language is None:
                stats["other_locale"] += 1
                continue
            split = split_name(original.get("partition", ""))
            if "id" not in original:
                raise ValueError("MASSIVE 缺少原始源 id")
            source_id = f"massive:{original['id']}"
            if source_id in groups and groups[source_id] != split:
                raise ValueError(f"MASSIVE 跨语言源 ID 分区冲突：{source_id}")
            groups[source_id] = split
            row_key = (source_id, language)
            if row_key in seen_ids:
                raise ValueError(f"MASSIVE 重复的 source_id/locale：{row_key}")
            seen_ids.add(row_key)
            utterance = normalize_text(original.get("utt"))
            intent = original.get("intent")
            if not utterance or not isinstance(intent, str) or not intent:
                raise ValueError("MASSIVE 的 utt/intent 不能为空")
            try:
                slots = parse_slots(utterance, original.get("annot_utt", ""))
            except ValueError:
                if config.get("invalid_annotation_policy", "error") == "skip":
                    stats["invalid_annotation"] += 1
                    continue
                raise
            selected.append({"source_id": source_id, "language": language, "split": split,
                             "utterance": utterance, "completion": make_completion(intent, slots)})
        if stats.get("stopped_max_docs") or stats.get("stopped_max_bytes"):
            break

    # 先保留官方分区，再去除跨分区重复输入；test 优先，不把它移动到训练分区。
    priority = {"test": 0, "validation": 1, "train": 2}
    # 一种语言发生跨分区重复时，整个源 ID 组（包含其他语言版本）一起丢弃。
    # 这样不会留下低优先级样本的翻译版本，间接泄漏到高优先级评测分区。
    text_groups = {}
    for row in selected:
        text_groups.setdefault((row["language"], text_hash(row["utterance"])), []).append(row)
    dropped_groups = set()
    for duplicate_group in text_groups.values():
        split_set = {row["split"] for row in duplicate_group}
        if len(split_set) > 1:
            if config.get("cross_split_duplicate_policy", "error") != "keep_evaluation":
                raise ValueError("MASSIVE 同一输入跨分区重复；审计后可显式设 keep_evaluation")
            best_split = min(split_set, key=priority.__getitem__)
            dropped_groups.update(row["source_id"] for row in duplicate_group if row["split"] != best_split)
    stats["cross_split_duplicate_source_groups_dropped"] = len(dropped_groups)
    seen_text = {}
    retained = []
    for row in sorted(selected, key=lambda item: (priority[item["split"]], item["source_id"], item["language"])):
        if row["source_id"] in dropped_groups:
            stats["cross_split_duplicate_dropped"] += 1
            continue
        key = (row["language"], text_hash(row["utterance"]))
        if key in seen_text:
            previous = seen_text[key]
            if previous["split"] != row["split"]:
                raise AssertionError("源 ID 组去重后仍存在跨 split 重复输入")
            if previous["completion"] == row["completion"]:
                stats["same_split_duplicate_dropped"] += 1
                continue
            stats["ambiguous_same_input"] += 1
        seen_text[key] = row
        retained.append(row)

    # 类别表仅来源于 train；validation/test 不参与标签空间和负例规则的拟合。
    intents = sorted({json.loads(row["completion"])["intent"] for row in retained if row["split"] == "train"})
    if len(intents) < 2:
        raise ValueError("MASSIVE train 至少需要两个意图类别")
    output = config_path(config["output_dir"], root)
    make_output_dir(output, overwrite=bool(config.get("overwrite", False)))
    manifest: dict[str, Any] = {
        "schema_version": 1, "kind": "massive_instruction_tasks", "source_version": config.get("source_version", "fixture"),
        "source_license": "CC-BY-4.0 (MASSIVE data); Apache-2.0 applies separately to upstream code",
        "completion_schema": {"intent": "string", "slots": [{"slot": "string", "value": "string"}]},
        "preference_provenance": {"type": "programmatic_gold_vs_wrong_intent", "human_preference": False,
                                  "rule": "保持相同 JSON/schema/槽位，仅替换为另一个训练意图标签",
                                  "limitation": "不代表人类偏好；偏好学习结论限于此人工负例分布"},
        "split_policy": "保留官方 partition；先按跨语言源 ID 审计，再构造任务和偏好；test 仅最终评测",
        "intents_from_train": intents, "statistics": dict(stats), "splits": {}, "config": config,
        "input_provenance": input_provenance(config["inputs"], root, max_bytes),
    }
    for split in SPLITS:
        rows = [{"source_id": row["source_id"], "language": row["language"],
                 "prompt": make_prompt(row["language"], row["utterance"], intents), "completion": row["completion"]}
                for row in retained if row["split"] == split]
        rows.sort(key=lambda row: (row["source_id"], row["language"]))
        language_counts = Counter(row["language"] for row in rows)
        if config.get("require_all_splits", True) and any(language_counts[lang] == 0 for lang in LANGUAGES):
            raise ValueError(f"{split} 缺少中/英样本；先获取足够的官方分区数据")
        sft_path = output / "sft" / f"{split}.jsonl"
        dpo_path = output / "dpo" / f"{split}.jsonl"
        write_jsonl(sft_path, rows)
        write_jsonl(dpo_path, (construct_preference(row, intents) for row in rows))
        manifest["splits"][split] = {"sft": str(sft_path.relative_to(output)).replace("\\", "/"),
                                    "dpo": str(dpo_path.relative_to(output)).replace("\\", "/"),
                                    "rows": len(rows), "languages": dict(language_counts),
                                    "sft_sha256": sha256_file(sft_path), "dpo_sha256": sha256_file(dpo_path)}
    write_jsonl(output / "source_group_audit.jsonl", (
        {"source_id": row["source_id"], "language": row["language"], "split": row["split"],
         "input_sha256": text_hash(row["utterance"]), "dropped_source_group": row["source_id"] in dropped_groups}
        for row in selected))
    write_json(output / "task_manifest.json", manifest)
    return output / "task_manifest.json"
