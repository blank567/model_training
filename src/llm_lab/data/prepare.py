"""制备入口：python -m llm_lab.data.prepare --config configs/data/prepare_smoke.json。"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from llm_lab.tokenization.bpe import EOS_TOKEN, train_bpe

from .common import (
    LANGUAGES, SPLITS, assigned_split, config_path, json_line, load_config,
    input_provenance, make_output_dir, normalize_text, positive_int, read_jsonl, sha256_file,
    split_name, text_hash, write_json, write_jsonl,
)


def clean_documents(config: dict[str, Any], root: Path):
    """有限扫描，先文档去重/源分区再切 token，避免同一篇文章被切到不同 split。"""
    max_docs = positive_int(config, "max_docs")
    max_bytes = positive_int(config, "max_bytes")
    fractions = config.get("split_fractions", {"train": 0.98, "validation": 0.01, "test": 0.01})
    seed = int(config.get("seed", 42))
    quality = config.get("quality", {})
    min_chars = int(quality.get("min_chars", 40))
    max_chars = int(quality.get("max_chars", 100000))
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("quality 的 min_chars/max_chars 非法")
    groups: dict[str, str] = {}
    hashes: dict[str, str] = {}
    result = []
    stats: Counter = Counter()
    for path_value in config["inputs"]:
        for row in islice(read_jsonl(config_path(path_value, root)), max_docs - stats["scanned"]):
            stats["scanned"] += 1
            source_id = str(row.get("source_id", "")).strip()
            language = row.get("language")
            if not source_id or language not in LANGUAGES:
                raise ValueError("预训练文档需要非空 source_id 和 language=en/zh")
            text = normalize_text(row.get("text"))
            raw_split = row.get("split", row.get("partition"))
            split = split_name(raw_split) if raw_split is not None else assigned_split(source_id, seed, fractions)
            if source_id in groups and groups[source_id] != split:
                raise ValueError(f"同源跨 split 泄漏：{source_id}")
            groups[source_id] = split
            if len(text) < min_chars or len(text) > max_chars:
                stats["quality_length"] += 1
                continue
            # 极端单字符重复和重复行可过滤；阈值显式记录，保留可复查统计。
            repetition = max(Counter(text).values()) / len(text)
            lines = [line for line in text.splitlines() if line]
            unique_line_ratio = len(set(lines)) / max(1, len(lines))
            if repetition > float(quality.get("max_character_fraction", 0.5)):
                stats["quality_character_repetition"] += 1
                continue
            if unique_line_ratio < float(quality.get("min_unique_line_ratio", 0.5)):
                stats["quality_line_repetition"] += 1
                continue
            digest = text_hash(text)
            if digest in hashes:
                if hashes[digest] != split:
                    raise ValueError(f"相同规范化文档跨 split 泄漏：{source_id}")
                stats["duplicates"] += 1
                continue
            normalized = {"source_id": source_id, "language": language, "split": split,
                          "text": text, "text_sha256": digest}
            size = len(json_line(normalized).encode("utf-8"))
            if stats["bytes"] + size > max_bytes:
                stats["stopped_max_bytes"] += 1
                return result, dict(stats)
            hashes[digest] = split
            stats["bytes"] += size
            stats["accepted"] += 1
            result.append(normalized)
    return result, dict(stats)


def prepare_pretrain(config: dict[str, Any], root: str | Path | None = None) -> Path:
    root = Path(root or Path.cwd()).resolve()
    max_tokens = positive_int(config, "max_tokens")
    documents, stats = clean_documents(config, root)
    training = [row for row in documents if row["split"] == "train"]
    if not training:
        raise ValueError("清洗后 train 文档为空，不能训练 tokenizer")
    output = config_path(config["output_dir"], root)
    make_output_dir(output, overwrite=bool(config.get("overwrite", False)))
    tokenizer_path = output / "tokenizer.json"
    token_config = config.get("tokenizer", {})
    tokenizer = train_bpe((row["text"] for row in training), tokenizer_path,
                          vocab_size=int(token_config.get("vocab_size", 8192)),
                          min_frequency=int(token_config.get("min_frequency", 2)))
    requested_vocab = int(token_config.get("vocab_size", 8192))
    if tokenizer.get_vocab_size() != requested_vocab:
        raise ValueError(f"训练语料不足以形成指定词表：请求 {requested_vocab}，实际 {tokenizer.get_vocab_size()}。"
                         "请增加 train 数据，或显式一起调整 tokenizer/model vocab_size；不会静默更改模型参数量。")
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    buffers = {split: {language: [] for language in LANGUAGES} for split in SPLITS}
    counts = {split: {language: 0 for language in LANGUAGES} for split in SPLITS}
    consumed = 0
    emitted_docs = []
    token_limits = config.get("max_tokens_per_split_per_language", {})
    for row in documents:
        tokens = tokenizer.encode(row["text"], add_special_tokens=False).ids + [eos_id]
        split, language = row["split"], row["language"]
        per_stream_limit = int(token_limits.get(split, max_tokens))
        if len(tokens) + consumed > max_tokens:
            stats["documents_skipped_token_budget"] = stats.get("documents_skipped_token_budget", 0) + 1
            continue
        if counts[split][language] + len(tokens) > per_stream_limit:
            continue
        # 不在文章中间截断：每个已接受文档结尾都有 EOS。
        buffers[split][language].append(np.asarray(tokens, dtype="<u4"))
        counts[split][language] += len(tokens)
        consumed += len(tokens)
        emitted_docs.append(row)
    if config.get("require_all_streams", True):
        missing = [f"{split}/{lang}" for split in SPLITS for lang in LANGUAGES if counts[split][lang] == 0]
        if missing:
            raise ValueError(f"缺少非空 token 流：{', '.join(missing)}；增加数据预算或检查分区")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pretrain_token_streams",
        "tokenizer": {"path": "tokenizer.json", "sha256": sha256_file(tokenizer_path),
                      "vocab_size": tokenizer.get_vocab_size(), "eos_id": eos_id,
                      "pad_id": tokenizer.token_to_id("<|pad|>"), "trained_on_split": "train"},
        "splits": {},
        "sampling": {"unit": "effective_tokens", "language_probabilities": {"en": 0.5, "zh": 0.5},
                     "note": "训练器从两个语言 token 流等概率采样相同长度块；不按文档条数配比。"},
        "budgets": {key: config[key] for key in ("max_docs", "max_bytes", "max_tokens")},
        "quality": config.get("quality", {}), "statistics": stats,
        "token_count": consumed,
        "tokenizer_training_document_hashes": sorted(row["text_sha256"] for row in training),
        "config": config,
        "input_provenance": input_provenance(config["inputs"], root, config["max_bytes"]),
    }
    for split in SPLITS:
        manifest["splits"][split] = {}
        for language in LANGUAGES:
            path = output / f"{split}.{language}.bin"
            with path.open("wb") as handle:
                for array in buffers[split][language]:
                    array.tofile(handle)
            manifest["splits"][split][language] = {
                "path": path.name, "token_count": counts[split][language], "dtype": "uint32",
                "byte_order": "little", "sha256": sha256_file(path),
            }
    write_jsonl(output / "documents.jsonl", emitted_docs)
    # 审计包含 tokenizer 看过但因 token 预算未输出的训练文档，不隐藏其来源。
    write_jsonl(output / "document_audit.jsonl", ({key: row[key] for key in
                ("source_id", "language", "split", "text_sha256")} for row in documents))
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description="制备隔离且有预算上限的预训练/任务数据")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    if config.get("kind", "pretrain") == "massive":
        from .massive import prepare_massive
        result = prepare_massive(config, root)
    else:
        result = prepare_pretrain(config, root)
    print(result)


if __name__ == "__main__":
    main()
