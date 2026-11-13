"""独立核验 SFT 数据文件、Token、回答掩码和分区隔离；不加载模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from sft_data_quality import fingerprint, normalize, quality_reason


IDENTITY_FIELDS = ("schema_version", "kind", "splits", "tokenizer", "tokenization", "chinese_normalization")
PREFIX = "System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n"
EMPTY_SYSTEM_PREFIX = "User:\n{prompt}\n\nAssistant:\n"
SOURCE_FIELDS = ("dataset", "revision", "file", "row_index", "source_subset", "raw_row_sha256",
                 "original_prompt_sha256", "original_response_sha256", "original_context_sha256",
                 "normalized_text_sha256", "license")
SHA_FIELDS = ("raw_row_sha256", "original_prompt_sha256", "original_response_sha256",
              "original_context_sha256", "normalized_text_sha256")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dataset_identity(manifest: dict) -> str:
    """不包含验收报告或时间，防止 manifest/report 的校验身份形成循环。"""
    return canonical_hash({key: manifest[key] for key in IDENTITY_FIELDS})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contained_file(directory: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and bool(relative), "manifest 文件路径必须为非空字符串")
    name = Path(relative)
    _require(not name.is_absolute() and ".." not in name.parts, "manifest 文件路径越界")
    path = directory / name
    _require(not path.is_symlink(), "manifest 文件不能是符号链接")
    resolved = path.resolve()
    _require(resolved.is_relative_to(directory) and resolved.is_file(), "manifest 文件不存在或越界")
    return resolved


def _normalizer(manifest: dict):
    # 数据规范模块不会导入 torch 或模型；真实中文转换始终使用已有官方 C++ OpenCC。
    from llm_lab.data.text_normalization import build_chinese_normalizer
    info = manifest["chinese_normalization"]
    fixture = manifest.get("fixture") is True
    _require(isinstance(info, dict), "缺少中文规范化身份")
    _require(info.get("mode") == "t2s" or (fixture and info.get("mode") == "none"),
             "正式 SFT 必须采用 t2s；none 仅允许 fixture=true")
    normalizer = build_chinese_normalizer(info["mode"])
    _require(normalizer.provenance == info, "中文规范化身份与实际实现不一致")
    return normalizer


def _check_source(row: dict, label: str) -> None:
    source = row.get("source")
    _require(isinstance(source, dict) and all(key in source for key in SOURCE_FIELDS), label + " 来源字段缺失")
    for key in ("dataset", "file", "license"):
        _require(isinstance(source[key], str) and bool(source[key]), label + " 来源字符串缺失: " + key)
    _require(isinstance(source["source_subset"], str), label + " source_subset 类型错误")
    _require(isinstance(source["revision"], str) and bool(re.fullmatch(r"[0-9a-f]{40}", source["revision"])),
             label + " 来源 revision 必须为完整 commit SHA")
    _require(type(source["row_index"]) is int and source["row_index"] >= 0, label + " 原始行号无效")
    for key in SHA_FIELDS:
        _require(isinstance(source[key], str) and bool(re.fullmatch(r"[0-9a-f]{64}", source[key])),
                 label + " 来源 SHA 无效: " + key)
    texts = {key: row[key] for key in ("system", "prompt", "response", "input_context")}
    _require(canonical_hash(texts) == source["normalized_text_sha256"], label + " 规范化正文身份不匹配")


def _check_row(row: dict, split: str, index: int, tokenizer, manifest: dict, normalizer) -> None:
    label = f"{split}:{index}"
    _require(isinstance(row, dict), label + " 必须为 JSON 对象")
    _require(row.get("schema_version") == 1 and row.get("split") == split, label + " schema/split 无效")
    for key in ("id", "group_id", "task"):
        _require(isinstance(row.get(key), str) and bool(row[key]), label + " 缺少 " + key)
    _require(row.get("language") in {"zh", "en"}, label + " language 无效")
    for key in ("system", "prompt", "response", "input_context"):
        _require(isinstance(row.get(key), str), label + " 文本字段必须为字符串: " + key)
        _require(normalize(row[key]) == row[key], label + " 文本不满足 Unicode/空白规范: " + key)
        if row["language"] == "zh":
            _require(normalizer.convert(row[key], "zh") == row[key], label + " 中文不在简体固定点: " + key)
    _require(bool(row["prompt"]) and bool(row["response"]), label + " 问答不能为空")
    _check_source(row, label)
    reason = quality_reason((row["system"] + "\n" + row["prompt"]).strip(), row["response"],
                            row["language"], row["source"]["source_subset"])
    _require(reason is None, label + " 仍触发明确质量规则: " + str(reason))
    input_ids, mask = row.get("input_ids"), row.get("loss_mask")
    _require(isinstance(input_ids, list) and bool(input_ids), label + " 缺少 input_ids")
    _require(isinstance(mask, list) and len(mask) == len(input_ids), label + " mask 长度不匹配")
    _require(all(type(value) is int for value in input_ids), label + " Token ID 必须为整数")
    _require(all(type(value) is int and value in {0, 1} for value in mask), label + " mask 必须为整数 0/1")
    info, settings = manifest["tokenizer"], manifest["tokenization"]
    _require(len(input_ids) <= settings["max_sequence_tokens"], label + " 超过上下文长度")
    _require(all(0 <= value < info["vocab_size"] for value in input_ids), label + " Token ID 超出词表")
    _require(not ({info["pad_id"], info["unk_id"]} & set(input_ids)), label + " 包含 PAD/UNK")
    _require(input_ids[-1] == info["eos_id"] and mask[-1] == 1, label + " 回答末尾 EOS/监督位无效")
    _require(info["eos_id"] not in input_ids[:-1], label + " 正文内出现 EOS")
    prompt_count, answer_count = row.get("prompt_token_count"), row.get("answer_token_count")
    _require(type(prompt_count) is int and prompt_count > 0, label + " prompt_token_count 无效")
    _require(type(answer_count) is int and answer_count > 1, label + " answer_token_count 无效")
    _require(prompt_count + answer_count == len(input_ids), label + " Token 计数不匹配")
    _require(mask == [0] * prompt_count + [1] * answer_count, label + " prompt/回答 mask 边界错误")
    template = settings["prefix_template"] if row["system"] else settings["empty_system_prefix_template"]
    prefix = template.format(system=row["system"], prompt=row["prompt"])
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    answer_ids = tokenizer.encode(row["response"], add_special_tokens=False).ids + [info["eos_id"]]
    _require(len(prefix_ids) == prompt_count and len(answer_ids) == answer_count, label + " BPE 计数不一致")
    _require(prefix_ids + answer_ids == input_ids, label + " input_ids 与独立 BPE 重编码不一致")


def audit_dataset(directory: Path) -> dict:
    """失败抛 ValueError；通过时只报告文件/结构/明确规则，不宣称语义全正确。"""
    from tokenizers import Tokenizer
    directory = Path(directory).resolve()
    manifest_path = _contained_file(directory, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1 and manifest.get("kind") == "sft_tokenized_conversations",
             "不支持的 SFT manifest")
    _require(set(manifest.get("splits", {})) == {"train", "validation", "test", "pilot"}, "split 清单不完整")
    identity = dataset_identity(manifest)
    _require(manifest.get("data_identity") == identity, "data_identity 与 manifest 数据身份不匹配")
    settings = manifest["tokenization"]
    _require(settings.get("encoding") == "prefix_and_response_separately", "编码边界协议不支持")
    _require(settings.get("prefix_template") == PREFIX, "带 system 模板不匹配")
    _require(settings.get("empty_system_prefix_template") == EMPTY_SYSTEM_PREFIX, "无 system 模板不匹配")
    _require(type(settings.get("max_sequence_tokens")) is int and 1 <= settings["max_sequence_tokens"] <= 1024,
             "上下文长度配置无效")
    normalizer = _normalizer(manifest)
    tokenizer_info = manifest["tokenizer"]
    for key in ("eos_id", "pad_id", "unk_id", "vocab_size"):
        _require(type(tokenizer_info.get(key)) is int, "tokenizer ID/词表字段无效")
    _require(len({tokenizer_info[key] for key in ("eos_id", "pad_id", "unk_id")}) == 3,
             "特殊 Token ID 不能重合")
    tokenizer_path = _contained_file(directory, tokenizer_info.get("path"))
    _require(sha256_file(tokenizer_path) == tokenizer_info.get("sha256"), "tokenizer SHA256 不匹配")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    _require(tokenizer.get_vocab_size() == tokenizer_info["vocab_size"], "实际词表大小不匹配")
    for token, key in (("<|eos|>", "eos_id"), ("<|pad|>", "pad_id"), ("<|unk|>", "unk_id")):
        _require(tokenizer.token_to_id(token) == tokenizer_info[key], "实际特殊 Token ID 不匹配: " + key)
    seen = {key: {} for key in ("id", "group_id", "prompt", "input_context")}
    train_rows: dict[str, str] = {}
    result = {}
    for split in ("train", "validation", "test", "pilot"):
        spec = manifest["splits"][split]
        path = _contained_file(directory, spec.get("path"))
        _require(sha256_file(path) == spec.get("sha256"), split + " 文件 SHA256 不匹配")
        _require(type(spec.get("examples")) is int and spec["examples"] > 0, split + " 样本计数无效")
        count = total_tokens = 0
        answer_tokens = Counter({"zh": 0, "en": 0})
        language_examples = Counter({"zh": 0, "en": 0})
        ids_in_split = set()
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                _require(bool(line.strip()), split + " 出现空 JSONL 行")
                row = json.loads(line)
                _check_row(row, split, number, tokenizer, manifest, normalizer)
                identifier = row["id"]
                _require(identifier not in ids_in_split, split + " 存在重复 ID")
                ids_in_split.add(identifier)
                content_hash = canonical_hash({key: value for key, value in row.items() if key != "split"})
                if split == "pilot":
                    _require(train_rows.get(identifier) == content_hash, "pilot 包含 train 外样本或内容不一致")
                else:
                    for key in seen:
                        value = row[key] if key in {"id", "group_id"} else fingerprint(row[key])
                        if not value:
                            continue
                        previous_split = seen[key].get(value)
                        _require(previous_split is None or previous_split == split, "跨分区泄漏: " + key)
                        seen[key][value] = split
                    if split == "train":
                        train_rows[identifier] = content_hash
                count += 1
                total_tokens += len(row["input_ids"])
                answer_tokens[row["language"]] += row["answer_token_count"]
                language_examples[row["language"]] += 1
        _require(count == spec["examples"], split + " 样本数量与 manifest 不匹配")
        zh_share = answer_tokens["zh"] / sum(answer_tokens.values())
        _require(.45 <= zh_share <= .55, split + " 中英回答 Token 占比不在 45–55%")
        result[split] = {"examples": count, "input_tokens": total_tokens,
                         "answer_tokens": dict(answer_tokens), "language_examples": dict(language_examples),
                         "zh_answer_token_fraction": zh_share, "sha256": spec["sha256"]}
    return {"schema_version": 1, "status": "passed", "data_identity": identity,
            "fixture": manifest.get("fixture") is True,
            "structurally_ready": manifest.get("fixture") is not True,
            "splits": result, "tokenizer_sha256": tokenizer_info["sha256"],
            "checks": {"file_hashes": True, "retokenization": True, "answer_mask_and_eos": True,
                       "normalized_text_identity": True, "declared_group_and_exact_split_isolation": True,
                       "pilot_subset": True, "language_token_balance": True,
                       "chinese_fixed_point": normalizer.mode == "t2s", "explicit_quality_rules": True},
            "semantic_quality_verified": False,
            "limits": ["不证明所有回答的事实与语义正确。", "分区独立复核覆盖明确 group_id 与规范化精确键，不证明全部近重复或跨语言去重。",
                       "本验收不加载基础模型，也不等同 SFT GPU 训练或任务效果验收。"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_dataset(args.data_dir)
    if args.output:
        output = args.output.resolve()
        root = args.data_dir.resolve()
        _require(output.is_relative_to(root), "验收报告必须留在该数据目录")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        protected = {root / "manifest.json", _contained_file(root, manifest["tokenizer"]["path"])}
        protected.update(_contained_file(root, info["path"]) for info in manifest["splits"].values())
        _require(output not in protected, "报告不能覆盖 manifest、tokenizer 或 split 文件")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        _require(not temporary.is_symlink(), "报告临时文件不能是符号链接")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
