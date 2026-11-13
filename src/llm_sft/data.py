"""自研 SFT 的已验收数据读取、一次错位和可恢复逐轮采样。

只复用冻结分词结果，不运行模型或重新构建原始语料。右侧 PAD 配合因果
注意力不会影响前面的有效位置；PAD 与用户提示均通过 -100 排除出 Loss。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any


IDENTITY_FIELDS = ("schema_version", "kind", "splits", "tokenizer", "tokenization", "chinese_normalization")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(directory: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("SFT manifest 必须声明非空相对文件名")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("SFT 文件路径越界")
    path = directory / relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory) or not path.is_file():
        raise ValueError("SFT 文件缺失、为符号链接或越界")
    return path


def deterministic_epoch_batches(length: int, micro: int, accum: int, seed: int, epoch: int) -> list[list[list[int]]]:
    """返回 optimizer 更新组 → microbatch → 行号，每轮恰好覆盖一次。

    局部 RNG 不改变模型随机状态。恢复只需保存 epoch 和下一更新组游标，
    重新生成同一计划并跳过已消费组；最后不足 micro/accum 的组完整保留。
    """
    for name, value, minimum in (("length", length, 0), ("micro", micro, 1), ("accum", accum, 1),
                                 ("seed", seed, 0), ("epoch", epoch, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} 必须是大于等于 {minimum} 的整数")
    if seed >= 2**64:
        raise ValueError("seed 必须小于 2**64")
    order = list(range(length))
    random.Random(f"llm-sft:{seed}:{epoch}").shuffle(order)
    micros = [order[index:index + micro] for index in range(0, length, micro)]
    return [micros[index:index + accum] for index in range(0, len(micros), accum)]


class SFTCorpus:
    """读取已冻结的一个分区；默认不允许训练路径读独立测试题。"""

    def __init__(self, manifest_path: str | Path, split: str, require_acceptance: bool = True,
                 *, allow_test: bool = False):
        if split not in {"train", "validation", "pilot"} and not (split == "test" and allow_test):
            raise ValueError("SFT 分区只允许 train/validation/pilot；test 需要显式 allow_test=True")
        if type(require_acceptance) is not bool or type(allow_test) is not bool:
            raise ValueError("require_acceptance/allow_test 必须为 bool")
        raw_path = Path(manifest_path)
        if raw_path.is_symlink():
            raise ValueError("SFT manifest 不能是符号链接")
        self.manifest_path = raw_path.resolve()
        self.directory = self.manifest_path.parent
        if any((self.directory / name).exists() for name in ("SUPERSEDED.json", "DO_NOT_TRAIN.json")):
            raise ValueError("该 SFT 数据已废弃或隔离，禁止训练")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest = self.manifest
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "sft_tokenized_conversations":
            raise ValueError("不支持的 SFT 数据格式")
        if set(manifest.get("splits", {})) != {"train", "validation", "test", "pilot"}:
            raise ValueError("SFT 数据必须保留完整 train/validation/test/pilot 分区身份")
        try:
            identity = canonical_hash({name: manifest[name] for name in IDENTITY_FIELDS})
        except KeyError as error:
            raise ValueError("SFT manifest 缺少身份字段") from error
        if manifest.get("data_identity") != identity:
            raise ValueError("SFT data_identity 与实际 manifest 不一致")
        self.split = split
        self.fixture = manifest.get("fixture") is True
        if not require_acceptance and not self.fixture:
            raise ValueError("正式 SFT 数据不能跳过独立验收；仅 CPU fixture 可关闭检查")
        if not self.fixture and manifest["chinese_normalization"].get("mode") != "t2s":
            raise ValueError("正式 SFT 数据必须采用已验收的简体 t2s 规范")
        self.pad_id = manifest["tokenizer"]["pad_id"]
        self.unk_id = manifest["tokenizer"]["unk_id"]
        self.eos_id = manifest["tokenizer"]["eos_id"]
        self.vocab_size = manifest["tokenizer"]["vocab_size"]
        if any(type(value) is not int for value in (self.pad_id, self.unk_id, self.eos_id, self.vocab_size)):
            raise ValueError("SFT 词表 ID 必须为整数")
        if len({self.pad_id, self.unk_id, self.eos_id}) != 3 or not all(0 <= value < self.vocab_size for value in (self.pad_id, self.unk_id, self.eos_id)):
            raise ValueError("SFT 特殊 Token ID 不合法")
        self.tokenizer_path = _file(self.directory, manifest["tokenizer"]["path"])
        tokenizer_sha = sha256_file(self.tokenizer_path)
        if tokenizer_sha != manifest["tokenizer"]["sha256"]:
            raise ValueError("SFT tokenizer SHA256 不匹配")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        if tokenizer.get_vocab_size() != self.vocab_size or any(tokenizer.token_to_id(token) != identifier
                for token, identifier in (("<|pad|>", self.pad_id), ("<|unk|>", self.unk_id), ("<|eos|>", self.eos_id))):
            raise ValueError("SFT 分词器词表或特殊 ID 与 manifest 不一致")
        if manifest["tokenization"].get("encoding") != "prefix_and_response_separately":
            raise ValueError("SFT 编码边界协议不兼容")
        self.max_sequence_tokens = manifest["tokenization"]["max_sequence_tokens"]
        if type(self.max_sequence_tokens) is not int or not 2 <= self.max_sequence_tokens <= 1024:
            raise ValueError("SFT 上下文上限不合法")
        acceptance_sha = None
        report = None
        if require_acceptance:
            accepted = manifest.get("acceptance", {})
            if accepted.get("status") != "passed":
                raise ValueError("SFT 数据尚未完成独立验收")
            report_path = _file(self.directory, accepted.get("report"))
            acceptance_sha = sha256_file(report_path)
            if acceptance_sha != accepted.get("sha256"):
                raise ValueError("SFT 验收报告 SHA256 不匹配")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("status") != "passed" or report.get("data_identity") != identity
                    or report.get("tokenizer_sha256") != tokenizer_sha
                    or report.get("fixture", False) != self.fixture
                    or (not self.fixture and report.get("structurally_ready") is not True)):
                raise ValueError("SFT 验收报告未绑定当前数据身份")
            checks = report.get("checks", {})
            required_checks = ("file_hashes", "retokenization", "answer_mask_and_eos", "normalized_text_identity",
                               "declared_group_and_exact_split_isolation", "pilot_subset", "language_token_balance",
                               "explicit_quality_rules")
            if not all(checks.get(name) is True for name in required_checks) or (not self.fixture and checks.get("chinese_fixed_point") is not True):
                raise ValueError("SFT 独立验收检查项缺失或未通过")
        # 只检查其他分区的文件摘要，不解析其题目；防止已验收文件被悄悄修改。
        for name, spec in manifest["splits"].items():
            path = _file(self.directory, spec["path"])
            if sha256_file(path) != spec["sha256"]:
                raise ValueError(f"SFT {name} 数据 SHA256 不匹配")
            if report is not None:
                checked = report.get("splits", {}).get(name, {})
                if checked.get("sha256") != spec["sha256"] or checked.get("examples") != spec["examples"]:
                    raise ValueError("SFT 验收中的分区身份不匹配")
        try:
            spec = manifest["splits"][split]
        except KeyError as error:
            raise ValueError("SFT 分区不存在") from error
        self.rows = []
        self.language_indices: dict[str, list[int]] = {"zh": [], "en": []}
        seen_ids = set()
        with _file(self.directory, spec["path"]).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                self._validate_row(row, number)
                if row["id"] in seen_ids:
                    raise ValueError("SFT 分区出现重复 ID")
                seen_ids.add(row["id"])
                self.language_indices[row["language"]].append(len(self.rows))
                self.rows.append(row)
        if type(spec["examples"]) is not int or len(self.rows) != spec["examples"] or not self.rows:
            raise ValueError("SFT 样本数与 manifest 不一致或分区为空")
        self.identity = {"manifest_sha256": sha256_file(self.manifest_path), "data_identity": identity,
                         "tokenizer_sha256": tokenizer_sha, "split": split,
                         "split_sha256": spec["sha256"], "acceptance_sha256": acceptance_sha,
                         "examples": len(self.rows)}

    def _validate_row(self, row: dict, number: int) -> None:
        label = f"{self.split}:{number}"
        if (row.get("schema_version") != 1 or row.get("split") != self.split
                or not isinstance(row.get("id"), str) or not row["id"]
                or row.get("language") not in self.language_indices):
            raise ValueError(label + " 样本 schema/身份/语言不合法")
        ids, mask = row.get("input_ids"), row.get("loss_mask")
        if not isinstance(ids, list) or not 2 <= len(ids) <= self.max_sequence_tokens:
            raise ValueError(label + " Token 长度不合法")
        if (not all(type(value) is int and 0 <= value < self.vocab_size for value in ids)
                or self.pad_id in ids or self.unk_id in ids or self.eos_id in ids[:-1]
                or ids[-1] != self.eos_id):
            raise ValueError(label + " Token ID/EOS 不合法")
        if not isinstance(mask, list) or len(mask) != len(ids) or any(type(value) is not int or value not in (0, 1) for value in mask):
            raise ValueError(label + " Loss Mask 不合法")
        before, after = row.get("prompt_token_count"), row.get("answer_token_count")
        if (type(before) is not int or before < 1 or type(after) is not int or after < 2
                or before + after != len(ids) or mask != [0] * before + [1] * after):
            raise ValueError(label + " prompt/回答监督边界不一致")

    def __len__(self) -> int:
        return len(self.rows)

    def collate(self, indices: list[int], device: str = "cpu") -> dict:
        """输出 [B,T-1] 的 x/y；y 已错位且提示/PAD 均为 -100。"""
        import torch
        if not isinstance(indices, (list, tuple)) or not indices or any(type(index) is not int or not 0 <= index < len(self.rows) for index in indices):
            raise ValueError("SFT batch 需要非空且合法的行号")
        if self.fixture and torch.device(device).type != "cpu":
            raise ValueError("fixture 数据只能用于 CPU 正确性测试")
        rows = [self.rows[index] for index in indices]
        width = max(len(row["input_ids"]) - 1 for row in rows)
        x = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        y = torch.full((len(rows), width), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            values = torch.tensor(row["input_ids"], dtype=torch.long)
            mask = torch.tensor(row["loss_mask"][1:], dtype=torch.bool)
            length = len(values) - 1
            x[index, :length] = values[:-1]
            y[index, :length] = values[1:].masked_fill(~mask, -100)
        # 损失归一化应使用本批真正的回答 Token 数，不能用补齐后的矩阵尺寸。
        supervised = int((y != -100).sum())
        return {"x": x.to(device), "y": y.to(device), "supervised_tokens": supervised,
                "input_tokens": sum(len(row["input_ids"]) - 1 for row in rows),
                "sequence_tokens": sum(len(row["input_ids"]) for row in rows),
                "language": [row["language"] for row in rows],
                "sample_ids": [row["id"] for row in rows]}
