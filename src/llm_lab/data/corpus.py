"""读取有版本和校验和的 flat uint32 token 流。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import LANGUAGES, ensure_under, sha256_file, split_name


class TokenCorpus:
    def __init__(self, manifest_path: str | Path, split: str = "train", seq_len: int = 512):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("不支持的 token manifest schema_version")
        tokenizer_item = self.manifest["tokenizer"]
        tokenizer_path = ensure_under(self.root / tokenizer_item["path"], self.root)
        if sha256_file(tokenizer_path) != tokenizer_item["sha256"]:
            raise ValueError("tokenizer 校验和不符")
        if seq_len < 1:
            raise ValueError("seq_len 必须为正数")
        self.seq_len = int(seq_len)
        self.split = split_name(split)
        self.streams = {}
        for language in LANGUAGES:
            item = self.manifest["splits"][self.split][language]
            if item["dtype"] != "uint32":
                raise ValueError("token 流必须为 uint32")
            path = ensure_under(self.root / item["path"], self.root)
            count = int(item["token_count"])
            if path.stat().st_size != count * 4:
                raise ValueError(f"token_count 与文件大小不一致：{path}")
            if item.get("sha256") and sha256_file(path) != item["sha256"]:
                raise ValueError(f"token 文件校验和不符：{path}")
            self.streams[language] = np.memmap(path, dtype="<u4", mode="r") if count else np.array([], dtype="<u4")

    def get_language_tokens(self, language: str):
        return self.streams[language]

    def sample_batch(self, language: str, batch_size: int, *, seed: int | None = None, rng=None):
        """返回 [B,T] 的输入与右移一位的标签；允许调用方保存 RNG 状态恢复采样。"""
        import torch

        tokens = self.streams[language]
        if batch_size < 1 or len(tokens) <= self.seq_len:
            raise ValueError("batch_size 必须为正；该语言 token 数必须大于 seq_len")
        generator = rng if rng is not None else np.random.default_rng(seed)
        starts = generator.integers(0, len(tokens) - self.seq_len, size=batch_size)
        x = np.stack([tokens[start:start + self.seq_len] for start in starts]).astype(np.int64)
        y = np.stack([tokens[start + 1:start + self.seq_len + 1] for start in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return sum(max(0, (len(tokens) - 1) // self.seq_len) for tokens in self.streams.values())
