"""读取准备好的双语 uint32 Token 流，不在 GPU 训练中在线清洗数据。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from llm_lab.common.io import canonical_hash, read_json, sha256_file


class BinaryCorpus:
    def __init__(self, manifest_path: str | Path, split: str, seq_len: int):
        self.path = Path(manifest_path).resolve()
        self.manifest = read_json(self.path)
        self.split, self.seq_len = split, seq_len
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.arrays = {}
        file_hashes = {}
        root = self.path.parent
        for language in ("en", "zh"):
            spec = self.manifest["splits"][split][language]
            file = (root / spec["path"]).resolve()
            if not file.is_relative_to(root) or spec["dtype"] != "uint32":
                raise ValueError("Corpus files must be relative uint32 paths inside manifest directory")
            count = int(spec["token_count"])
            if count <= seq_len or file.stat().st_size != count * 4:
                raise ValueError(f"Invalid/too short {split}/{language} token stream")
            digest = sha256_file(file)
            if spec.get("sha256") and digest != spec["sha256"]:
                raise ValueError(f"Corpus checksum mismatch: {file}")
            self.arrays[language] = np.memmap(file, dtype="<u4", mode="r")
            file_hashes[language] = digest
        tokenizer = self.manifest["tokenizer"]
        tokenizer_file = (root / tokenizer["path"]).resolve()
        if not tokenizer_file.is_relative_to(root) or sha256_file(tokenizer_file) != tokenizer["sha256"]:
            raise ValueError("Tokenizer path/checksum mismatch")
        self.fingerprint = canonical_hash({"manifest": sha256_file(self.path), "files": file_hashes, "split": split})

    def batch(self, language: str, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        stream = self.arrays[language]
        offsets = rng.integers(0, len(stream) - self.seq_len, size=batch_size)
        # 复制只读 memmap，避免不可写数组与 PyTorch 共享内存。
        chunks = np.stack([stream[int(start):int(start) + self.seq_len + 1] for start in offsets]).astype(np.int64)
        return torch.from_numpy(chunks[:, :-1].copy()), torch.from_numpy(chunks[:, 1:].copy())

