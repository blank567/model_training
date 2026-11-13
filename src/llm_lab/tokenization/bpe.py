"""Byte-level BPE 的训练与加载，不依赖远程模型 tokenizer。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
EOS_TOKEN = "<|eos|>"


def train_bpe(texts: Iterable[str], output_path: str | Path, *, vocab_size: int = 8192, min_frequency: int = 2):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if vocab_size < 260:
        raise ValueError("Byte-level BPE vocab_size 至少为 260，以覆盖 256 字节和特殊符号")
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[PAD_TOKEN, UNK_TOKEN, EOS_TOKEN],
        show_progress=False,
    )
    # texts 必须由调用方从 train 分区提供；prepare 会保存实际参与拟合的文档 hash。
    tokenizer.train_from_iterator(texts, trainer=trainer)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
