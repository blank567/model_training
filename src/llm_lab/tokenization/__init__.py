"""只用训练分区拟合的 Byte-level BPE。"""

from .bpe import load_tokenizer, train_bpe

__all__ = ["train_bpe", "load_tokenizer"]
