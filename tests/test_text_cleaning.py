"""优化后的类别过滤必须与原先逐字符 UCD 判据完全一致。"""

from __future__ import annotations

import random
import re
import unicodedata

import pytest

from llm_lab.data.common import _category_c_pattern, normalize_text


def original_normalize_text(text):
    """冻结优化前的完整规则，验证调用顺序、字符类别及空白语义。"""
    if not isinstance(text, str):
        raise ValueError("text 必须是字符串")
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def test_category_filter_matches_every_unicode_codepoint_including_surrogates():
    pattern = _category_c_pattern()
    for code in range(0x110000):
        character = chr(code)
        expected = character if character in "\n\t" or unicodedata.category(character)[0] != "C" else ""
        assert pattern.sub("", character) == expected, f"U+{code:06X}"


def test_full_unicode_string_matches_original_complete_cleaning():
    text = "".join(map(chr, range(0x110000)))
    expected = original_normalize_text(text)
    actual = normalize_text(text)
    assert actual == expected
    assert normalize_text(actual) == original_normalize_text(expected)


@pytest.mark.parametrize("text", [
    "", " \t\r\n\n\n ", "e\x00\u0301", "Cafe\u0301\r\n\tA\u00a0B\u2003C\n\n\nD",
    "\ud800x\udfff\U0010ffff\u200b\x00\x1f\ufeff\ue000\t\n",
    "中文\u200b測試，组合 e\u0301，emoji 👩\u200d💻 和 🌏。\r\n\t继续",
    "a\x1c\x1d\x1e\x1fb\x85c\u2028d\u2029e\v\f\n\tend",
])
def test_mixed_controls_whitespace_and_second_pass_preserve_original_rules(text):
    assert normalize_text(text) == original_normalize_text(text)
    assert normalize_text(normalize_text(text)) == original_normalize_text(original_normalize_text(text))


def test_control_removal_can_require_second_nfc_and_is_not_collapsed():
    # 第一次 NFC 时控制符隔开 e 与音标；删除后第二轮才可以组合，不能优化掉二次调用。
    first = normalize_text("e\x00\u0301")
    assert first == "e\u0301"
    assert normalize_text(first) == "é"


def test_seeded_mixed_unicode_text_and_repeated_cleaning_match():
    random_source = random.Random(42)
    alphabet = list(range(128)) + [0x300, 0x301, 0xA0, 0x200B, 0x2028, 0x2029,
                                 0xD800, 0xDFFF, 0xE000, 0x4E2D, 0x6587, 0x1F30F, 0x10FFFF]
    for _ in range(1000):
        text = "".join(chr(random_source.choice(alphabet) if random_source.random() < .9 else random_source.randrange(0x110000))
                       for _ in range(200))
        actual, expected = normalize_text(text), original_normalize_text(text)
        assert actual == expected
        assert normalize_text(actual) == original_normalize_text(expected)


@pytest.mark.parametrize("value", [None, 42, [], {}])
def test_non_string_error_is_unchanged(value):
    with pytest.raises(ValueError, match="text 必须是字符串"):
        normalize_text(value)


def test_pattern_build_is_lazy_and_cached(monkeypatch):
    _category_c_pattern.cache_clear()
    real_category = unicodedata.category
    calls = 0
    def counted(character):
        nonlocal calls
        calls += 1
        return real_category(character)
    monkeypatch.setattr("llm_lab.data.common.unicodedata.category", counted)
    assert _category_c_pattern.cache_info().currsize == 0
    normalize_text("first document")
    first_calls = calls
    assert first_calls > 1000000
    normalize_text("第二篇文档")
    assert calls == first_calls
    assert _category_c_pattern.cache_info().currsize == 1
