"""中文转换单元测试不安装依赖；真实 C++ 绑定测试可在远程 CPU 环境补跑。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_lab.data import text_normalization as normalization


@pytest.fixture
def official_stub(tmp_path, monkeypatch):
    package = tmp_path / "opencc"
    resources = package / "clib/share/opencc"
    resources.mkdir(parents=True)
    config = {"normalization": [{"dict": {"file": "compat.ocd2"}}], "conversion_chain": [
        {"dict": {"type": "group", "dicts": [{"file": "phrases.ocd2"}, {"file": "chars.ocd2"}]}}]}
    (resources / "t2s.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("compat.ocd2", "phrases.ocd2", "chars.ocd2"):
        (resources / name).write_bytes(name.encode())
    extension = package / "clib/opencc_clib.so"
    extension.write_bytes(b"native binary fixture")
    calls = []

    class Native:
        pass

    class Converter(Native):
        def __init__(self, path, **options):
            calls.append((path, options))

        def convert(self, text):
            calls.append(text)
            return text.replace("繁體", "繁体").replace("漢語", "汉语")

    module = SimpleNamespace(__file__=str(package / "__init__.py"), OpenCC=Converter,
        opencc_clib=SimpleNamespace(_OpenCC=Native, __file__=str(extension), __version__=normalization.OPENCC_VERSION))
    monkeypatch.setattr(normalization.metadata, "version", lambda name: normalization.OPENCC_VERSION)
    monkeypatch.setattr(normalization.importlib, "import_module", lambda name: module)
    return module, resources, calls


@pytest.mark.parametrize("mode", [None, "none"])
def test_none_does_not_import_or_query_optional_dependency(monkeypatch, mode):
    monkeypatch.setattr(normalization.importlib, "import_module", lambda name: pytest.fail("must remain lazy"))
    monkeypatch.setattr(normalization.metadata, "version", lambda name: pytest.fail("must remain lazy"))
    normalizer = normalization.build_chinese_normalizer(mode)
    text = "繁體 e\u0301 🌏"
    assert normalizer.normalize(text, language="zh") == text
    assert normalizer.convert(text, language="en") == text
    assert normalizer.provenance["implementation"] == "identity"


def test_cpp_conversion_only_applies_to_chinese_and_preserves_english_exactly(official_stub):
    _, resources, calls = official_stub
    normalizer = normalization.build_chinese_normalizer("t2s")
    assert calls[0][0] == str(resources / "t2s.json")
    assert calls[0][1] == {"include_tofu_risk_dictionaries": True}
    assert normalizer.normalize("繁體漢語 🌏", "zh") == "繁体汉语 🌏"
    text = "English with 繁體 and e\u0301\r\n"
    assert normalizer.convert(text, "en") == text
    assert calls[1:] == ["繁體漢語 🌏", "繁体汉语 🌏"]
    assert normalizer.convert("", "zh") == ""


def test_provenance_tracks_config_nested_dictionaries_and_native_binary(official_stub):
    _, resources, _ = official_stub
    first = normalization.build_chinese_normalizer("t2s")
    provenance = first.provenance
    assert provenance["package"] == "OpenCC" and provenance["package_version"] == normalization.OPENCC_VERSION
    assert provenance["schema_version"] == 2
    assert provenance["normalization_algorithm"] == "bounded_fixed_point"
    assert provenance["max_passes"] == 8 and provenance["cycle_detection"] == "sha256_state"
    assert {item["path"] for item in provenance["resources"]} == {"t2s.json", "compat.ocd2", "phrases.ocd2", "chars.ocd2"}
    assert len(provenance["native_extension_sha256"]) == 64
    json.dumps(provenance)  # manifest 可直接记录该对象。
    provenance["resources"].clear()
    assert len(first.provenance["resources"]) == 4
    (resources / "chars.ocd2").write_bytes(b"changed dictionary")
    second = normalization.build_chinese_normalizer("t2s")
    assert first.provenance["resource_fingerprint"] != second.provenance["resource_fingerprint"]


def test_missing_dependency_fails_at_build_before_document_processing(monkeypatch):
    def missing(name):
        raise normalization.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(normalization.metadata, "version", missing)
    with pytest.raises(RuntimeError, match=r"官方 C\+\+ 绑定"):
        normalization.build_chinese_normalizer("t2s")


def test_wrong_version_and_pure_python_namespace_collision_are_rejected(official_stub, monkeypatch):
    module, _, _ = official_stub
    monkeypatch.setattr(normalization.metadata, "version", lambda name: "1.0.0")
    with pytest.raises(RuntimeError, match="版本必须"):
        normalization.build_chinese_normalizer("t2s")
    monkeypatch.setattr(normalization.metadata, "version", lambda name: normalization.OPENCC_VERSION)
    module.opencc_clib = None
    with pytest.raises(RuntimeError, match="纯 Python"):
        normalization.build_chinese_normalizer("t2s")


def test_missing_or_escaping_dictionary_fails_fast(official_stub):
    _, resources, _ = official_stub
    (resources / "chars.ocd2").unlink()
    with pytest.raises(RuntimeError, match="资源缺失"):
        normalization.build_chinese_normalizer("t2s")
    (resources / "chars.ocd2").write_bytes(b"fixture")
    (resources / "t2s.json").write_text(json.dumps({"dict": {"file": "../../outside.ocd2"}}))
    with pytest.raises(RuntimeError, match="越出"):
        normalization.build_chinese_normalizer("t2s")


def test_invalid_modes_languages_and_input_types_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        normalization.build_chinese_normalizer("s2t")
    normalizer = normalization.build_chinese_normalizer()
    with pytest.raises(ValueError, match="language"):
        normalizer.convert("text", "ja")
    with pytest.raises(TypeError, match="str"):
        normalizer.convert(None)


def test_multistage_conversion_reaches_fixed_point_for_both_public_entries(official_stub):
    module, _, calls = official_stub
    # 真实 Wiki 中的“浮於潛”先变“浮於潜”，下一次才能匹配“浮於”。
    def staged(self, text):
        calls.append(text)
        return {"浮於潛": "浮於潜", "浮於潜": "浮于潜"}.get(text, text)
    module.OpenCC.convert = staged
    normalizer = normalization.build_chinese_normalizer("t2s")
    assert normalizer.convert("浮於潛") == "浮于潜"
    assert calls[1:] == ["浮於潛", "浮於潜", "浮于潜"]
    assert normalizer.normalize("浮於潛") == "浮于潜"
    assert normalizer.convert(normalizer.convert("浮於潛")) == "浮于潜"


def test_nonadjacent_cycle_is_rejected_instead_of_choosing_arbitrary_output(official_stub):
    module, _, calls = official_stub
    def cycle(self, text):
        calls.append(text)
        return {"甲": "乙", "乙": "丙", "丙": "甲"}[text]
    module.OpenCC.convert = cycle
    normalizer = normalization.build_chinese_normalizer("t2s")
    with pytest.raises(RuntimeError, match="固定点转换出现循环"):
        normalizer.convert("甲")
    assert calls[1:] == ["甲", "乙", "丙"]


@pytest.mark.parametrize("changes,converges", [(7, True), (8, False)])
def test_eighth_call_must_prove_stability_or_fail(official_stub, changes, converges):
    module, _, calls = official_stub
    def advance(self, text):
        calls.append(text)
        number = int(text)
        return str(min(number + 1, changes))
    module.OpenCC.convert = advance
    normalizer = normalization.build_chinese_normalizer("t2s")
    if converges:
        assert normalizer.convert("0") == "7"
    else:
        with pytest.raises(RuntimeError, match="最多 8 次转换内未收敛"):
            normalizer.normalize("0")
    assert len(calls) - 1 == 8


def test_non_string_at_later_pass_is_rejected(official_stub):
    module, _, _ = official_stub
    module.OpenCC.convert = lambda self, text: "changed" if text == "start" else None
    normalizer = normalization.build_chinese_normalizer("t2s")
    with pytest.raises(RuntimeError, match="返回类型异常"):
        normalizer.convert("start")


def test_real_official_cpp_opencc_t2s_and_fingerprint():
    pytest.importorskip("opencc", reason="remote CPU must run after installing pinned official OpenCC")
    normalizer = normalization.build_chinese_normalizer("t2s")
    assert normalizer.convert("繁體中文與漢語測試。", "zh") == "繁体中文与汉语测试。"
    assert normalizer.convert("English 繁體\n", "en") == "English 繁體\n"
    assert normalizer.provenance["implementation"] == "opencc_cpp"
    assert normalizer.provenance["resource_fingerprint"]
    # 此片段来自真实中文 Wikipedia source_id=...:zh:3105 的回归问题。
    # 保留底层单次结果断言，避免把生产策略误记为单次标准 t2s。
    single = normalizer._converter.convert("浮於潛")
    assert single == "浮於潜"
    assert normalizer._converter.convert(single) == "浮于潜"
    assert normalizer.normalize("浮於潛") == "浮于潜"
    assert normalizer.convert(normalizer.convert("浮於潛")) == "浮于潜"
    assert normalizer.provenance["normalization_algorithm"] == "bounded_fixed_point"
    assert all(Path(item["path"]).suffix in {".json", ".ocd2", ".ocd"} for item in normalizer.provenance["resources"])
