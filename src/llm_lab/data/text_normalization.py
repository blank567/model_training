"""可审计的中文繁转简边界；NFC/清洗由调用者在转换前完成。

只支持官方 OpenCC C++ Python 绑定，不回退到同名纯 Python 重实现。
构造时加载一次转换器与词典；逐文档使用已加载的 C++ 实例做有界固定点迭代。
某些词组需要前一轮先转换字形，下一轮才能命中词典，因此单次 t2s 并不总是幂等。
"""

from __future__ import annotations

import copy
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
from typing import Any

from llm_lab.common.io import canonical_hash, sha256_file


OPENCC_VERSION = "1.4.2"
T2S_MAX_PASSES = 8


class ChineseNormalizer:
    def __init__(self, mode: str, converter: Any, provenance: dict[str, Any]):
        self.mode = mode
        self._converter = converter
        self._provenance = provenance

    @property
    def provenance(self) -> dict[str, Any]:
        # 返回副本，防止下游补充 manifest 字段时意外改写本实例的指纹。
        return copy.deepcopy(self._provenance)

    def convert(self, text: str, language: str = "zh") -> str:
        if not isinstance(text, str):
            raise TypeError("中文转换输入必须为 str")
        if language not in {"en", "zh"}:
            raise ValueError("language 必须为 en 或 zh")
        if self.mode == "none" or language == "en" or not text:
            return text
        current = text
        # 只保存状态摘要和相邻两份文本，避免将大文档的八个版本同时留在内存。
        seen = {hashlib.sha256(current.encode("utf-8")).digest()}
        for pass_number in range(1, T2S_MAX_PASSES + 1):
            converted = self._converter.convert(current)
            if not isinstance(converted, str):
                raise RuntimeError("官方 OpenCC 返回类型异常，拒绝继续制备")
            if converted == current:
                return converted
            digest = hashlib.sha256(converted.encode("utf-8")).digest()
            if digest in seen:
                raise RuntimeError(f"OpenCC t2s 固定点转换出现循环（第 {pass_number} 次转换），拒绝继续制备")
            seen.add(digest)
            current = converted
        # 第八次结果仍在变化，不能把尚未确认稳定的输出用于 hash、分区或编码。
        raise RuntimeError(f"OpenCC t2s 在最多 {T2S_MAX_PASSES} 次转换内未收敛到固定点，拒绝继续制备")

    def normalize(self, text: str, language: str = "zh") -> str:
        """供数据制备调用的同义入口；本模块不额外施加 NFC 或修改英文。"""
        return self.convert(text, language)


def _dictionary_files(value: Any) -> set[str]:
    """遍历 config 的 normalization/conversion_chain/group，包含间接字典节点。"""
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "file":
                if not isinstance(child, str):
                    raise ValueError("OpenCC config 的 file 必须为字符串")
                result.add(child)
            else:
                result.update(_dictionary_files(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_dictionary_files(child))
    return result


def _resource_provenance(config: Path) -> list[dict[str, Any]]:
    directory = config.parent.resolve()
    parsed = json.loads(config.read_text(encoding="utf-8"))
    references = _dictionary_files(parsed)
    if not references:
        raise RuntimeError("官方 t2s.json 未声明词典，拒绝生成不可审计指纹")
    resources = []
    for name in sorted({config.name, *references}):
        relative = Path(name)
        resource = (directory / relative).resolve()
        if relative.is_absolute() or not resource.is_relative_to(directory) or not resource.is_file():
            raise RuntimeError("OpenCC config/dictionary 资源缺失或越出安装资源目录")
        resources.append({"path": relative.as_posix(), "sha256": sha256_file(resource),
                          "size_bytes": resource.stat().st_size})
    return resources


def build_chinese_normalizer(mode: str | None = "none") -> ChineseNormalizer:
    """none 完全不加载 OpenCC；t2s 在任何大规模读写前 fail fast。"""
    mode = "none" if mode is None else mode
    if not isinstance(mode, str) or mode not in {"none", "t2s"}:
        raise ValueError("中文转换 mode 仅支持 none/t2s")
    base = {"schema_version": 2, "mode": mode, "implementation": "identity",
            "package": None, "package_version": None, "native_version": None,
            "config": None, "resources": [], "resource_fingerprint": None,
            "native_extension_sha256": None, "include_tofu_risk_dictionaries": None,
            "normalization_algorithm": "identity", "max_passes": 0, "cycle_detection": None}
    if mode == "none":
        return ChineseNormalizer(mode, None, base)
    try:
        package_version = metadata.version("OpenCC")
        module = importlib.import_module("opencc")
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError(f"t2s 需要官方 C++ 绑定 OpenCC=={OPENCC_VERSION}；请安装 requirements/remote.txt") from exc
    if package_version != OPENCC_VERSION:
        raise RuntimeError(f"OpenCC 版本必须为 {OPENCC_VERSION}，实际为 {package_version}；不能静默改变语料规范")
    native = getattr(module, "opencc_clib", None)
    native_type = getattr(native, "_OpenCC", None)
    converter_type = getattr(module, "OpenCC", None)
    extension = Path(getattr(native, "__file__", ""))
    if (not isinstance(native_type, type) or not isinstance(converter_type, type)
            or not issubclass(converter_type, native_type) or not extension.is_file()
            or extension.suffix.lower() not in {".so", ".pyd"}):
        raise RuntimeError("检测到非官方 C++ OpenCC 模块或损坏绑定；拒绝纯 Python 回退")
    native_version = str(getattr(native, "__version__", ""))
    if native_version != OPENCC_VERSION:
        raise RuntimeError("OpenCC Python distribution 与 C++ 绑定版本不一致")
    # 固定安装包内绝对路径，避免当前工作目录的同名 t2s.json 抢先覆盖配置。
    config = Path(module.__file__).resolve().parent / "clib" / "share" / "opencc" / "t2s.json"
    if not config.is_file():
        raise RuntimeError("官方 OpenCC 安装缺少 clib/share/opencc/t2s.json")
    resources = _resource_provenance(config)
    converter = converter_type(str(config), include_tofu_risk_dictionaries=True)
    base.update(implementation="opencc_cpp", package="OpenCC", package_version=package_version,
                native_version=native_version, config="t2s.json", resources=resources,
                resource_fingerprint=canonical_hash(resources), native_extension_sha256=sha256_file(extension),
                include_tofu_risk_dictionaries=True, normalization_algorithm="bounded_fixed_point",
                max_passes=T2S_MAX_PASSES, cycle_detection="sha256_state")
    return ChineseNormalizer(mode, converter, base)
