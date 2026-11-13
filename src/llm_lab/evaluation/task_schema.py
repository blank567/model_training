"""训练与评测共用的严格 MASSIVE JSON 契约，只有标准库依赖。"""

from __future__ import annotations

import json
from typing import Any


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段: {key}")
        result[key] = value
    return result


def parse_task_json(text: str) -> dict[str, Any]:
    """拒绝代码围栏、额外解释、重复 key 和额外字段。"""
    if not isinstance(text, str):
        raise ValueError("任务输出必须是 JSON 字符串")
    obj = json.loads(text, object_pairs_hook=unique_json_object)
    if not isinstance(obj, dict) or set(obj) != {"intent", "slots"}:
        raise ValueError("输出字段必须恰好为 intent 和 slots")
    if not isinstance(obj["intent"], str) or not obj["intent"].strip():
        raise ValueError("intent 必须为非空字符串")
    if not isinstance(obj["slots"], list):
        raise ValueError("slots 必须为列表")
    for item in obj["slots"]:
        if not isinstance(item, dict) or set(item) != {"slot", "value"}:
            raise ValueError("每个槽位字段必须恰好为 slot 和 value")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in ("slot", "value")):
            raise ValueError("slot/value 必须为非空字符串")
    return obj
