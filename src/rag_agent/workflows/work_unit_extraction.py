from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .generic_state import WorkUnit


def extract_work_units_from_tool_invocations(
    tool_invocations: Sequence[Mapping[str, object]],
) -> list[WorkUnit]:
    """Find the first multi-item work queue returned by tools."""

    for invocation in tool_invocations:
        parsed = _parse_tool_result(invocation.get("result"))
        queue = _find_multi_item_list(parsed)
        if queue:
            return [_to_work_unit(item) for item in queue]
    return []


def _parse_tool_result(raw: object) -> object:
    if not isinstance(raw, str) or not raw.strip():
        return raw
    text = raw.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:  # noqa: BLE001
            continue
        return _unwrap_content_blocks(parsed)
    return raw


def _unwrap_content_blocks(value: object) -> object:
    if isinstance(value, list):
        unwrapped: list[object] = []
        for item in value:
            if isinstance(item, Mapping) and item.get("type") == "text" and "text" in item:
                unwrapped.append(_parse_tool_result(item.get("text")))
            else:
                unwrapped.append(_unwrap_content_blocks(item))
        return unwrapped[0] if len(unwrapped) == 1 else unwrapped
    if isinstance(value, Mapping):
        return {str(k): _unwrap_content_blocks(v) for k, v in value.items()}
    return value


def _find_multi_item_list(value: object, *, depth: int = 0) -> list[dict[str, Any]]:
    if isinstance(value, list):
        items = [dict(item) for item in value if isinstance(item, Mapping)]
        return items if len(items) > 1 else []
    if not isinstance(value, Mapping):
        return []

    immediate_candidates = [
        [dict(item) for item in nested if isinstance(item, Mapping)]
        for nested in value.values()
        if isinstance(nested, list)
    ]
    immediate_candidates = [items for items in immediate_candidates if len(items) > 1]
    if len(immediate_candidates) == 1 and (depth == 0 or not _has_scalar_sibling(value)):
        return immediate_candidates[0]
    if len(immediate_candidates) > 1:
        return []

    nested_mappings = [nested for nested in value.values() if isinstance(nested, Mapping)]
    if len(nested_mappings) != 1:
        return []
    return _find_multi_item_list(nested_mappings[0], depth=depth + 1)


def _has_scalar_sibling(value: Mapping[str, object]) -> bool:
    return any(not isinstance(nested, list | Mapping) for nested in value.values())


def _to_work_unit(item: dict[str, Any]) -> WorkUnit:
    raw_identity = item.get("id") or item.get("ID") or item.get("name") or item.get("key")
    identity = str(raw_identity).strip() if raw_identity is not None else ""
    if not identity:
        dumped = json.dumps(item, sort_keys=True, default=str)
        identity = hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]
    return {"id": identity, "payload": item}
