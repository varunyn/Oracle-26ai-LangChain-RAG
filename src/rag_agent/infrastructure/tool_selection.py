from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from .direct_mcp_tools import get_mcp_tool_metadata, get_mcp_tool_metadata_async


class ToolDescriptor(TypedDict):
    canonical_name: str
    tool_name: str
    server_key: str
    description: str
    input_schema: dict[str, Any]


class ToolSelectionFailure(TypedDict):
    stage: str
    error_type: str
    message: str


class ToolSelectionResult(TypedDict):
    question: str
    limit: int | None
    selected_tools: list[ToolDescriptor]
    total_tools: int
    selection_failed: bool
    failure: ToolSelectionFailure | None


def _normalize_descriptor(raw: Mapping[str, Any]) -> ToolDescriptor | None:
    canonical_name = raw.get("canonical_name")
    if not isinstance(canonical_name, str) or not canonical_name.strip():
        return None

    server_key = raw.get("server_key")
    tool_name = raw.get("tool_name")
    description = raw.get("description")
    input_schema = raw.get("input_schema")

    if not isinstance(server_key, str):
        server_key = ""
    if not isinstance(tool_name, str):
        tool_name = ""
    if not isinstance(description, str):
        description = ""
    if not isinstance(input_schema, Mapping):
        input_schema = {}

    return {
        "canonical_name": canonical_name.strip(),
        "tool_name": tool_name.strip(),
        "server_key": server_key.strip(),
        "description": description.strip(),
        "input_schema": dict(cast(Mapping[str, Any], input_schema)),
    }


def _build_failure(exc: Exception) -> ToolSelectionFailure:
    return {
        "stage": "metadata_load",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _apply_limit(descriptors: list[ToolDescriptor], limit: int | None) -> list[ToolDescriptor]:
    if limit is None:
        return descriptors
    if limit <= 0:
        return []
    return descriptors[:limit]


def _build_result(
    *,
    question: str,
    limit: int | None,
    selected_tools: list[ToolDescriptor],
    total_tools: int,
    selection_failed: bool,
    failure: ToolSelectionFailure | None,
) -> ToolSelectionResult:
    return {
        "question": question,
        "limit": limit,
        "selected_tools": selected_tools,
        "total_tools": total_tools,
        "selection_failed": selection_failed,
        "failure": failure,
    }


async def select_mcp_tools_for_question_async(
    question: str,
    *,
    limit: int | None = None,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> ToolSelectionResult:
    try:
        metadata_list = await get_mcp_tool_metadata_async(
            server_keys=server_keys,
            run_config=run_config,
        )
    except Exception as exc:  # noqa: BLE001
        return _build_result(
            question=question,
            limit=limit,
            selected_tools=[],
            total_tools=0,
            selection_failed=True,
            failure=_build_failure(exc),
        )

    descriptors: list[ToolDescriptor] = []
    for raw in metadata_list:
        if not isinstance(raw, Mapping):
            continue
        descriptor = _normalize_descriptor(cast(Mapping[str, Any], raw))
        if descriptor is None:
            continue
        descriptors.append(descriptor)

    descriptors.sort(key=lambda item: item["canonical_name"])
    selected = _apply_limit(descriptors, limit)

    return _build_result(
        question=question,
        limit=limit,
        selected_tools=selected,
        total_tools=len(descriptors),
        selection_failed=False,
        failure=None,
    )


def select_mcp_tools_for_question(
    question: str,
    *,
    limit: int | None = None,
    server_keys: Sequence[str] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> ToolSelectionResult:
    try:
        metadata_list = get_mcp_tool_metadata(server_keys=server_keys, run_config=run_config)
    except Exception as exc:  # noqa: BLE001
        return _build_result(
            question=question,
            limit=limit,
            selected_tools=[],
            total_tools=0,
            selection_failed=True,
            failure=_build_failure(exc),
        )

    descriptors: list[ToolDescriptor] = []
    for raw in metadata_list:
        if not isinstance(raw, Mapping):
            continue
        descriptor = _normalize_descriptor(cast(Mapping[str, Any], raw))
        if descriptor is None:
            continue
        descriptors.append(descriptor)

    descriptors.sort(key=lambda item: item["canonical_name"])
    selected = _apply_limit(descriptors, limit)

    return _build_result(
        question=question,
        limit=limit,
        selected_tools=selected,
        total_tools=len(descriptors),
        selection_failed=False,
        failure=None,
    )


__all__ = [
    "ToolDescriptor",
    "ToolSelectionFailure",
    "ToolSelectionResult",
    "select_mcp_tools_for_question",
    "select_mcp_tools_for_question_async",
]
