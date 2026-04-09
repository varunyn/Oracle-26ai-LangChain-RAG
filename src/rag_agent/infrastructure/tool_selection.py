from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from api.settings import get_settings

from .direct_mcp_tools import get_mcp_tool_metadata, get_mcp_tool_metadata_async
from .oci_models import get_embedding_model


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


def _schema_terms(input_schema: Mapping[str, Any]) -> list[str]:
    terms: list[str] = []
    properties = input_schema.get("properties")
    if isinstance(properties, Mapping):
        for key, value in properties.items():
            terms.append(str(key))
            if isinstance(value, Mapping):
                title = value.get("title")
                description = value.get("description")
                if isinstance(title, str) and title.strip():
                    terms.append(title.strip())
                if isinstance(description, str) and description.strip():
                    terms.append(description.strip())
    required = input_schema.get("required")
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        for item in required:
            if isinstance(item, str) and item.strip():
                terms.append(item.strip())
    return terms


def _descriptor_text(descriptor: ToolDescriptor) -> str:
    parts = [
        descriptor["canonical_name"],
        descriptor["tool_name"],
        descriptor["server_key"],
        descriptor["description"],
        *_schema_terms(descriptor["input_schema"]),
    ]
    return "\n".join(part for part in parts if part)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _rank_descriptors(question: str, descriptors: list[ToolDescriptor]) -> list[ToolDescriptor]:
    if not question.strip() or not descriptors:
        return []

    embeddings = get_embedding_model(get_settings().EMBED_MODEL_TYPE)
    question_embedding = embeddings.embed_query(question)
    scored: list[tuple[float, ToolDescriptor]] = []
    threshold = 0.25

    for descriptor in descriptors:
        descriptor_embedding = embeddings.embed_query(_descriptor_text(descriptor))
        score = _cosine_similarity(question_embedding, descriptor_embedding)
        if score >= threshold:
            scored.append((score, descriptor))

    scored.sort(key=lambda item: (item[0], item[1]["canonical_name"]), reverse=True)
    return [descriptor for _, descriptor in scored]


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

    ranked = _rank_descriptors(question, descriptors)
    selected = _apply_limit(ranked, limit)

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

    ranked = _rank_descriptors(question, descriptors)
    selected = _apply_limit(ranked, limit)

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
