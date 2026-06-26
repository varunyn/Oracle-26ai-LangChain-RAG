from __future__ import annotations

from src.rag_agent.runtime.chat_service import _references_from_result


def merge_references(mode: str, result: dict[str, object]) -> dict[str, object]:
    references = _references_from_result(
        result,
        include_empty_core=mode in {"rag", "mixed"},
        include_empty_mcp_tools=mode in {"mcp", "mixed"},
    )
    references["mode"] = mode
    return references
