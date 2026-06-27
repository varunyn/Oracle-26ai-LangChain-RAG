from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from src.rag_agent.runtime.chat_service import _references_from_result

logger = logging.getLogger(__name__)


def merge_references(mode: str, result: dict[str, object]) -> dict[str, object]:
    references = _references_from_result(
        result,
        include_empty_core=mode in {"rag", "mixed"},
        include_empty_mcp_tools=mode in {"mcp", "mixed"},
    )
    error = result.get("error")
    if isinstance(error, dict):
        references["error"] = error
    references["mode"] = mode
    return references


def assistant_message_from_result(mode: str, result: dict[str, object]) -> AIMessage:
    final_answer = result.get("final_answer")
    content: str | list[Any]
    if isinstance(final_answer, str):
        content = final_answer
    elif isinstance(final_answer, list):
        content = final_answer
    elif final_answer is None:
        content = ""
    else:
        content = str(final_answer)
    return AIMessage(
        content=content,
        additional_kwargs=merge_references(mode, result),
    )


def assistant_message_from_exception(mode: str, exc: Exception) -> AIMessage:
    logger.exception("LangGraph %s node failed", mode)
    return assistant_message_from_result(
        mode,
        {
            "final_answer": (
                "I couldn't complete the request because the runtime "
                "backend returned an error. Please try again after the backend "
                "connection is healthy."
            ),
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        },
    )
