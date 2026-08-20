from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

from langchain_core.messages import AIMessage, RemoveMessage

from src.rag_agent.graphs.runtime import references_from_result, result_to_assistant_message

logger = logging.getLogger(__name__)


def merge_references(mode: str, result: dict[str, object]) -> dict[str, object]:
    references = references_from_result(result, mode=mode)
    error = result.get("error")
    if isinstance(error, dict):
        references["error"] = error
    return cast(dict[str, object], references)


def assistant_message_from_result(
    mode: str, result: dict[str, object], *, message_id: str | None = None
) -> AIMessage:
    return result_to_assistant_message(mode, result, message_id=message_id)


def messages_from_result(
    mode: str,
    result: dict[str, object],
    state_messages: Sequence[object] | None,
    *,
    message_id: str | None = None,
) -> list[object]:
    messages = list(state_messages or [])
    if not messages:
        return [assistant_message_from_result(mode, result, message_id=message_id)]

    references = merge_references(mode, result)
    final_answer = _normalize_ai_content(result.get("final_answer"))
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, AIMessage):
            continue
        if final_answer and _normalize_ai_content(message.content) != final_answer:
            break
        updated_message = _copy_ai_message_with_references(
            message, references, message_id=message_id
        )
        if message_id and message.id and message.id != message_id:
            return [RemoveMessage(id=message.id), updated_message]
        else:
            messages[index] = updated_message
        return messages

    messages.append(assistant_message_from_result(mode, result, message_id=message_id))
    return messages


def assistant_message_from_exception(
    mode: str, exc: Exception, *, message_id: str | None = None
) -> AIMessage:
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
        message_id=message_id,
    )


def _copy_ai_message_with_references(
    message: AIMessage, references: dict[str, object], *, message_id: str | None = None
) -> AIMessage:
    additional_kwargs = {**dict(message.additional_kwargs), **references}
    response_metadata = {**dict(message.response_metadata), **references}
    copy = getattr(message, "model_copy", None)
    if callable(copy):
        return cast(
            AIMessage,
            copy(
                update={
                    "additional_kwargs": additional_kwargs,
                    "response_metadata": response_metadata,
                    "id": message_id or message.id,
                }
            ),
        )
    return AIMessage(
        content=message.content,
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata,
        tool_calls=list(message.tool_calls),
        id=message_id or message.id,
        name=message.name,
    )


def _normalize_ai_content(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            parts.append(str(item))
        return "".join(parts).strip()
    return str(value or "").strip()
