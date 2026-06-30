from __future__ import annotations

import ast
import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from src.rag_agent.infrastructure.oci_models import get_llm

from .llm_invocation import invoke_llm_with_optional_config, message_text, suppress_llm_streaming


def _content_text(content: object) -> str:
    content = _structured_content(content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            block_type = str(item.get("type") or "")
            if block_type in {"text", "text_delta"}:
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content or "")


def _structured_content(content: object) -> object:
    if not isinstance(content, str):
        return content
    text = content.strip()
    if not text.startswith("[") or "\"type\"" not in text and "'type'" not in text:
        return content
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return content
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    return content


def _message_role(message: object) -> str | None:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, dict):
        role = str(message.get("role") or "").strip().lower()
        if role in {"user", "human"}:
            return "user"
        if role in {"assistant", "ai"}:
            return "assistant"
        if role == "system":
            return "system"
        message_type = str(message.get("type") or "").strip().lower()
        if message_type in {"user", "human"}:
            return "user"
        if message_type in {"assistant", "ai"}:
            return "assistant"
        if message_type == "system":
            return "system"
    return None


def _message_id(message: object) -> str | None:
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        message_id = getattr(message, "id", None)
        if isinstance(message_id, str) and message_id.strip():
            return message_id.strip()
    if isinstance(message, dict):
        message_id = message.get("id")
        if isinstance(message_id, str) and message_id.strip():
            return message_id.strip()
    return None


def _message_content(message: object) -> object:
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        return _structured_content(message.content or "")
    if isinstance(message, dict):
        return _structured_content(message.get("content") or "")
    return ""


def _langchain_content(content: object) -> str | list[object]:
    if isinstance(content, list):
        return content
    return _content_text(content)


def to_langchain_messages(messages: list[dict[str, object]]) -> list[Any]:
    converted: list[Any] = []
    for item in messages:
        role = _message_role(item)
        content = _langchain_content(item.get("content") or "")
        message_id = _message_id(item)
        if role == "user":
            converted.append(HumanMessage(content=content, id=message_id))
        elif role == "assistant":
            converted.append(AIMessage(content=content, id=message_id))
        elif role == "system":
            converted.append(SystemMessage(content=content, id=message_id))
    return converted


def hydrate_thread_messages(
    thread_state: dict[str, Any],
    incoming_messages: list[dict[str, object]],
) -> list[dict[str, object]]:
    prior_messages = list(thread_state.get("messages") or [])
    if not prior_messages:
        return list(incoming_messages)

    incoming_lc = to_langchain_messages(incoming_messages)
    prior_lc = [message for message in prior_messages if message_signature(message)]
    if not prior_lc:
        return list(incoming_messages)
    if messages_are_prefix(prior_lc, incoming_lc):
        return list(incoming_messages)
    return [*langchain_messages_to_dicts(prior_lc), *incoming_messages]


def latest_user_message(messages: list[dict[str, object]]) -> str:
    latest = ""
    for item in messages:
        content = _content_text(item.get("content") or "")
        if _message_role(item) == "user":
            latest = content.strip() or latest
    return latest


def chat_history_before_latest_user(messages: list[dict[str, object]]) -> list[Any]:
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if _message_role(messages[idx]) == "user":
            last_user_idx = idx
            break
    if last_user_idx <= 0:
        return []
    return to_langchain_messages(messages[:last_user_idx])


async def contextualize_question(
    *,
    question: str,
    chat_history: list[Any],
    model_id: str | None,
    run_config: RunnableConfig | None,
) -> str:
    if not question.strip() or not chat_history:
        return question

    transcript = "\n".join(
        f"{message_role_label(message)}: {getattr(message, 'content', '')}"
        for message in chat_history
        if str(getattr(message, "content", "")).strip()
    )
    if not transcript.strip():
        return question

    prompt = (
        "Rewrite the latest user question into a standalone question using the "
        "conversation history. Preserve specific entities, constraints, and intent. "
        "Return only the standalone question.\n\n"
        f"Conversation history:\n{transcript}\n\n"
        f"Latest user question:\n{question}"
    )
    llm = suppress_llm_streaming(get_llm(model_id=model_id))
    response = await asyncio.to_thread(
        invoke_llm_with_optional_config,
        llm,
        [HumanMessage(content=prompt)],
        run_config,
    )
    standalone = message_text(response).strip()
    return standalone or question


def messages_are_prefix(prefix: Sequence[object], messages: Sequence[object]) -> bool:
    if len(prefix) > len(messages):
        return False
    for expected, actual in zip(prefix, messages, strict=False):
        if message_signature(expected) != message_signature(actual):
            return False
    return True


def new_incoming_messages(
    prior_messages: Sequence[object],
    incoming_messages: Sequence[object],
) -> list[Any]:
    if not prior_messages:
        return list(incoming_messages)
    if not incoming_messages:
        return []
    prior_len = len(prior_messages)
    if messages_are_prefix(prior_messages, incoming_messages):
        return list(incoming_messages[prior_len:])
    return list(incoming_messages)


def merge_chat_messages(left: Sequence[object], right: Sequence[object]) -> list[object]:
    left_messages = list(left or [])
    right_messages = list(right or [])
    if not left_messages:
        return right_messages
    if not right_messages:
        return left_messages
    if messages_are_prefix(left_messages, right_messages):
        return right_messages
    return [*left_messages, *new_incoming_messages(left_messages, right_messages)]


def message_signature(message: object) -> tuple[str, str] | None:
    role = _message_role(message)
    if role is None:
        return None
    message_id = _message_id(message)
    if message_id is not None:
        return (role, f"id:{message_id}")
    return (role, _content_text(_message_content(message)))


def message_role_label(message: object) -> str:
    return _message_role(message) or "message"


def langchain_messages_to_dicts(messages: Sequence[object]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for message in messages:
        role = _message_role(message)
        if role is None:
            continue
        payload: dict[str, object] = {"role": role, "content": _message_content(message)}
        message_id = _message_id(message)
        if message_id is not None:
            payload["id"] = message_id
        serialized.append(payload)
    return serialized


__all__ = [
    "chat_history_before_latest_user",
    "contextualize_question",
    "hydrate_thread_messages",
    "langchain_messages_to_dicts",
    "latest_user_message",
    "message_role_label",
    "message_signature",
    "merge_chat_messages",
    "messages_are_prefix",
    "new_incoming_messages",
    "to_langchain_messages",
]
