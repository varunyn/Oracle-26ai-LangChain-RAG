from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from src.rag_agent.infrastructure.oci_models import get_llm

from .llm_invocation import invoke_llm_with_optional_config


def to_langchain_messages(messages: list[dict[str, object]]) -> list[Any]:
    converted: list[Any] = []
    for item in messages:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "")
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
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
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "")
        if role == "user":
            latest = content.strip() or latest
    return latest


def chat_history_before_latest_user(messages: list[dict[str, object]]) -> list[Any]:
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        role = str(messages[idx].get("role") or "").strip().lower()
        if role == "user":
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
    llm = get_llm(model_id=model_id)
    response = await asyncio.to_thread(
        invoke_llm_with_optional_config,
        llm,
        [HumanMessage(content=prompt)],
        run_config,
    )
    standalone = str(getattr(response, "content", "") or "").strip()
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


def message_signature(message: object) -> tuple[str, str] | None:
    if isinstance(message, HumanMessage):
        return ("user", str(message.content or ""))
    if isinstance(message, AIMessage):
        return ("assistant", str(message.content or ""))
    if isinstance(message, SystemMessage):
        return ("system", str(message.content or ""))
    if isinstance(message, dict):
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "")
        if role in {"user", "assistant", "system"}:
            return (role, content)
    return None


def message_role_label(message: object) -> str:
    signature = message_signature(message)
    return signature[0] if signature else "message"


def langchain_messages_to_dicts(messages: Sequence[object]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for message in messages:
        signature = message_signature(message)
        if signature is None:
            continue
        role, content = signature
        serialized.append({"role": role, "content": content})
    return serialized


__all__ = [
    "chat_history_before_latest_user",
    "contextualize_question",
    "hydrate_thread_messages",
    "langchain_messages_to_dicts",
    "latest_user_message",
    "message_role_label",
    "message_signature",
    "messages_are_prefix",
    "new_incoming_messages",
    "to_langchain_messages",
]
