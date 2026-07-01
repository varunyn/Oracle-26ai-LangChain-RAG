from __future__ import annotations

import asyncio
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from src.rag_agent.infrastructure.oci_models import get_llm

from .llm_invocation import invoke_llm_with_optional_config, message_text, suppress_llm_streaming


def _content_text(content: object) -> str:
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


def _message_role(message: BaseMessage) -> str | None:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    return None


def _message_content(message: BaseMessage) -> object:
    return message.content or ""


def latest_user_message(messages: Sequence[BaseMessage]) -> str:
    latest = ""
    for item in messages:
        content = _content_text(_message_content(item))
        if _message_role(item) == "user":
            latest = content.strip() or latest
    return latest


def latest_user_message_id(messages: Sequence[BaseMessage]) -> str | None:
    for item in reversed(messages):
        if _message_role(item) != "user":
            continue
        message_id = getattr(item, "id", None)
        if isinstance(message_id, str) and message_id.strip():
            return message_id.strip()
    return None


def chat_history_before_latest_user(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if _message_role(messages[idx]) == "user":
            last_user_idx = idx
            break
    if last_user_idx <= 0:
        return []
    return list(messages[:last_user_idx])


async def contextualize_question(
    *,
    question: str,
    chat_history: Sequence[BaseMessage],
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


def message_role_label(message: BaseMessage) -> str:
    return _message_role(message) or "message"


__all__ = [
    "chat_history_before_latest_user",
    "contextualize_question",
    "latest_user_message",
    "latest_user_message_id",
    "message_role_label",
]
