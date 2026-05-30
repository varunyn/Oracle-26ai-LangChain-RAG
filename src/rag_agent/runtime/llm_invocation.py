from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from langchain_core.runnables.config import RunnableConfig


def invoke_llm_with_optional_config(
    llm: object,
    messages: Sequence[object],
    run_config: RunnableConfig | None,
) -> object:
    invoke = getattr(llm, "invoke")
    if run_config:
        try:
            return invoke(messages, config=run_config)
        except TypeError:
            return invoke(messages)
    return invoke(messages)


async def stream_llm_chunks_with_optional_config(
    llm: object,
    messages: Sequence[object],
    run_config: RunnableConfig | None,
) -> AsyncIterator[tuple[str, object]]:
    astream = getattr(llm, "astream", None)
    if not callable(astream):
        raise TypeError("Configured LLM does not support async streaming.")

    stream = astream(messages, config=run_config) if run_config else astream(messages)
    async for chunk in stream:
        yield _message_text(chunk), chunk


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = str(block.get("type") or "")
                if block_type in {"text", "text_delta"}:
                    parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts)

    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        return str(text_attr())
    if text_attr is not None:
        return str(cast(Any, text_attr))
    return ""


__all__ = ["invoke_llm_with_optional_config", "stream_llm_chunks_with_optional_config"]
