from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from langchain_core.runnables.config import RunnableConfig


def suppress_llm_streaming(llm: object) -> object:
    with_config = getattr(llm, "with_config", None)
    if not callable(with_config):
        return llm
    try:
        return with_config({"tags": ["nostream"]})
    except TypeError:
        try:
            return with_config(tags=["nostream"])
        except TypeError:
            return llm


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


async def ainvoke_llm_with_optional_config(
    llm: object,
    messages: Sequence[object],
    run_config: RunnableConfig | None,
) -> object:
    ainvoke = getattr(llm, "ainvoke", None)
    if callable(ainvoke):
        if run_config:
            try:
                return await ainvoke(messages, config=run_config)
            except TypeError:
                return await ainvoke(messages)
        return await ainvoke(messages)
    return await asyncio.to_thread(invoke_llm_with_optional_config, llm, messages, run_config)


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
        yield message_text(chunk), chunk


def message_text(message: object) -> str:
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


__all__ = [
    "ainvoke_llm_with_optional_config",
    "invoke_llm_with_optional_config",
    "message_text",
    "stream_llm_chunks_with_optional_config",
    "suppress_llm_streaming",
]
