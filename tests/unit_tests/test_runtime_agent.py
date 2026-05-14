from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

from src.rag_agent.runtime.agent import RuntimeAgent
from src.rag_agent.runtime.chat_service import ChatRuntimeService


def test_runtime_agent_normalize_messages_accepts_langchain_types() -> None:
    messages = RuntimeAgent.normalize_messages(
        [
            {"type": "human", "content": "hello"},
            {"type": "ai", "content": "hi"},
            {"type": "system", "content": "rules"},
        ],
        None,
    )

    assert [m.role for m in messages] == ["user", "assistant", "system"]
    assert [m.content for m in messages] == ["hello", "hi", "rules"]


def test_runtime_agent_normalize_messages_falls_back_to_message_field() -> None:
    messages = RuntimeAgent.normalize_messages(None, "fallback")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "fallback"


def test_chat_runtime_service_exposes_v3_event_stream(monkeypatch) -> None:
    async def fake_stream_runtime_events(
        self: ChatRuntimeService,
        **kwargs: object,
    ) -> AsyncIterator[dict[str, object]]:
        _ = self
        _ = kwargs
        yield {"type": "tool_event", "data": {"phase": "start", "tool_name": "lookup"}}
        yield {"type": "text", "delta": "Hello"}
        yield {"type": "references", "data": {"standalone_question": "Hello?"}}

    monkeypatch.setattr(ChatRuntimeService, "_stream_runtime_events", fake_stream_runtime_events)
    service = ChatRuntimeService()

    async def run() -> list[dict[str, object]]:
        return [
            event
            async for event in service.astream_events(
                {"messages": [{"role": "user", "content": "Hello"}]},
                config={"configurable": {"thread_id": "thread-v3"}},
                version="v3",
            )
        ]

    events = asyncio.run(run())

    assert events[0]["method"] == "tool_calls"
    assert events[1]["method"] == "messages"
    message_params = cast(dict[str, object], events[1]["params"])
    message_data = cast(tuple[dict[str, object], dict[str, object]], message_params["data"])
    message_delta = cast(dict[str, object], message_data[0]["delta"])
    assert message_delta["text"] == "Hello"
    assert events[2]["method"] == "custom"
    custom_params = cast(dict[str, object], events[2]["params"])
    custom_data = cast(dict[str, object], custom_params["data"])
    assert custom_data["type"] == "references"
