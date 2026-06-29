from __future__ import annotations

import asyncio
from typing import cast

import pytest
from langchain_core.documents import Document

from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.agent import normalize_messages
from src.rag_agent.runtime.chat_service import ChatRuntimeService


def test_runtime_agent_normalize_messages_accepts_langchain_types() -> None:
    messages = normalize_messages(
        [
            {"id": "client-user-1", "type": "human", "content": "hello"},
            {"type": "ai", "content": "hi"},
            {"type": "system", "content": "rules"},
        ],
        None,
    )

    assert messages[0].id == "client-user-1"
    assert [m.role for m in messages] == ["user", "assistant", "system"]
    assert [m.content for m in messages] == ["hello", "hi", "rules"]


def test_runtime_agent_normalize_messages_falls_back_to_message_field() -> None:
    messages = normalize_messages(None, "fallback")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "fallback"


def test_chat_runtime_service_exposes_v3_event_stream(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_run_chat(
        self: ChatRuntimeService,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = self
        calls.append(kwargs)
        progress = cast(
            object,
            kwargs.get("tool_progress_callback"),
        )
        if callable(progress):
            progress({"phase": "start", "tool_name": "lookup"})
        return {
            "final_answer": "Hello",
            "standalone_question": "Hello?",
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": True,
            "mcp_tools_used": ["lookup"],
        }

    monkeypatch.setattr(ChatRuntimeService, "run_chat", fake_run_chat)
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

    assert calls[0]["stream"] is True
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


def test_chat_runtime_service_streams_rag_answer_deltas(monkeypatch) -> None:
    docs = [
        Document(
            page_content="Northway Solutions pays invoices within 45 days.",
            metadata={"source": "northway.md"},
        )
    ]
    stream_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        rag_runtime,
        "retrieve_oracle_docs",
        lambda **kwargs: docs,
    )
    monkeypatch.setattr(
        rag_runtime,
        "rerank_retrieved_docs",
        lambda query, docs, *, enable_reranker: docs,
    )

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        _ = kwargs
        return "First second", None, "fake-model"

    async def fake_stream_rag_answer(**kwargs: object):
        stream_calls.append(kwargs)
        yield "First ", object(), "fake-stream-model"
        yield "second", object(), "fake-stream-model"

    monkeypatch.setattr(rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)
    monkeypatch.setattr(
        rag_runtime,
        "stream_rag_answer",
        fake_stream_rag_answer,
        raising=False,
    )

    service = ChatRuntimeService()

    async def run() -> list[dict[str, object]]:
        return [
            event
            async for event in service.astream_events(
                {"messages": [{"role": "user", "content": "What are the payment terms?"}]},
                config={
                    "configurable": {
                        "thread_id": "thread-rag-stream",
                        "collection_name": "oracle_web_embeddings",
                        "mode": "rag",
                    }
                },
                version="v3",
            )
        ]

    events = asyncio.run(run())

    text_deltas: list[str] = []
    for event in events:
        if event["method"] != "messages":
            continue
        params = cast(dict[str, object], event["params"])
        message_data = cast(tuple[dict[str, object], dict[str, object]], params["data"])
        message_delta = cast(dict[str, object], message_data[0]["delta"])
        text_deltas.append(str(message_delta["text"]))

    assert text_deltas == ["First ", "second"]
    assert stream_calls
    assert stream_calls[0]["docs"] == docs
    assert stream_calls[0]["question"] == "What are the payment terms?"

    state = service.get_state_values({"configurable": {"thread_id": "thread-rag-stream"}})
    assert state is not None
    assert state["final_answer"] == "First second"


@pytest.mark.parametrize("mode", ["mcp", "mixed"])
def test_chat_runtime_service_rejects_langgraph_owned_stream_modes(mode: str) -> None:
    service = ChatRuntimeService()

    async def run() -> list[dict[str, object]]:
        return [
            event
            async for event in service.astream_events(
                {"messages": [{"role": "user", "content": "Hello"}]},
                config={
                    "configurable": {
                        "thread_id": f"thread-{mode}-stream",
                        "collection_name": "oracle_web_embeddings",
                        "mode": mode,
                    }
                },
                version="v3",
            )
        ]

    with pytest.raises(NotImplementedError, match="LangGraph"):
        asyncio.run(run())
