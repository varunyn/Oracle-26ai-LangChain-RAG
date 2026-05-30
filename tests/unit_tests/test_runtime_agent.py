from __future__ import annotations

import asyncio
from typing import cast

from langchain_core.documents import Document

from src.rag_agent.runtime import chat_service, rag_runtime
from src.rag_agent.runtime.agent import normalize_messages
from src.rag_agent.runtime.chat_service import ChatRuntimeService
from src.rag_agent.runtime.mcp_turn import MCPAgentTurn


def test_runtime_agent_normalize_messages_accepts_langchain_types() -> None:
    messages = normalize_messages(
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


def test_chat_runtime_service_emits_final_mixed_answer_without_rag_docs(monkeypatch) -> None:
    mcp_answer_callbacks: list[object] = []

    async def fake_run_mcp_agent_turn(**kwargs: object) -> MCPAgentTurn:
        answer_delta_callback = kwargs.get("answer_delta_callback")
        mcp_answer_callbacks.append(answer_delta_callback)
        if callable(answer_delta_callback):
            answer_delta_callback("Mixed ")
            answer_delta_callback("answer")
        return MCPAgentTurn(
            answer="Mixed answer",
            tools_used=["calculator"],
            tool_invocations=[
                {
                    "tool_name": "calculator",
                    "args": {"query": "payment terms"},
                    "result": "Payment terms are net 30.",
                }
            ],
            resolved_model_id="fake-model",
        )

    monkeypatch.setattr(chat_service, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
    service = ChatRuntimeService()

    async def run() -> list[dict[str, object]]:
        return [
            event
            async for event in service.astream_events(
                {"messages": [{"role": "user", "content": "Payment terms?"}]},
                config={
                    "configurable": {
                        "thread_id": "thread-mixed-stream",
                        "collection_name": "oracle_web_embeddings",
                        "mode": "mixed",
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

    assert mcp_answer_callbacks == [None]
    assert text_deltas == ["Mixed answer"]
    state = service.get_state_values({"configurable": {"thread_id": "thread-mixed-stream"}})
    assert state is not None
    assert state["final_answer"] == "Mixed answer"


def test_chat_runtime_service_streams_mixed_rag_answer_from_retrieved_docs(monkeypatch) -> None:
    docs = [
        Document(
            page_content="Northway Solutions standard terms are Net 30.",
            metadata={"source": "northway.md"},
        )
    ]
    stream_calls: list[dict[str, object]] = []
    mcp_answer_callbacks: list[object] = []

    class FakeRetrievalTool:
        name = "oracle_retrieval"
        _retrieval_state = {"docs": docs}

    async def fake_run_mcp_agent_turn(**kwargs: object) -> MCPAgentTurn:
        mcp_answer_callbacks.append(kwargs.get("answer_delta_callback"))
        return MCPAgentTurn(
            answer="Buffered MCP answer should not stream.",
            tools_used=["oracle_retrieval", "Calculator_calculate"],
            tool_invocations=[
                {
                    "tool_name": "oracle_retrieval",
                    "args": {"query": "Northway Solutions payment terms"},
                    "result": "Northway Solutions standard terms are Net 30.",
                },
                {
                    "tool_name": "Calculator_calculate",
                    "args": {"expression": "30 + 0"},
                    "result": "30",
                },
            ],
            resolved_model_id="fake-model",
        )

    async def fake_stream_rag_answer(**kwargs: object):
        stream_calls.append(kwargs)
        yield "RAG ", object(), "fake-rag-model"
        yield "final", object(), "fake-rag-model"

    monkeypatch.setattr(chat_service, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: FakeRetrievalTool(),
    )
    monkeypatch.setattr(rag_runtime, "stream_rag_answer", fake_stream_rag_answer)

    service = ChatRuntimeService()

    async def run() -> list[dict[str, object]]:
        return [
            event
            async for event in service.astream_events(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What are Northway Solutions payment terms?",
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": "thread-mixed-rag-stream",
                        "collection_name": "oracle_web_embeddings",
                        "mode": "mixed",
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

    assert mcp_answer_callbacks == [None]
    assert text_deltas == ["RAG ", "final"]
    assert stream_calls
    assert stream_calls[0]["docs"] == docs
    assert "Calculator_calculate" in str(stream_calls[0]["supplemental_context"])

    state = service.get_state_values({"configurable": {"thread_id": "thread-mixed-rag-stream"}})
    assert state is not None
    assert state["final_answer"] == "RAG final"
    assert state["mcp_tools_used"] == ["oracle_retrieval", "Calculator_calculate"]


def test_chat_runtime_service_reports_mixed_retrieval_failure(monkeypatch) -> None:
    class FakeRetrievalTool:
        name = "oracle_retrieval"
        _retrieval_state = {"docs": [], "error": "DPY-6001: service unavailable"}

    async def fake_run_mcp_agent_turn(**kwargs: object) -> MCPAgentTurn:
        return MCPAgentTurn(
            answer="",
            tools_used=["oracle_retrieval"],
            tool_invocations=[
                {
                    "tool_name": "oracle_retrieval",
                    "args": {"query": "Northway Solutions payment terms"},
                    "result": "Oracle retrieval failed while searching the knowledge base.",
                }
            ],
            resolved_model_id="fake-model",
        )

    monkeypatch.setattr(chat_service, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: FakeRetrievalTool(),
    )

    service = ChatRuntimeService()
    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "What are Northway Solutions payment terms?"}],
            model_id=None,
            thread_id="thread-mixed-retrieval-error",
            session_id=None,
            mode="mixed",
            collection_name="oracle_web_embeddings",
            enable_reranker=False,
            enable_tracing=False,
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert "database is available" in str(result["final_answer"])
    assert result["error"] == result["final_answer"]
    assert result["citations"] == []
    assert result["mcp_tools_used"] == ["oracle_retrieval"]
