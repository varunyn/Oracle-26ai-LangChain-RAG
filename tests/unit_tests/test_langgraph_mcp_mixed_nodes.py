from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs.nodes import mcp, mixed, rag


@dataclass
class FakeMcpTurn:
    answer: str = "Tool answer"
    tools_used: list[str] | None = None
    tool_invocations: list[dict[str, object]] | None = None
    resolved_model_id: str = "fake-mcp-model"

    def __post_init__(self) -> None:
        if self.tools_used is None:
            self.tools_used = ["lookup"]
        if self.tool_invocations is None:
            self.tool_invocations = [{"tool_name": "lookup", "status": "success"}]


def test_mcp_node_runs_agent_turn_without_chat_runtime_service(monkeypatch) -> None:
    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    emitted: list[dict[str, object]] = []

    def stream_writer(payload: dict[str, object]) -> None:
        emitted.append(payload)

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
        callback = kwargs["tool_progress_callback"]
        assert callable(callback)
        callback({"phase": "start", "tool_name": "lookup", "tool_run_id": "call-1"})
        callback({"phase": "end", "tool_name": "lookup", "tool_run_id": "call-1", "result": "ok"})
        return FakeMcpTurn()

    monkeypatch.setattr(mcp, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
    monkeypatch.setattr(mcp, "get_stream_writer", lambda: stream_writer)
    monkeypatch.setattr(mcp, "start_langfuse_chat_trace", lambda **kwargs: _FakeTraceContextManager())
    monkeypatch.setattr(mcp, "get_llm", lambda model_id=None: SimpleNamespace(model_id="fake-mcp-model"))

    runtime = SimpleNamespace(
        context={"model_id": "model-1", "session_id": "session-1", "enable_tracing": False},
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        mcp.run_mcp_node(
            {"messages": [HumanMessage(content="Use a tool")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Tool answer"
    assert assistant.additional_kwargs["mode"] == "mcp"
    assert assistant.additional_kwargs["mcp_used"] is True
    assert assistant.additional_kwargs["mcp_tools_used"] == ["lookup"]
    assert [event["name"] for event in emitted] == ["mcp_tool_activity", "mcp_tool_activity"]
    assert [event["payload"]["status"] for event in emitted] == ["running", "finished"]


def test_mixed_node_runs_agent_turn_without_chat_runtime_service(monkeypatch) -> None:
    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    class FakeRetrievalTool:
        name = "oracle_retrieval"
        _retrieval_state = {
            "docs": [SimpleNamespace(page_content="Context", metadata={"source": "doc.md"})]
        }

    emitted: list[dict[str, object]] = []

    def stream_writer(payload: dict[str, object]) -> None:
        emitted.append(payload)

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
        callback = kwargs["tool_progress_callback"]
        assert callable(callback)
        callback({"phase": "start", "tool_name": "lookup", "tool_run_id": "call-1"})
        callback({"phase": "end", "tool_name": "lookup", "tool_run_id": "call-1", "result": "ok"})
        return FakeMcpTurn(
            answer="Mixed answer",
            tools_used=["oracle_retrieval", "lookup"],
            tool_invocations=[
                {"tool_name": "oracle_retrieval", "result": "Context"},
                {"tool_name": "lookup", "result": "Aux"},
            ],
            resolved_model_id="fake-mixed-model",
        )

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        _ = kwargs
        return "Mixed answer", None, "fake-mixed-model"

    monkeypatch.setattr(mixed, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
    monkeypatch.setattr(mixed, "get_stream_writer", lambda: stream_writer)
    monkeypatch.setattr(mixed, "start_langfuse_chat_trace", lambda **kwargs: _FakeTraceContextManager())
    monkeypatch.setattr(mixed, "get_llm", lambda model_id=None: SimpleNamespace(model_id="fake-mixed-model"))
    monkeypatch.setattr(
        mixed.rag_runtime,
        "build_oracle_retrieval_tool",
        lambda **kwargs: FakeRetrievalTool(),
    )
    monkeypatch.setattr(
        mixed.rag_runtime,
        "rerank_retrieved_docs",
        lambda query, docs, *, enable_reranker: docs,
    )
    monkeypatch.setattr(mixed.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)
    monkeypatch.setattr(mixed.rag_runtime, "citations_from_docs", lambda docs: [{"source": "doc.md"}])
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda docs: [{"source": "doc.md"}])

    runtime = SimpleNamespace(
        context={
            "model_id": "model-1",
            "session_id": "session-1",
            "collection_name": "kb",
            "enable_reranker": False,
            "enable_tracing": False,
        },
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        mixed.run_mixed_node(
            {"messages": [HumanMessage(content="Use tools and docs")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Mixed answer"
    assert assistant.additional_kwargs["mode"] == "mixed"
    assert assistant.additional_kwargs["mcp_used"] is True
    assert assistant.additional_kwargs["citations"] == [{"source": "doc.md"}]
    assert [event["name"] for event in emitted] == ["mcp_tool_activity", "mcp_tool_activity"]
    assert [event["payload"]["status"] for event in emitted] == ["running", "finished"]


def test_mixed_node_emits_mcp_activity_only_for_mcp_turns(monkeypatch) -> None:
    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    class FakeRetrievalTool:
        name = "oracle_retrieval"
        _retrieval_state = {
            "docs": [SimpleNamespace(page_content="Context", metadata={"source": "doc.md"})]
        }

    retrieval_only_events: list[dict[str, object]] = []
    mcp_turn_events: list[dict[str, object]] = []

    async def fake_run_mcp_agent_turn_without_mcp(**kwargs: object) -> FakeMcpTurn:
        _ = kwargs
        return FakeMcpTurn(
            answer="RAG only answer",
            tools_used=["oracle_retrieval"],
            tool_invocations=[{"tool_name": "oracle_retrieval", "result": "Context"}],
            resolved_model_id="fake-mixed-model",
        )

    async def fake_run_mcp_agent_turn_with_mcp(**kwargs: object) -> FakeMcpTurn:
        callback = kwargs["tool_progress_callback"]
        assert callable(callback)
        callback({"phase": "start", "tool_name": "lookup", "tool_run_id": "call-1"})
        callback({"phase": "end", "tool_name": "lookup", "tool_run_id": "call-1", "result": "ok"})
        return FakeMcpTurn(
            answer="Mixed answer",
            tools_used=["oracle_retrieval", "lookup"],
            tool_invocations=[
                {"tool_name": "oracle_retrieval", "result": "Context"},
                {"tool_name": "lookup", "result": "ok"},
            ],
            resolved_model_id="fake-mixed-model",
        )

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        _ = kwargs
        return "Mixed answer", None, "fake-mixed-model"

    monkeypatch.setattr(mixed, "start_langfuse_chat_trace", lambda **kwargs: _FakeTraceContextManager())
    monkeypatch.setattr(mixed, "get_llm", lambda model_id=None: SimpleNamespace(model_id="fake-mixed-model"))
    monkeypatch.setattr(
        mixed.rag_runtime,
        "build_oracle_retrieval_tool",
        lambda **kwargs: FakeRetrievalTool(),
    )
    monkeypatch.setattr(
        mixed.rag_runtime,
        "rerank_retrieved_docs",
        lambda query, docs, *, enable_reranker: docs,
    )
    monkeypatch.setattr(mixed.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)
    monkeypatch.setattr(mixed.rag_runtime, "citations_from_docs", lambda docs: [{"source": "doc.md"}])
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda docs: [{"source": "doc.md"}])

    runtime = SimpleNamespace(
        context={
            "model_id": "model-1",
            "session_id": "session-1",
            "collection_name": "kb",
            "enable_reranker": False,
            "enable_tracing": False,
        },
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    monkeypatch.setattr(mixed, "run_mcp_agent_turn", fake_run_mcp_agent_turn_without_mcp)
    monkeypatch.setattr(mixed, "get_stream_writer", lambda: retrieval_only_events.append)
    asyncio.run(mixed.run_mixed_node({"messages": [HumanMessage(content="Use docs only")]}, runtime))  # type: ignore[arg-type]

    monkeypatch.setattr(mixed, "run_mcp_agent_turn", fake_run_mcp_agent_turn_with_mcp)
    monkeypatch.setattr(mixed, "get_stream_writer", lambda: mcp_turn_events.append)
    asyncio.run(mixed.run_mixed_node({"messages": [HumanMessage(content="Use docs and tools")]}, runtime))  # type: ignore[arg-type]

    assert retrieval_only_events == []
    assert mcp_turn_events[0]["name"] == "mcp_tool_activity"


def test_rag_node_does_not_emit_mcp_activity_for_retrieval_only(monkeypatch) -> None:
    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    emitted: list[dict[str, object]] = []

    async def fake_contextualize_question(**kwargs: object) -> str:
        return "payment terms"

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        _ = kwargs
        return "Net 30 days.", None, "fake-rag-model"

    monkeypatch.setattr(rag, "start_langfuse_chat_trace", lambda **kwargs: _FakeTraceContextManager())
    monkeypatch.setattr(rag, "get_thread_id", lambda runtime: "thread-1")
    monkeypatch.setattr(rag, "build_run_config", lambda **kwargs: {})
    monkeypatch.setattr(rag, "contextualize_question", fake_contextualize_question)
    monkeypatch.setattr(
        rag.rag_runtime,
        "retrieve_oracle_docs",
        lambda **kwargs: [SimpleNamespace(page_content="Terms", metadata={"source": "terms.pdf"})],
    )
    monkeypatch.setattr(rag.rag_runtime, "rerank_retrieved_docs", lambda *args, **kwargs: args[1])
    monkeypatch.setattr(rag.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)
    monkeypatch.setattr(rag.rag_runtime, "citations_from_docs", lambda docs: [{"source": "terms.pdf"}])
    monkeypatch.setattr(rag.rag_runtime, "serialize_docs", lambda docs: [{"source": "terms.pdf"}])
    monkeypatch.setattr(rag, "emit_usage_observability", lambda **kwargs: (None, None))

    runtime = SimpleNamespace(
        context={
            "model_id": "model-1",
            "session_id": "session-1",
            "collection_name": "kb",
            "enable_reranker": False,
            "enable_tracing": False,
        },
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        rag.run_rag_node(
            {"messages": [HumanMessage(content="What are the terms?")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.additional_kwargs["mode"] == "rag"
    assert assistant.additional_kwargs["citations"] == [{"source": "terms.pdf"}]
    assert emitted == []
