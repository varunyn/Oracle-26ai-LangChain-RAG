from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.rag_agent.graphs.nodes import mcp, mixed, rag


@dataclass
class FakeMcpTurn:
    answer: str = "Tool answer"
    tools_used: list[str] | None = None
    tool_invocations: list[dict[str, object]] | None = None
    resolved_model_id: str = "fake-mcp-model"
    state_messages: list[object] | None = None

    def __post_init__(self) -> None:
        if self.tools_used is None:
            self.tools_used = ["lookup"]
        if self.tool_invocations is None:
            self.tool_invocations = [{"tool_name": "lookup", "status": "success"}]
        if self.state_messages is None:
            self.state_messages = [
                AIMessage(
                    id="assistant-tool-call",
                    content=".",
                    tool_calls=[
                        {"id": "call-1", "name": "lookup", "args": {"query": "invoice"}}
                    ],
                ),
                ToolMessage(
                    content="ok",
                    tool_call_id="call-1",
                    name="lookup",
                ),
                AIMessage(id="assistant-final", content=self.answer),
            ]


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

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
        run_config = kwargs["run_config"]
        assert isinstance(run_config, dict)
        assert run_config["callbacks"] == ["outer-callback"]
        assert kwargs["require_tool_call"] is True
        return FakeMcpTurn()

    monkeypatch.setattr(mcp, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
    monkeypatch.setattr(mcp, "start_langfuse_chat_trace", lambda **kwargs: _FakeTraceContextManager())
    monkeypatch.setattr(mcp, "get_llm", lambda model_id=None: SimpleNamespace(model_id="fake-mcp-model"))

    runtime = SimpleNamespace(
        context={"model_id": "model-1", "session_id": "session-1", "enable_tracing": False},
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        mcp.run_mcp_node(
            {"messages": [HumanMessage(content="Use a tool")]},
            {"callbacks": ["outer-callback"]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assert len(result["messages"]) == 3
    assistant = result["messages"][-1]
    tool_call_message = result["messages"][0]
    assert isinstance(tool_call_message, AIMessage)
    assert tool_call_message.tool_calls[0]["id"] == "call-1"
    assert isinstance(result["messages"][1], ToolMessage)
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Tool answer"
    assert assistant.additional_kwargs["mode"] == "mcp"
    assert assistant.additional_kwargs["mcp_used"] is True
    assert assistant.additional_kwargs["mcp_tools_used"] == ["lookup"]


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

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
        run_config = kwargs["run_config"]
        assert isinstance(run_config, dict)
        assert run_config["callbacks"] == ["outer-callback"]
        assert kwargs["require_tool_call"] is True
        return FakeMcpTurn(
            answer="Draft tool answer",
            tools_used=["oracle_retrieval", "lookup"],
            tool_invocations=[
                {"tool_name": "oracle_retrieval", "result": "Context"},
                {"tool_name": "lookup", "result": "Aux"},
            ],
            resolved_model_id="fake-mixed-model",
        )

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        raise AssertionError("mixed node should not synthesize after an agent final answer")

    monkeypatch.setattr(mixed, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
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
            {"callbacks": ["outer-callback"]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assert len(result["messages"]) == 3
    tool_call_message = result["messages"][0]
    assistant = result["messages"][-1]
    assert isinstance(tool_call_message, AIMessage)
    assert tool_call_message.tool_calls[0]["id"] == "call-1"
    assert isinstance(result["messages"][1], ToolMessage)
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Draft tool answer"
    assert assistant.additional_kwargs["mode"] == "mixed"
    assert assistant.additional_kwargs["mcp_used"] is True
    assert assistant.additional_kwargs["citations"] == [{"source": "doc.md"}]


def test_mixed_node_synthesizes_when_tool_loop_has_no_final_answer(monkeypatch) -> None:
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

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
        _ = kwargs
        return FakeMcpTurn(
            answer=".",
            tools_used=["oracle_retrieval"],
            tool_invocations=[{"tool_name": "oracle_retrieval", "result": "Context"}],
            resolved_model_id="fake-mixed-model",
            state_messages=[
                AIMessage(
                    id="assistant-tool-call",
                    content=".",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "oracle_retrieval",
                            "args": {"query": "invoice"},
                        }
                    ],
                ),
                ToolMessage(
                    content="Context",
                    tool_call_id="call-1",
                    name="oracle_retrieval",
                ),
            ],
        )

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        assert kwargs["question"] == "Use docs"
        return "Synthesized answer", None, "fake-mixed-model"

    monkeypatch.setattr(mixed, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
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
            {"messages": [HumanMessage(content="Use docs")]},
            {"callbacks": ["outer-callback"]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assert len(result["messages"]) == 3
    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Synthesized answer"
    assert assistant.additional_kwargs["citations"] == [{"source": "doc.md"}]


def test_mixed_node_preserves_native_tool_messages_only_for_mcp_turns(monkeypatch) -> None:
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

    async def fake_run_mcp_agent_turn_without_mcp(**kwargs: object) -> FakeMcpTurn:
        assert kwargs["require_mcp_tool_call_when_referenced"] is True
        return FakeMcpTurn(
            answer="RAG only answer",
            tools_used=["oracle_retrieval"],
            tool_invocations=[{"tool_name": "oracle_retrieval", "result": "Context"}],
            resolved_model_id="fake-mixed-model",
            state_messages=[AIMessage(id="assistant-final", content="RAG only answer")],
        )

    async def fake_run_mcp_agent_turn_with_mcp(**kwargs: object) -> FakeMcpTurn:
        assert kwargs["require_mcp_tool_call_when_referenced"] is True
        return FakeMcpTurn(
            answer="Draft answer",
            tools_used=["oracle_retrieval", "lookup"],
            tool_invocations=[
                {"tool_name": "oracle_retrieval", "result": "Context"},
                {"tool_name": "lookup", "result": "ok"},
            ],
            resolved_model_id="fake-mixed-model",
        )

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        raise AssertionError("mixed node should not synthesize after an agent final answer")

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
    retrieval_only_result = asyncio.run(
        mixed.run_mixed_node(
            {"messages": [HumanMessage(content="Use docs only")]},
            {"callbacks": ["outer-callback"]},
            runtime,  # type: ignore[arg-type]
        )
    )

    monkeypatch.setattr(mixed, "run_mcp_agent_turn", fake_run_mcp_agent_turn_with_mcp)
    mcp_turn_result = asyncio.run(
        mixed.run_mixed_node(
            {"messages": [HumanMessage(content="Use docs and tools")]},
            {"callbacks": ["outer-callback"]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assert len(retrieval_only_result["messages"]) == 1
    assert isinstance(retrieval_only_result["messages"][0], AIMessage)
    assert all(not isinstance(message, ToolMessage) for message in retrieval_only_result["messages"])
    assert len(mcp_turn_result["messages"]) == 3
    assert isinstance(mcp_turn_result["messages"][0], AIMessage)
    assert mcp_turn_result["messages"][0].tool_calls[0]["id"] == "call-1"
    assert isinstance(mcp_turn_result["messages"][1], ToolMessage)


def test_mixed_node_does_not_emit_tool_messages_for_retrieval_only(monkeypatch) -> None:
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
