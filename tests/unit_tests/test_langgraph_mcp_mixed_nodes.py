from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs.nodes import mcp, mixed


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

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
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
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
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
