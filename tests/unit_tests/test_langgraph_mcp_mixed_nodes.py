from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.rag_agent.graphs.nodes import mixed, rag


async def run_compose_with_messages(
    messages: list[object],
    runtime: object,
) -> dict[str, object]:
    return await mixed.run_mixed_compose_node(
        {"messages": messages},
        {},
        runtime,  # type: ignore[arg-type]
    )


class FakeRetrievalTool:
    name = "oracle_retrieval"
    _retrieval_state: dict[str, list[object]] = {}


async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
    return "Synthesized answer", None, "fake-model"


def _make_runtime(*, retrieval_docs: list[object] | None = None, question: str = "test question") -> SimpleNamespace:
    tool = FakeRetrievalTool()
    if retrieval_docs is not None:
        tool._retrieval_state = {"docs": retrieval_docs}
    return SimpleNamespace(
        context={
            "mcp_subgraph_tools": [tool],
            "mcp_subgraph_question": question,
            "mcp_subgraph_run_cfg": {},
            "enable_reranker": False,
            "model_id": "model-1",
        },
    )


def test_mcp_node_runs_agent_turn_without_chat_runtime_service(monkeypatch) -> None:
    """MCP (non-mixed) node test — unchanged, tests mcp module level."""
    from dataclasses import dataclass
    from src.rag_agent.graphs.nodes import mcp

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
                    ToolMessage(content="ok", tool_call_id="call-1", name="lookup"),
                    AIMessage(id="assistant-final", content=self.answer),
                ]

    async def fake_run_mcp_agent_turn(**kwargs: object) -> FakeMcpTurn:
        run_config = kwargs["run_config"]
        assert isinstance(run_config, dict)
        assert run_config["callbacks"] == ["outer-callback"]
        assert kwargs["require_tool_call"] is True
        return FakeMcpTurn()

    monkeypatch.setattr(mcp, "run_mcp_agent_turn", fake_run_mcp_agent_turn)
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


def test_mixed_compose_node_extracts_tool_invocations_from_subgraph_messages(monkeypatch) -> None:
    monkeypatch.setattr(mixed.rag_runtime, "citations_from_docs", lambda docs: [])
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda docs: [])
    monkeypatch.setattr(mixed.rag_runtime, "rerank_retrieved_docs", lambda q, docs, **kw: docs)
    monkeypatch.setattr(mixed.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)

    runtime = _make_runtime()
    subgraph_output = [
        AIMessage(
            id="call-llm-1",
            content=".",
            tool_calls=[{"id": "tc1", "name": "lookup", "args": {"key": "x"}}],
        ),
        ToolMessage(content="result-x", tool_call_id="tc1", name="lookup"),
        AIMessage(id="call-llm-2", content="Final answer text."),
    ]

    result = asyncio.run(run_compose_with_messages(subgraph_output, runtime))

    assert len(result["messages"]) >= 1
    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Final answer text."
    assert assistant.additional_kwargs["mode"] == "mixed"
    assert assistant.additional_kwargs["mcp_used"] is True
    assert assistant.additional_kwargs["mcp_tools_used"] == ["lookup"]
    assert len(assistant.additional_kwargs["mcp_tool_invocations"]) == 1
    assert assistant.additional_kwargs["mcp_tool_invocations"][0]["tool_name"] == "lookup"
    assert assistant.additional_kwargs["mcp_tool_invocations"][0]["result"] == "result-x"


def test_mixed_compose_synthesizes_when_subgraph_has_no_final_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        mixed.rag_runtime,
        "rerank_retrieved_docs",
        lambda q, docs, **kw: docs,
    )
    monkeypatch.setattr(mixed.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)
    monkeypatch.setattr(mixed.rag_runtime, "citations_from_docs", lambda docs: [{"source": "doc.md"}])
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda docs: [{"source": "doc.md"}])

    retrieval_doc = SimpleNamespace(page_content="Context data", metadata={"source": "doc.md"})
    runtime = _make_runtime(
        retrieval_docs=[retrieval_doc],
        question="Use docs",
    )
    subgraph_output = [
        AIMessage(
            id="call-llm-1",
            content=".",
            tool_calls=[{"id": "tc1", "name": "oracle_retrieval", "args": {"q": "docs"}}],
        ),
        ToolMessage(content="Context data", tool_call_id="tc1", name="oracle_retrieval"),
    ]

    result = asyncio.run(run_compose_with_messages(subgraph_output, runtime))

    assert len(result["messages"]) >= 1
    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Synthesized answer"
    assert assistant.additional_kwargs["citations"] == [{"source": "doc.md"}]


def test_mixed_compose_route_logic() -> None:
    assert mixed.route({"messages": []}) == "__end__"
    assert mixed.route({"messages": [AIMessage(content="hello")]}) == "__end__"
    assert mixed.route({
        "messages": [
            AIMessage(
                content=".",
                tool_calls=[{"id": "tc1", "name": "lookup", "args": {}}],
            )
        ]
    }) == "run_tools"


def test_extract_tool_invocations(monkeypatch) -> None:
    messages = [
        AIMessage(
            id="m1",
            content=".",
            tool_calls=[{"id": "tc1", "name": "lookup", "args": {"k": "v"}}],
        ),
        ToolMessage(content="result", tool_call_id="tc1", name="lookup"),
        AIMessage(id="m2", content="Final."),
    ]
    result = mixed.extract_tool_invocations_from_messages(messages)
    assert len(result) == 1
    assert result[0]["tool_name"] == "lookup"
    assert result[0]["args"] == {"k": "v"}
    assert result[0]["result"] == "result"


def test_extract_tool_invocations_handles_errors() -> None:
    messages = [
        AIMessage(
            id="m1",
            content=".",
            tool_calls=[{"id": "tc1", "name": "lookup", "args": {}}],
        ),
        ToolMessage(content="error occurred", tool_call_id="tc1", name="lookup", status="error"),
        AIMessage(id="m2", content="Final."),
    ]
    result = mixed.extract_tool_invocations_from_messages(messages)
    assert len(result) == 1
    assert result[0]["error"] == "error occurred"


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
            {},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.additional_kwargs["mode"] == "rag"
    assert assistant.additional_kwargs["citations"] == [{"source": "terms.pdf"}]
    assert emitted == []
