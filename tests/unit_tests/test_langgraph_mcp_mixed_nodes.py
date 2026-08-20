from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from src.rag_agent.graphs import tool_agent_execution
from src.rag_agent.graphs.mcp_policies import (
    NO_ORACLE_CONTEXT_ANSWER,
    ORACLE_RETRIEVAL_FAILED_ANSWER,
)
from src.rag_agent.graphs.nodes import mcp, mixed, rag
from src.rag_agent.graphs.runtime import stable_terminal_message_id
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore


async def run_compose_with_messages(
    messages: list[object],
    runtime: object,
    heartbeat_calls: list[str] | None = None,
    release_calls: list[str] | None = None,
    terminal_calls: list[str] | None = None,
    terminal_error: BaseException | None = None,
) -> dict[str, object]:
    original = mixed.reconstruct_tool_agent_turn
    original_release = mixed.release_tool_agent_turn
    original_failure_release = mixed.release_tool_agent_turn_after_failure
    original_heartbeat = mixed.run_with_lease_heartbeat
    original_marker = mixed.mark_tool_agent_turn_terminal
    mixed.reconstruct_tool_agent_turn = lambda **_kwargs: _await_turn(runtime)  # type: ignore[assignment]

    async def release(*_args: object) -> None:
        if release_calls is not None:
            release_calls.append("mixed")

    mixed.release_tool_agent_turn = release  # type: ignore[assignment]
    mixed.release_tool_agent_turn_after_failure = release  # type: ignore[assignment]

    async def mark_terminal(*_args: object) -> None:
        if terminal_error is not None:
            raise terminal_error
        if terminal_calls is not None:
            terminal_calls.append("mixed")

    mixed.mark_tool_agent_turn_terminal = mark_terminal

    async def heartbeat_wrapper(_config, _turn, operation_factory, **_kwargs):
        if heartbeat_calls is not None:
            heartbeat_calls.append("heartbeat")
        return await operation_factory()

    mixed.run_with_lease_heartbeat = heartbeat_wrapper  # type: ignore[assignment]
    try:
        return await mixed.run_mixed_compose_node(
            {"messages": messages}, {"configurable": {}}, runtime
        )  # type: ignore[arg-type]
    finally:
        mixed.reconstruct_tool_agent_turn = original
        mixed.release_tool_agent_turn = original_release
        mixed.release_tool_agent_turn_after_failure = original_failure_release
        mixed.run_with_lease_heartbeat = original_heartbeat
        mixed.mark_tool_agent_turn_terminal = original_marker


async def run_mcp_compose_with_messages(
    messages: list[object],
    runtime: object,
    release_calls: list[str] | None = None,
    terminal_calls: list[str] | None = None,
    terminal_error: BaseException | None = None,
) -> dict[str, object]:
    original = mcp.reconstruct_tool_agent_turn
    original_release = mcp.release_tool_agent_turn
    original_failure_release = mcp.release_tool_agent_turn_after_failure
    original_marker = mcp.mark_tool_agent_turn_terminal
    mcp.reconstruct_tool_agent_turn = lambda **_kwargs: _await_turn(runtime)  # type: ignore[assignment]

    async def release(*_args: object) -> None:
        if release_calls is not None:
            release_calls.append("mcp")

    mcp.release_tool_agent_turn = release  # type: ignore[assignment]
    mcp.release_tool_agent_turn_after_failure = release  # type: ignore[assignment]

    async def mark_terminal(*_args: object) -> None:
        if terminal_error is not None:
            raise terminal_error
        if terminal_calls is not None:
            terminal_calls.append("mcp")

    mcp.mark_tool_agent_turn_terminal = mark_terminal
    try:
        return await mcp.run_mcp_compose(
            {"messages": messages}, {"configurable": {}}, runtime
        )  # type: ignore[arg-type]
    finally:
        mcp.reconstruct_tool_agent_turn = original
        mcp.release_tool_agent_turn = original_release
        mcp.release_tool_agent_turn_after_failure = original_failure_release
        mcp.mark_tool_agent_turn_terminal = original_marker


async def _await_turn(runtime: object) -> dict[str, object]:
    return runtime.execution_turn  # type: ignore[attr-defined, no-any-return]


async def _done() -> None:
    return None


async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
    return "Synthesized answer", None, "fake-model"


def _make_runtime(
    *,
    retrieval_docs: list[Document] | None = None,
    retrieval_error: str | None = None,
    question: str = "test question",
    enable_reranker: bool = False,
) -> SimpleNamespace:
    evidence = OracleRetrievalEvidenceStore()
    if retrieval_docs is not None or retrieval_error is not None:
        evidence.record(
            invocation_id="tc1",
            query=question,
            documents=retrieval_docs or [],
            error=retrieval_error,
        )
    return SimpleNamespace(
        context={
            "enable_reranker": False,
            "model_id": "model-1",
        },
        execution_turn={
            "chat_history": [],
            "model_id": "model-1",
            "question": question,
            "run_config": {},
            "system_prompt": "Use tools.",
            "tools": [],
            "oracle_retrieval_evidence": evidence,
            "tool_round_limit": 5,
            "enable_reranker": enable_reranker,
            "lease": SimpleNamespace(thread_id="thread-1", turn_id="turn-1"),
        },
    )


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

    release_calls: list[str] = []
    terminal_calls: list[str] = []
    result = asyncio.run(
        run_compose_with_messages(
            subgraph_output,
            runtime,
            release_calls=release_calls,
            terminal_calls=terminal_calls,
        )
    )

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
    assert release_calls == ["mixed"]
    assert terminal_calls == ["mixed"]
    assert assistant.id == stable_terminal_message_id("mixed", "thread-1", "turn-1")


def test_mcp_compose_uses_only_the_current_turn_messages() -> None:
    runtime = _make_runtime(question="current question")
    messages = [
        HumanMessage(content="previous question"),
        AIMessage(
            content="",
            tool_calls=[{"id": "previous-call", "name": "previous_tool", "args": {}}],
        ),
        ToolMessage(
            content="previous result",
            tool_call_id="previous-call",
            name="previous_tool",
        ),
        AIMessage(content="Previous answer."),
        HumanMessage(content="current question"),
        AIMessage(content="Current answer."),
    ]

    release_calls: list[str] = []
    terminal_calls: list[str] = []
    result = asyncio.run(
        run_mcp_compose_with_messages(messages, runtime, release_calls, terminal_calls)
    )

    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Current answer."
    assert assistant.additional_kwargs["mcp_used"] is False
    assert release_calls == ["mcp"]
    assert terminal_calls == ["mcp"]
    assert assistant.id == stable_terminal_message_id("mcp", "thread-1", "turn-1")
    assert assistant.additional_kwargs["mcp_tools_used"] == []
    assert assistant.additional_kwargs["mcp_tool_invocations"] == []


@pytest.mark.parametrize("mode", ["mcp", "mixed"])
def test_terminal_mark_failure_preserves_error_and_releases(mode: str) -> None:
    runtime = _make_runtime()
    messages = [HumanMessage(content="question"), AIMessage(content="answer")]
    release_calls: list[str] = []
    marker_error = RuntimeError(f"{mode} marker failed")

    with pytest.raises(RuntimeError, match=f"{mode} marker failed"):
        if mode == "mcp":
            asyncio.run(
                run_mcp_compose_with_messages(
                    messages,
                    runtime,
                    release_calls=release_calls,
                    terminal_error=marker_error,
                )
            )
        else:
            asyncio.run(
                run_compose_with_messages(
                    messages,
                    runtime,
                    release_calls=release_calls,
                    terminal_error=marker_error,
                )
            )

    assert release_calls == [mode]


@pytest.mark.parametrize("mode", ["mcp", "mixed"])
def test_stable_terminal_id_reducer_does_not_duplicate_terminal_answer(mode: str) -> None:
    runtime = _make_runtime()
    old_terminal = AIMessage(id=f"{mode}-model-id", content="answer")
    messages = [HumanMessage(id="question-1", content="question"), old_terminal]

    if mode == "mcp":
        result = asyncio.run(run_mcp_compose_with_messages(messages, runtime))
    else:
        result = asyncio.run(run_compose_with_messages(messages, runtime))

    update = result["messages"]
    reduced = add_messages(messages, update)  # type: ignore[arg-type]
    terminal_answers = [
        message
        for message in reduced
        if isinstance(message, AIMessage) and message.content == "answer"
    ]

    assert len(terminal_answers) == 1
    assert terminal_answers[0].id == stable_terminal_message_id(mode, "thread-1", "turn-1")
    assert len(update) == 2
    assert update[0].__class__.__name__ == "RemoveMessage"
    assert update[0].id == old_terminal.id
    assert update[1].id == terminal_answers[0].id

    replayed = add_messages(reduced, [terminal_answers[0]])
    replayed_answers = [
        message
        for message in replayed
        if isinstance(message, AIMessage) and message.content == "answer"
    ]
    assert len(replayed_answers) == 1
    assert replayed_answers[0].id == terminal_answers[0].id


def test_current_turn_messages_exclude_prior_terminal_answer() -> None:
    messages = [
        HumanMessage(content="previous question"),
        AIMessage(content="Previous answer."),
        HumanMessage(content="current question"),
    ]

    transcript = tool_agent_execution.analyze_tool_execution(
        tool_agent_execution.messages_since_latest_user(messages)
    )

    assert transcript["has_terminal_answer"] is False
    assert transcript["final_answer"] == ""


def test_mixed_compose_synthesizes_when_subgraph_has_no_final_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        mixed.rag_runtime,
        "rerank_retrieved_docs",
        lambda q, docs, **kw: docs,
    )
    monkeypatch.setattr(mixed.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)
    monkeypatch.setattr(
        mixed.rag_runtime, "citations_from_docs", lambda docs: [{"source": "doc.md"}]
    )
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda docs: [{"source": "doc.md"}])

    retrieval_doc = Document(page_content="Context data", metadata={"source": "doc.md"})
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

    heartbeat_calls: list[str] = []
    result = asyncio.run(run_compose_with_messages(subgraph_output, runtime, heartbeat_calls))

    assert len(result["messages"]) >= 1
    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Synthesized answer"
    assert assistant.additional_kwargs["citations"] == [{"source": "doc.md"}]
    assert heartbeat_calls == ["heartbeat", "heartbeat"]


def test_mixed_rerank_is_thread_offloaded_and_heartbeat_bound(monkeypatch) -> None:
    retrieval_doc = Document(page_content="Context data", metadata={"source": "doc.md"})
    runtime = _make_runtime(
        retrieval_docs=[retrieval_doc],
        question="Use docs",
        enable_reranker=True,
    )
    messages = [
        AIMessage(
            content=".",
            tool_calls=[{"id": "tc1", "name": "oracle_retrieval", "args": {}}],
        ),
        ToolMessage(content="Context data", tool_call_id="tc1", name="oracle_retrieval"),
        AIMessage(content="Final answer."),
    ]
    to_thread_calls: list[str] = []

    async def fake_to_thread(function, *args, **kwargs):
        to_thread_calls.append(function.__name__)
        return function(*args, **kwargs)

    monkeypatch.setattr(mixed.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(mixed.rag_runtime, "rerank_retrieved_docs", lambda _q, docs, **_kw: docs)
    monkeypatch.setattr(mixed.rag_runtime, "citations_from_docs", lambda _docs: [])
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda _docs: [])
    heartbeat_calls: list[str] = []

    asyncio.run(run_compose_with_messages(messages, runtime, heartbeat_calls))

    assert to_thread_calls == ["<lambda>"]
    assert heartbeat_calls == ["heartbeat"]


def test_mixed_rerank_cancellation_propagates_from_thread_boundary(monkeypatch) -> None:
    runtime = _make_runtime(
        retrieval_docs=[Document(page_content="Context", metadata={"source": "doc.md"})],
        enable_reranker=True,
    )
    messages = [HumanMessage(content="Use docs"), AIMessage(content="Final answer.")]

    async def cancelled_to_thread(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(mixed.asyncio, "to_thread", cancelled_to_thread)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            mixed._compose_mixed_result({"messages": messages}, turn=runtime.execution_turn)
        )


def test_mixed_compose_returns_no_context_for_empty_oracle_evidence(monkeypatch) -> None:
    monkeypatch.setattr(mixed.rag_runtime, "citations_from_docs", lambda docs: [])
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda docs: [])

    runtime = _make_runtime(retrieval_docs=[])
    messages = [
        AIMessage(
            content=".",
            tool_calls=[{"id": "tc1", "name": "oracle_retrieval", "args": {}}],
        ),
        ToolMessage(content="", tool_call_id="tc1", name="oracle_retrieval"),
        AIMessage(content="Made-up answer."),
    ]

    result = asyncio.run(run_compose_with_messages(messages, runtime))

    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == NO_ORACLE_CONTEXT_ANSWER


def test_mixed_compose_returns_retrieval_failure_for_error_evidence(monkeypatch) -> None:
    monkeypatch.setattr(mixed.rag_runtime, "citations_from_docs", lambda docs: [])
    monkeypatch.setattr(mixed.rag_runtime, "serialize_docs", lambda docs: [])

    runtime = _make_runtime(retrieval_error="Oracle connection timed out")
    messages = [
        AIMessage(
            content=".",
            tool_calls=[{"id": "tc1", "name": "oracle_retrieval", "args": {}}],
        ),
        ToolMessage(content="failed", tool_call_id="tc1", name="oracle_retrieval"),
    ]

    result = asyncio.run(run_compose_with_messages(messages, runtime))

    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == ORACLE_RETRIEVAL_FAILED_ANSWER


def test_mixed_compose_route_logic() -> None:
    assert tool_agent_execution.route({"messages": []}) == "__end__"
    assert tool_agent_execution.route({"messages": [AIMessage(content="hello")]}) == "__end__"
    assert (
        tool_agent_execution.route(
            {
                "messages": [
                    AIMessage(
                        content=".",
                        tool_calls=[{"id": "tc1", "name": "lookup", "args": {}}],
                    )
                ]
            }
        )
        == "run_tools"
    )


def test_call_llm_node_uses_configured_mcp_round_limit(monkeypatch) -> None:
    class FakeModel:
        async def ainvoke(self, messages: object, *, config: object) -> AIMessage:
            _ = messages, config
            return AIMessage(content="Final answer")

    monkeypatch.setattr(tool_agent_execution, "get_llm", lambda **kwargs: FakeModel())
    monkeypatch.setattr(
        tool_agent_execution,
        "reconstruct_tool_agent_turn",
        lambda **_kwargs: _await_turn(_kwargs["runtime"]),
    )
    monkeypatch.setattr(tool_agent_execution, "release_tool_agent_turn", lambda *_args: _done())
    monkeypatch.setattr(
        tool_agent_execution,
        "run_with_lease_heartbeat",
        lambda _config, _turn, operation_factory, **_kwargs: operation_factory(),
    )
    runtime = SimpleNamespace(
        context={"max_rounds": 3},
        execution_turn={
            "tools": [],
            "model_id": "model-1",
            "system_prompt": "Use tools when needed.",
            "chat_history": [],
            "question": "test question",
            "tool_round_limit": 3,
        },
    )

    result = asyncio.run(
        tool_agent_execution.call_llm_node(
            {"configurable": {}},
            {},
            runtime,  # type: ignore[arg-type]
        )
    )

    assert result["remaining_steps"] == 2


def test_call_llm_node_uses_mcp_round_setting_by_default(monkeypatch) -> None:
    class FakeModel:
        async def ainvoke(self, messages: object, *, config: object) -> AIMessage:
            _ = messages, config
            return AIMessage(content="Final answer")

    monkeypatch.setattr(tool_agent_execution, "get_llm", lambda **kwargs: FakeModel())
    monkeypatch.setattr(
        tool_agent_execution,
        "reconstruct_tool_agent_turn",
        lambda **_kwargs: _await_turn(_kwargs["runtime"]),
    )
    monkeypatch.setattr(tool_agent_execution, "release_tool_agent_turn", lambda *_args: _done())
    monkeypatch.setattr(
        tool_agent_execution,
        "run_with_lease_heartbeat",
        lambda _config, _turn, operation_factory, **_kwargs: operation_factory(),
    )
    runtime = SimpleNamespace(
        context={},
        execution_turn={
            "tools": [],
            "model_id": "model-1",
            "system_prompt": "Use tools when needed.",
            "chat_history": [],
            "question": "test question",
            "tool_round_limit": 3,
        },
    )

    result = asyncio.run(
        tool_agent_execution.call_llm_node(
            {"configurable": {}},
            {},
            runtime,  # type: ignore[arg-type]
        )
    )

    assert result["remaining_steps"] == 2


def test_analyze_tool_execution_pairs_tool_results_with_calls() -> None:
    messages = [
        AIMessage(
            id="m1",
            content=".",
            tool_calls=[{"id": "tc1", "name": "lookup", "args": {"k": "v"}}],
        ),
        ToolMessage(content="result", tool_call_id="tc1", name="lookup"),
        AIMessage(id="m2", content="Final."),
    ]
    transcript = tool_agent_execution.analyze_tool_execution(messages)
    assert transcript["final_answer"] == "Final."
    assert transcript["has_terminal_answer"] is True
    assert transcript["tools_used"] == ["lookup"]
    assert transcript["tool_invocations"] == [
        {
            "invocation_id": "tc1",
            "tool_name": "lookup",
            "args": {"k": "v"},
            "result": "result",
        }
    ]


def test_analyze_tool_execution_records_tool_errors() -> None:
    messages = [
        AIMessage(
            id="m1",
            content=".",
            tool_calls=[{"id": "tc1", "name": "lookup", "args": {}}],
        ),
        ToolMessage(content="error occurred", tool_call_id="tc1", name="lookup", status="error"),
        AIMessage(id="m2", content="Final."),
    ]
    transcript = tool_agent_execution.analyze_tool_execution(messages)
    assert transcript["tool_invocations"][0]["error"] == "error occurred"


def test_analyze_tool_execution_records_incomplete_tool_calls_as_failures() -> None:
    transcript = tool_agent_execution.analyze_tool_execution(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "lookup", "args": {}}],
            )
        ]
    )

    assert transcript["tool_invocations"] == [
        {
            "invocation_id": "tc1",
            "tool_name": "lookup",
            "args": {},
            "error": tool_agent_execution.INCOMPLETE_TOOL_CALL_ERROR,
        }
    ]
    assert transcript["tools_used"] == ["lookup"]
    assert transcript["has_terminal_answer"] is False


def test_analyze_tool_execution_preserves_call_order_with_incomplete_calls() -> None:
    transcript = tool_agent_execution.analyze_tool_execution(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc1", "name": "first", "args": {}},
                    {"id": "tc2", "name": "second", "args": {}},
                ],
            ),
            ToolMessage(content="second result", tool_call_id="tc2", name="second"),
        ]
    )

    assert transcript["tool_invocations"] == [
        {
            "invocation_id": "tc1",
            "tool_name": "first",
            "args": {},
            "error": tool_agent_execution.INCOMPLETE_TOOL_CALL_ERROR,
        },
        {
            "invocation_id": "tc2",
            "tool_name": "second",
            "args": {},
            "result": "second result",
        },
    ]
    assert transcript["tools_used"] == ["first", "second"]


def test_mcp_compose_reports_an_incomplete_tool_call_as_a_failure() -> None:
    runtime = _make_runtime(question="current question")
    result = asyncio.run(
        run_mcp_compose_with_messages(
            [
                HumanMessage(content="current question"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc1", "name": "lookup", "args": {}}],
                ),
            ],
            runtime,
        )
    )

    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert (
        assistant.content
        == "Workflow failed because tool execution failed: lookup. See tool output for details."
    )
    assert assistant.additional_kwargs["mcp_used"] is True
    assert assistant.additional_kwargs["mcp_tool_invocations"][0]["error"] == (
        tool_agent_execution.INCOMPLETE_TOOL_CALL_ERROR
    )


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
    monkeypatch.setattr(
        rag.rag_runtime, "citations_from_docs", lambda docs: [{"source": "terms.pdf"}]
    )
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
