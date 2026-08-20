import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.rag_agent.graphs import tool_agent_turn
from src.rag_agent.graphs.chat_agent import build_chat_agent, route_mode
from src.rag_agent.graphs.nodes import direct as direct_node_module
from src.rag_agent.graphs.nodes import mcp as mcp_node_module
from src.rag_agent.graphs.nodes import mixed as mixed_node_module
from src.rag_agent.graphs.nodes import rag as rag_node_module
from src.rag_agent.runtime.agent_server_checkpointer import LocalAsyncSqliteSaver


def _runtime(
    *, thread_id: str = "thread-123", context: dict[str, object] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        context=context,
        execution_info=SimpleNamespace(thread_id=thread_id),
    )


def _content(message: object) -> object:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def test_route_mode_reads_runtime_context() -> None:
    assert route_mode({"messages": []}, _runtime(context=None)) == "direct"
    assert route_mode({"messages": []}, _runtime(context={"mode": "direct"})) == "direct"
    assert route_mode({"messages": []}, _runtime(context={"mode": "mcp"})) == "mcp"
    assert route_mode({"messages": []}, _runtime(context={"mode": "mixed"})) == "mixed"
    assert route_mode({"messages": []}, _runtime(context={"mode": "rag"})) == "rag"


def test_route_mode_rejects_unimplemented_modes() -> None:
    with pytest.raises(NotImplementedError, match="bogus"):
        route_mode({"messages": []}, _runtime(context={"mode": "bogus"}))


def test_build_chat_agent_exposes_mixed_mode_execution_nodes() -> None:
    graph = build_chat_agent()

    node_names = set(graph.get_graph().nodes)

    assert {
        "mixed_route",
        "mixed_retrieval",
        "mixed_setup",
        "mixed_agent",
        "mixed_compose",
    } <= node_names
    assert {"mcp_setup", "mcp_agent", "mcp_compose"} <= node_names
    assert "mixed" not in node_names


def test_run_direct_node_uses_runtime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeLlm:
        model_id = "model-a"

    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    def fake_get_llm(*, model_id: str | None = None) -> FakeLlm:
        captured["model_id"] = model_id
        return FakeLlm()

    async def fake_ainvoke(llm: object, history: object, run_config: object) -> AIMessage:
        captured["llm"] = llm
        captured["history"] = history
        captured["run_config"] = run_config
        return AIMessage(content=[{"type": "text", "text": "READY"}])

    monkeypatch.setattr(direct_node_module, "get_llm", fake_get_llm)
    monkeypatch.setattr(direct_node_module, "ainvoke_llm_with_optional_config", fake_ainvoke)
    monkeypatch.setattr(
        direct_node_module,
        "emit_usage_observability",
        lambda **kwargs: (None, None),
    )

    result = asyncio.run(
        direct_node_module.run_direct_node(
            {"messages": [HumanMessage(content="hi")]},
            {},
            _runtime(
                context={
                    "mode": "direct",
                    "model_id": "model-a",
                    "enable_tracing": True,
                    "session_id": "session-direct",
                }
            ),
        )
    )

    run_config = captured["run_config"]
    assert isinstance(run_config, dict)
    assert captured["model_id"] == "model-a"
    assert run_config["configurable"]["thread_id"] == "thread-123"
    assert run_config["configurable"]["session_id"] == "session-direct"
    assert run_config["configurable"]["mode"] == "direct"
    assert run_config["configurable"]["enable_tracing"] is True
    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == [{"type": "text", "text": "READY"}]
    assert assistant.additional_kwargs["mode"] == "direct"


def test_run_rag_node_uses_runtime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    async def fake_contextualize_question(**kwargs: object) -> str:
        captured["contextualize_kwargs"] = kwargs
        return "standalone retrieve"

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        captured["synthesize_kwargs"] = kwargs
        return "RAG READY", None, "model-b"

    monkeypatch.setattr(rag_node_module, "contextualize_question", fake_contextualize_question)
    monkeypatch.setattr(
        rag_node_module.rag_runtime,
        "retrieve_oracle_docs",
        lambda **kwargs: (
            captured.__setitem__("retrieve_kwargs", kwargs)
            or [SimpleNamespace(page_content="doc", metadata={"source": "doc"})]
        ),
    )
    monkeypatch.setattr(
        rag_node_module.rag_runtime,
        "rerank_retrieved_docs",
        lambda query, docs, *, enable_reranker: (
            captured.__setitem__(
                "rerank_kwargs",
                {"query": query, "docs": docs, "enable_reranker": enable_reranker},
            )
            or docs
        ),
    )
    monkeypatch.setattr(
        rag_node_module.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer
    )
    monkeypatch.setattr(
        rag_node_module.rag_runtime, "citations_from_docs", lambda docs: [{"source": "doc"}]
    )
    monkeypatch.setattr(
        rag_node_module.rag_runtime, "serialize_docs", lambda docs: [{"id": "doc-1"}]
    )
    monkeypatch.setattr(
        rag_node_module,
        "emit_usage_observability",
        lambda **kwargs: (None, None),
    )

    result = asyncio.run(
        rag_node_module.run_rag_node(
            {"messages": [HumanMessage(content="retrieve")]},
            {},
            _runtime(
                context={
                    "mode": "rag",
                    "model_id": "model-b",
                    "collection_name": "default",
                    "enable_reranker": True,
                    "enable_tracing": False,
                    "session_id": "session-rag",
                }
            ),
        )
    )

    contextualize_kwargs = captured["contextualize_kwargs"]
    assert contextualize_kwargs["model_id"] == "model-b"
    assert captured["retrieve_kwargs"]["collection_name"] == "default"
    assert captured["retrieve_kwargs"]["query"] == "standalone retrieve"
    assert captured["rerank_kwargs"]["enable_reranker"] is True
    assert captured["synthesize_kwargs"]["model_id"] == "model-b"
    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "RAG READY"
    assert assistant.additional_kwargs["mode"] == "rag"


def test_run_rag_node_returns_assistant_error_when_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    async def fake_contextualize_question(**kwargs: object) -> str:
        _ = kwargs
        raise RuntimeError("DPY-6005: cannot connect to database")

    monkeypatch.setattr(rag_node_module, "contextualize_question", fake_contextualize_question)

    result = asyncio.run(
        rag_node_module.run_rag_node(
            {"messages": [HumanMessage(content="retrieve")]},
            {},
            _runtime(context={"mode": "rag"}),
        )
    )

    assistant = result["messages"][-1]
    assert isinstance(assistant, AIMessage)
    assert "couldn't complete the request" in str(assistant.content)
    assert assistant.additional_kwargs["mode"] == "rag"
    assert assistant.additional_kwargs["error"]["type"] == "RuntimeError"
    assert "DPY-6005" in assistant.additional_kwargs["error"]["message"]


def test_mcp_setup_records_recipe_without_runtime_turn_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    async def fake_load_adapter_tools(**kwargs: object) -> list[object]:
        captured["load_tools_kwargs"] = kwargs
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", fake_load_adapter_tools)
    monkeypatch.setattr(
        tool_agent_turn,
        "_mcp_definitions",
        lambda: {"calculator": {"transport": "stdio", "command": "calculator"}},
    )

    runtime = _runtime(
        context={
            "mode": "mcp",
            "model_id": "model-c",
            "mcp_server_keys": ["calculator"],
            "session_id": "session-mcp",
        }
    )

    async def run():
        async with LocalAsyncSqliteSaver.from_conn_string(":memory:") as saver:
            runtime.execution_info.run_id = "run-1"
            config = {
                "callbacks": ["outer-callback"],
                "metadata": {"source": "workflow-test"},
                "configurable": {"__pregel_checkpointer": saver},
            }
            await mcp_node_module.run_mcp_setup(
                {"messages": [HumanMessage(id="human-mcp", content="19 + 23")]}, config, runtime
            )
            return await saver.recipe_store.load(("thread-123", "human-mcp"))

    recipe = asyncio.run(run())
    assert recipe is not None
    assert recipe.model_key == "model-c"
    assert recipe.session_id == "session-mcp"
    assert recipe.mcp_server_keys == ("calculator",)
    assert captured["load_tools_kwargs"]["server_keys"] == ["calculator"]


def test_mixed_setup_records_recipe_without_runtime_turn_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

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
        description = "Oracle retrieval tool"

    async def fake_load_adapter_tools(**kwargs: object) -> list[object]:
        captured["load_tools_kwargs"] = kwargs
        return []

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", fake_load_adapter_tools)
    monkeypatch.setattr(
        tool_agent_turn,
        "_mcp_definitions",
        lambda: {"calculator": {"transport": "stdio", "command": "calculator"}},
    )
    monkeypatch.setattr(
        mixed_node_module.rag_runtime,
        "build_oracle_retrieval_tool",
        lambda **kwargs: (
            captured.__setitem__("retrieval_tool_kwargs", kwargs) or FakeRetrievalTool()
        ),
    )

    runtime = _runtime(
        context={
            "mode": "mixed",
            "model_id": "model-d",
            "collection_name": "ORACLE_WEB_EMBEDDINGS",
            "enable_reranker": True,
            "enable_tracing": True,
            "mcp_server_keys": ["calculator"],
            "session_id": "session-mixed",
        }
    )

    async def run():
        async with LocalAsyncSqliteSaver.from_conn_string(":memory:") as saver:
            runtime.execution_info.run_id = "run-1"
            config = {
                "callbacks": ["outer-callback"],
                "metadata": {"source": "workflow-test"},
                "configurable": {"__pregel_checkpointer": saver},
            }
            result = await mixed_node_module.run_mixed_mcp_setup(
                {"messages": [HumanMessage(id="human-mixed", content="payment terms plus 5")]},
                config,
                runtime,
            )
            return result, await saver.recipe_store.load(("thread-123", "human-mixed"))

    result, recipe = asyncio.run(run())
    assert "progress" in result
    assert recipe is not None
    assert recipe.model_key == "model-d"
    assert recipe.collection_key == "ORACLE_WEB_EMBEDDINGS"
    assert recipe.mcp_server_keys == ("calculator",)
    assert captured["retrieval_tool_kwargs"]["collection_name"] == "ORACLE_WEB_EMBEDDINGS"
    assert captured["load_tools_kwargs"]["server_keys"] == ["calculator"]


def test_build_chat_agent_preserves_messages_across_same_thread(tmp_path, monkeypatch) -> None:
    call_messages: list[list[object]] = []

    class FakeLlm:
        model_id = "model-a"

    class _FakeTrace:
        trace_context = None
        trace_id = None

    class _FakeTraceContextManager:
        def __enter__(self) -> _FakeTrace:
            return _FakeTrace()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type, exc, tb
            return False

    monkeypatch.setattr(direct_node_module, "get_llm", lambda **kwargs: FakeLlm())
    monkeypatch.setattr(
        direct_node_module,
        "emit_usage_observability",
        lambda **kwargs: (None, None),
    )

    async def fake_ainvoke(llm: object, history: object, run_config: object) -> AIMessage:
        _ = llm, run_config
        assert isinstance(history, list)
        call_messages.append(history)
        return AIMessage(content=f"reply-{len(call_messages)}")

    monkeypatch.setattr(direct_node_module, "ainvoke_llm_with_optional_config", fake_ainvoke)

    async def run() -> tuple[dict[str, object], dict[str, object]]:
        db_path = tmp_path / "chat-agent.sqlite"
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            await checkpointer.setup()
            graph = build_chat_agent(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "chat-thread"}}

            first_result = await graph.ainvoke(
                {"messages": [HumanMessage(content="hello")]},
                config,
            )
            second_result = await graph.ainvoke(
                {"messages": [HumanMessage(content="follow up")]},
                config,
            )
            return first_result, second_result

    first, second = asyncio.run(run())

    assert _content(first["messages"][-1]) == "reply-1"
    assert _content(second["messages"][-1]) == "reply-2"
    assert len(call_messages) == 2
    assert [_content(message) for message in call_messages[1]] == [
        "hello",
        "reply-1",
        "follow up",
    ]
