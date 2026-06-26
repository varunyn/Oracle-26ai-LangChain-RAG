import asyncio
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.rag_agent.graphs.chat_agent import build_chat_agent, route_mode
from src.rag_agent.graphs.nodes import direct as direct_node_module
from src.rag_agent.graphs.nodes import rag as rag_node_module


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
    assert route_mode({"messages": []}, _runtime(context={"mode": "rag"})) == "rag"


def test_route_mode_rejects_unimplemented_modes() -> None:
    with pytest.raises(NotImplementedError, match="mcp"):
        route_mode({"messages": []}, _runtime(context={"mode": "mcp"}))

    with pytest.raises(NotImplementedError, match="mixed"):
        route_mode({"messages": []}, _runtime(context={"mode": "mixed"}))


def test_run_direct_node_uses_runtime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class StubService:
        async def run_chat(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "final_answer": "READY",
                "citations": [],
                "reranker_docs": [],
                "context_usage": None,
                "mcp_used": False,
                "mcp_tools_used": [],
            }

    monkeypatch.setattr(direct_node_module, "ChatRuntimeService", StubService)

    result = asyncio.run(
        direct_node_module.run_direct_node(
            {"messages": [{"role": "user", "content": "hi"}]},
            _runtime(
                context={"mode": "direct", "model_id": "model-a", "enable_tracing": True}
            ),
        )
    )

    assert captured["model_id"] == "model-a"
    assert captured["thread_id"] == "thread-123"
    assert captured["mode"] == "direct"
    assert captured["enable_tracing"] is True
    assert result["messages"][-1]["content"] == "READY"
    assert result["references"]["mode"] == "direct"


def test_run_rag_node_uses_runtime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class StubService:
        async def run_chat(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "final_answer": "RAG READY",
                "citations": [{"source": "doc"}],
                "reranker_docs": [{"id": "doc-1"}],
                "context_usage": {"retrieved_docs_count": 1},
                "mcp_used": False,
                "mcp_tools_used": [],
            }

    monkeypatch.setattr(rag_node_module, "ChatRuntimeService", StubService)

    result = asyncio.run(
        rag_node_module.run_rag_node(
            {"messages": [{"role": "user", "content": "retrieve"}]},
            _runtime(
                context={
                    "mode": "rag",
                    "model_id": "model-b",
                    "collection_name": "default",
                    "enable_reranker": True,
                    "enable_tracing": False,
                }
            ),
        )
    )

    assert captured["model_id"] == "model-b"
    assert captured["thread_id"] == "thread-123"
    assert captured["collection_name"] == "default"
    assert captured["enable_reranker"] is True
    assert captured["mode"] == "rag"
    assert result["messages"][-1]["content"] == "RAG READY"
    assert result["references"]["mode"] == "rag"


def test_build_chat_agent_preserves_messages_across_same_thread(tmp_path, monkeypatch) -> None:
    call_messages: list[list[dict[str, object]]] = []

    class StubService:
        async def run_chat(self, **kwargs: object) -> dict[str, object]:
            messages = kwargs["messages"]
            assert isinstance(messages, list)
            call_messages.append(messages)
            final_answer = f"reply-{len(call_messages)}"
            return {
                "final_answer": final_answer,
                "citations": [],
                "reranker_docs": [],
                "context_usage": None,
                "mcp_used": False,
                "mcp_tools_used": [],
            }

    monkeypatch.setattr(direct_node_module, "ChatRuntimeService", StubService)

    async def run() -> tuple[dict[str, object], dict[str, object]]:
        db_path = tmp_path / "chat-agent.sqlite"
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            await checkpointer.setup()
            graph = build_chat_agent(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "chat-thread"}}

            first_result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "hello"}]},
                config,
            )
            second_result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "follow up"}]},
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


def test_build_chat_agent_dedupes_same_thread_full_transcript_replay(tmp_path, monkeypatch) -> None:
    call_messages: list[list[dict[str, object]]] = []

    class StubService:
        async def run_chat(self, **kwargs: object) -> dict[str, object]:
            messages = kwargs["messages"]
            assert isinstance(messages, list)
            call_messages.append(messages)
            final_answer = f"reply-{len(call_messages)}"
            return {
                "final_answer": final_answer,
                "citations": [],
                "reranker_docs": [],
                "context_usage": None,
                "mcp_used": False,
                "mcp_tools_used": [],
            }

    monkeypatch.setattr(direct_node_module, "ChatRuntimeService", StubService)

    async def run() -> None:
        db_path = tmp_path / "chat-agent-replay.sqlite"
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            await checkpointer.setup()
            graph = build_chat_agent(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "chat-thread"}}

            first = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "hello"}]},
                config,
            )
            await graph.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": _content(first["messages"][-1])},
                        {"role": "user", "content": "follow up"},
                    ]
                },
                config,
            )

    asyncio.run(run())

    assert len(call_messages) == 2
    assert [_content(message) for message in call_messages[1]] == [
        "hello",
        "reply-1",
        "follow up",
    ]
