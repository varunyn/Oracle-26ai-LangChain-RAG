import asyncio
from types import SimpleNamespace

import pytest

from src.rag_agent.graphs.chat_agent import route_mode
from src.rag_agent.graphs.nodes import direct as direct_node_module
from src.rag_agent.graphs.nodes import rag as rag_node_module


def _runtime(*, thread_id: str = "thread-123", **context: object) -> SimpleNamespace:
    return SimpleNamespace(
        context=context,
        execution_info=SimpleNamespace(thread_id=thread_id),
    )


def test_route_mode_reads_runtime_context() -> None:
    assert route_mode({"messages": []}, _runtime(mode="direct")) == "direct"
    assert route_mode({"messages": []}, _runtime(mode="rag")) == "rag"


def test_route_mode_rejects_unimplemented_modes() -> None:
    with pytest.raises(NotImplementedError, match="mcp"):
        route_mode({"messages": []}, _runtime(mode="mcp"))

    with pytest.raises(NotImplementedError, match="mixed"):
        route_mode({"messages": []}, _runtime(mode="mixed"))


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
            _runtime(mode="direct", model_id="model-a", enable_tracing=True),
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
                mode="rag",
                model_id="model-b",
                collection_name="default",
                enable_reranker=True,
                enable_tracing=False,
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
