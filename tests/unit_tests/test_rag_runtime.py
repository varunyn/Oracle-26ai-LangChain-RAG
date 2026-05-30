from __future__ import annotations

import asyncio
from types import TracebackType

from langchain_core.documents import Document

from src.rag_agent.runtime import rag_runtime


class FakeSettings:
    RAG_RETRIEVAL_TOP_K = 3


class FakeConnectionContext:
    def __enter__(self) -> object:
        return object()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def test_oracle_retrieval_tool_returns_error_content_when_vector_search_fails(
    monkeypatch,
) -> None:
    def failing_search_documents(**kwargs: object) -> list[object]:
        _ = kwargs
        raise RuntimeError("Failed due to a DB error: ORA-22275: invalid LOB locator specified")

    monkeypatch.setattr(rag_runtime, "get_pooled_connection", lambda: FakeConnectionContext())
    monkeypatch.setattr(rag_runtime, "get_embedding_model", lambda: object())
    monkeypatch.setattr(rag_runtime, "search_documents", failing_search_documents)

    tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name="ORACLE_WEB_EMBEDDINGS",
        filter_docs=lambda query, docs: docs,
    )

    content = tool.invoke({"query": "Northway Solutions payment terms"})

    assert "Oracle retrieval failed" in content
    assert "ORA-22275" in content
    assert getattr(tool, "_retrieval_state") == {
        "docs": [],
        "error": "Failed due to a DB error: ORA-22275: invalid LOB locator specified",
    }


def test_oracle_retrieval_tool_uses_configured_top_k(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_search_documents(**kwargs: object) -> list[Document]:
        captured.update(kwargs)
        return [Document(page_content="Payment terms are Net 30.")]

    monkeypatch.setattr("api.settings.get_settings", lambda: FakeSettings())
    monkeypatch.setattr(rag_runtime, "get_pooled_connection", lambda: FakeConnectionContext())
    monkeypatch.setattr(rag_runtime, "get_embedding_model", lambda: object())
    monkeypatch.setattr(rag_runtime, "search_documents", fake_search_documents)

    tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name="ORACLE_WEB_EMBEDDINGS",
        filter_docs=lambda query, docs: docs,
    )

    content = tool.invoke({"query": "Northway Solutions payment terms"})

    assert "Payment terms are Net 30" in content
    assert captured["top_k"] == 3


def test_stream_rag_answer_closes_owned_llm(monkeypatch) -> None:
    class FakeStreamingLLM:
        model_id = "fake-stream-model"

        def __init__(self) -> None:
            self.closed = False

        async def astream(self, messages: object, config: object | None = None):
            _ = messages, config
            yield type("Chunk", (), {"content": "Hello"})()
            yield type("Chunk", (), {"content": " world"})()

        async def aclose(self) -> None:
            self.closed = True

    fake_llm = FakeStreamingLLM()
    monkeypatch.setattr(rag_runtime, "get_llm", lambda model_id=None: fake_llm)

    async def collect() -> list[tuple[str, str]]:
        return [
            (text, model_id)
            async for text, _chunk, model_id in rag_runtime.stream_rag_answer(
                question="What are the payment terms?",
                docs=[Document(page_content="Payment terms are net 30.")],
                model_id=None,
                run_config={"configurable": {"thread_id": "thread-rag-close"}},
            )
        ]

    chunks = asyncio.run(collect())

    assert chunks == [("Hello", "fake-stream-model"), (" world", "fake-stream-model")]
    assert fake_llm.closed is True
