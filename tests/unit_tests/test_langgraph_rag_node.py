from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs.nodes import rag


def test_rag_node_retrieves_and_returns_citations_without_chat_runtime_service(
    monkeypatch,
) -> None:
    docs = [Document(page_content="Payment is due in 45 days.", metadata={"source": "terms.md"})]

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
        return "What are the payment terms?"

    async def fake_synthesize_rag_answer(**kwargs: object) -> tuple[str, None, str]:
        _ = kwargs
        return "Payment is due in 45 days.", None, "fake-rag-model"

    monkeypatch.setattr(rag, "contextualize_question", fake_contextualize_question)
    monkeypatch.setattr(rag.rag_runtime, "retrieve_oracle_docs", lambda **kwargs: docs)
    monkeypatch.setattr(
        rag.rag_runtime,
        "rerank_retrieved_docs",
        lambda query, docs, *, enable_reranker: docs,
    )
    monkeypatch.setattr(rag.rag_runtime, "synthesize_rag_answer", fake_synthesize_rag_answer)
    monkeypatch.setattr(rag.rag_runtime, "citations_from_docs", lambda docs: [{"source": "terms.md"}])
    monkeypatch.setattr(
        rag.rag_runtime,
        "serialize_docs",
        lambda docs: [{"source": "terms.md", "content": "Payment is due in 45 days."}],
    )
    monkeypatch.setattr(rag, "emit_usage_observability", lambda **kwargs: (None, None))
    monkeypatch.setattr(rag, "start_langfuse_chat_trace", lambda **kwargs: _FakeTraceContextManager())

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
            {"messages": [HumanMessage(content="What are the payment terms?")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Payment is due in 45 days."
    assert assistant.additional_kwargs["mode"] == "rag"
    assert assistant.additional_kwargs["citations"] == [{"source": "terms.md"}]
