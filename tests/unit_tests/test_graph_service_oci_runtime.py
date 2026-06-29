from __future__ import annotations

import asyncio
import inspect
from contextlib import contextmanager
from typing import cast

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.rag_agent.graphs import mcp_policies
from src.rag_agent.runtime import chat_service as mod
from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.chat_service import ChatRuntimeService


def test_chat_runtime_service_constructor_does_not_accept_legacy_graph_arg() -> None:
    assert "graph" not in inspect.signature(ChatRuntimeService).parameters


def test_chat_runtime_service_does_not_expose_unused_delete_thread_helper() -> None:
    assert not hasattr(ChatRuntimeService, "delete_thread")


def test_chat_runtime_service_does_not_expose_wrapper_cleanup_helpers() -> None:
    assert not hasattr(ChatRuntimeService, "_build_mode_result")
    assert not hasattr(ChatRuntimeService, "_prepare_mode_run_config")
    assert not hasattr(ChatRuntimeService, "_emit_mode_usage")
    assert not hasattr(ChatRuntimeService, "_finalize_mode_result")


def test_called_tool_names_combines_tools_used_and_invocations() -> None:
    assert mcp_policies._called_tool_names(
        tools_used=["oracle_retrieval", " "],
        tool_invocations=[
            {"tool_name": "Calculator_calculate", "result": "50"},
            {"tool_name": None},
        ],
    ) == {"oracle_retrieval", "calculator_calculate"}


def test_references_from_result_preserves_stream_and_storage_shapes() -> None:
    result = {
        "standalone_question": None,
        "citations": [],
        "reranker_docs": [],
        "mcp_tools_used": [],
    }

    assert mod._references_from_result(result, include_empty_core=True) == {
        "standalone_question": None,
        "citations": [],
        "reranker_docs": [],
    }
    assert mod._references_from_result(result, include_empty_mcp_tools=True) == {
        "citations": [],
        "reranker_docs": [],
        "mcp_tools_used": [],
    }


def test_graph_service_run_chat_direct_mode_uses_oci_llm(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            captured["messages"] = messages
            return AIMessage(content="Direct OCI answer")

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_llm", lambda model_id=None: FakeLLM()
    )

    service = ChatRuntimeService()

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "How can I create visual application?"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-1",
            session_id=None,
            collection_name=None,
            enable_reranker=None,
            enable_tracing=None,
            mode="direct",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "Direct OCI answer"
    assert result["standalone_question"] == "How can I create visual application?"
    assert captured["messages"]


def test_graph_service_direct_mode_hydrates_prior_thread_messages(monkeypatch) -> None:
    captured: dict[str, object] = {}
    service = ChatRuntimeService()
    thread_id = "thread-direct-memory"
    service._thread_state[thread_id] = {
        "messages": [
            HumanMessage(content="My deployment target is production."),
            AIMessage(content="I will keep production constraints in mind."),
        ],
        "final_answer": "I will keep production constraints in mind.",
    }

    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            captured["messages"] = messages
            return AIMessage(content="Use production-safe rollout steps.")

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_llm", lambda model_id=None: FakeLLM()
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "What should I do next?"}],
            model_id="google.gemini-2.5-pro",
            thread_id=thread_id,
            session_id=None,
            collection_name=None,
            enable_reranker=None,
            enable_tracing=None,
            mode="direct",
            mcp_server_keys=None,
            stream=False,
        )
    )

    messages = cast(list[object], captured["messages"])
    assert [getattr(message, "content", "") for message in messages] == [
        "My deployment target is production.",
        "I will keep production constraints in mind.",
        "What should I do next?",
    ]
    assert result["final_answer"] == "Use production-safe rollout steps."


def test_graph_service_run_chat_defaults_to_rag_when_mode_unspecified(monkeypatch) -> None:
    service = ChatRuntimeService()
    captured: dict[str, object] = {}

    async def fake_run_rag_mode(self: ChatRuntimeService, **kwargs: object) -> dict[str, object]:
        _ = self
        captured.update(kwargs)
        return {
            "final_answer": "rag-default-answer",
            "error": None,
            "standalone_question": "What is 2+2?",
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
        }

    monkeypatch.setattr(ChatRuntimeService, "_run_rag_mode", fake_run_rag_mode)
    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-1",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode=None,
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "rag-default-answer"
    assert result["mcp_used"] is False
    assert captured["normalized_mode"] == "rag"


@pytest.mark.parametrize("mode", ["mcp", "mixed"])
def test_graph_service_run_chat_rejects_langgraph_owned_modes(mode: str) -> None:
    service = ChatRuntimeService()
    with pytest.raises(NotImplementedError, match="LangGraph"):
        asyncio.run(
            service.run_chat(
                messages=[{"role": "user", "content": "Calculate the integral of x^2 * e^x."}],
                model_id="google.gemini-2.5-pro",
                thread_id="thread-1",
                session_id=None,
                collection_name="RAG_KNOWLEDGE_BASE",
                enable_reranker=None,
                enable_tracing=None,
                mode=mode,
                mcp_server_keys=["calculator"],
                stream=False,
            )
        )


def test_graph_service_run_chat_rag_mode_uses_oracle_retrieval(monkeypatch) -> None:
    service = ChatRuntimeService()

    @contextmanager
    def fake_get_pooled_connection():
        yield object()

    def fake_search_documents(**kwargs: object) -> list[Document]:
        assert kwargs["query"] == "What is Oracle 23AI?"
        assert kwargs["top_k"] == 5
        return [
            Document(
                page_content="Oracle Database 23ai introduces AI Vector Search.",
                metadata={"source": "Doc1", "page": "1"},
            )
        ]

    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(content="Oracle 23ai introduces AI Vector Search.")

    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_pooled_connection", fake_get_pooled_connection
    )
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.get_embedding_model", lambda: object())
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.search_documents", fake_search_documents)
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_llm", lambda model_id=None: FakeLLM()
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "What is Oracle 23AI?"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-1",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="rag",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "Oracle 23ai introduces AI Vector Search."
    assert result["citations"] == [{"source": "Doc1", "page": "1", "link": None}]
    assert result["reranker_docs"] == [
        {
            "page_content": "Oracle Database 23ai introduces AI Vector Search.",
            "metadata": {"source": "Doc1", "page": "1"},
        }
    ]


def test_graph_service_rag_mode_returns_not_found_when_retrieval_has_no_docs(
    monkeypatch,
) -> None:
    service = ChatRuntimeService()

    def fake_retrieve_oracle_docs(**kwargs: object) -> list[Document]:
        assert kwargs["query"] == "How can we land on moon?"
        return []

    async def fail_if_synthesizing_without_context(**kwargs: object):
        _ = kwargs
        raise AssertionError("RAG mode must not synthesize an answer without retrieved docs")

    monkeypatch.setattr(rag_runtime, "retrieve_oracle_docs", fake_retrieve_oracle_docs)
    monkeypatch.setattr(rag_runtime, "synthesize_rag_answer", fail_if_synthesizing_without_context)

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "How can we land on moon?"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-rag-empty",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="rag",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "I don't know the answer from the selected Oracle collection."
    assert result["citations"] == []
    assert result["reranker_docs"] == []
    assert result["context_usage"] is None


def test_graph_service_rag_mode_uses_native_reranker_when_enabled(monkeypatch) -> None:
    service = ChatRuntimeService()
    docs = [
        Document(page_content="Less relevant chunk", metadata={"source": "Doc1"}),
        Document(page_content="Best matching chunk", metadata={"source": "Doc2"}),
    ]
    captured: dict[str, object] = {}

    @contextmanager
    def fake_get_pooled_connection():
        yield object()

    def fake_search_documents(**kwargs: object) -> list[Document]:
        _ = kwargs
        return docs

    def fake_rerank_documents(query: str, input_docs: list[Document]) -> list[Document]:
        captured["query"] = query
        captured["docs"] = input_docs
        return [docs[1]]

    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            _ = messages
            return AIMessage(content="Best matching chunk")

    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_pooled_connection", fake_get_pooled_connection
    )
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.get_embedding_model", lambda: object())
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.search_documents", fake_search_documents)
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.oci_rerank_documents", fake_rerank_documents
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_llm", lambda model_id=None: FakeLLM()
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "What is the best chunk?"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-rerank",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=True,
            enable_tracing=None,
            mode="rag",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert captured == {"query": "What is the best chunk?", "docs": docs}
    assert result["reranker_docs"] == [
        {"page_content": "Best matching chunk", "metadata": {"source": "Doc2"}}
    ]


def test_graph_service_rag_mode_contextualizes_followup_before_retrieval(monkeypatch) -> None:
    service = ChatRuntimeService()
    thread_id = "thread-rag-memory"
    captured: dict[str, object] = {}
    service._thread_state[thread_id] = {
        "messages": [
            HumanMessage(content="Tell me about Oracle Database 23ai."),
            AIMessage(content="Oracle Database 23ai includes AI Vector Search."),
        ],
        "final_answer": "Oracle Database 23ai includes AI Vector Search.",
    }

    @contextmanager
    def fake_get_pooled_connection():
        yield object()

    def fake_search_documents(**kwargs: object) -> list[Document]:
        captured["query"] = kwargs["query"]
        return [
            Document(
                page_content="AI Vector Search supports semantic search in Oracle Database 23ai.",
                metadata={"source": "Doc1"},
            )
        ]

    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list[object]) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                captured["contextualize_prompt"] = getattr(messages[0], "content", "")
                return AIMessage(content="How does AI Vector Search work in Oracle Database 23ai?")
            return AIMessage(content="AI Vector Search supports semantic search.")

    fake_llm = FakeLLM()
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_pooled_connection", fake_get_pooled_connection
    )
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.get_embedding_model", lambda: object())
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.search_documents", fake_search_documents)
    monkeypatch.setattr("src.rag_agent.runtime.memory.get_llm", lambda model_id=None: fake_llm)
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.get_llm", lambda model_id=None: fake_llm)

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "How does that work?"}],
            model_id="google.gemini-2.5-pro",
            thread_id=thread_id,
            session_id=None,
            collection_name=None,
            enable_reranker=None,
            enable_tracing=None,
            mode="rag",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert captured["query"] == "How does AI Vector Search work in Oracle Database 23ai?"
    assert "Oracle Database 23ai includes AI Vector Search." in str(
        captured["contextualize_prompt"]
    )
    assert (
        result["standalone_question"] == "How does AI Vector Search work in Oracle Database 23ai?"
    )


def test_graph_service_run_chat_does_not_apply_custom_transform_prepass(monkeypatch) -> None:
    service = ChatRuntimeService()
    thread_id = "thread-transform"

    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            _ = messages
            return AIMessage(content="Direct response path")

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_llm", lambda model_id=None: FakeLLM()
    )

    service._thread_state[thread_id] = {
        "messages": [
            HumanMessage(content="Can I rename an existing application to a non-unique name?"),
            AIMessage(content="The application name must be unique in the identity domain."),
        ],
        "final_answer": "The application name must be unique in the identity domain.",
    }

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "give me answer in bullet points"}],
            model_id="google.gemini-2.5-pro",
            thread_id=thread_id,
            session_id=None,
            collection_name=None,
            enable_reranker=None,
            enable_tracing=None,
            mode="direct",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "Direct response path"


def test_graph_service_citations_use_realistic_oracle_metadata_keys() -> None:
    docs = [
        Document(
            page_content="Oracle Visual Builder lets you create applications visually.",
            metadata={
                "link": "https://docs.oracle.com/en/cloud/paas/app-builder-cloud/visual-applications.html",
                "title": "Visual Applications",
                "page_number": 7,
            },
        ),
        Document(
            page_content="Oracle APEX includes App Builder.",
            metadata={
                "url": "https://docs.oracle.com/en/database/oracle/apex/",
                "document_name": "APEX App Builder Guide",
            },
        ),
        Document(
            page_content="Visual applications are stored as metadata.",
            metadata={
                "file_name": "visual_applications.md",
                "file_path": "/docs/visual/visual_applications.md",
                "source_url": "https://docs.oracle.com/en/cloud/paas/visual-builder/visual-applications/",
            },
        ),
    ]

    citations = rag_runtime.citations_from_docs(docs)

    assert citations == [
        {
            "source": "Visual Applications",
            "page": "7",
            "link": "https://docs.oracle.com/en/cloud/paas/app-builder-cloud/visual-applications.html",
        },
        {
            "source": "APEX App Builder Guide",
            "page": None,
            "link": "https://docs.oracle.com/en/database/oracle/apex/",
        },
        {
            "source": "visual_applications.md",
            "page": None,
            "link": "https://docs.oracle.com/en/cloud/paas/visual-builder/visual-applications/",
        },
    ]


def test_filter_retrieved_docs_prefers_query_term_overlap() -> None:
    docs = [
        Document(
            page_content="How to configure OCI CLI on Linux",
            metadata={"source": "OCI CLI Guide"},
        ),
        Document(
            page_content="Create applications in Oracle Visual Builder quickly",
            metadata={"source": "Visual Builder Guide"},
        ),
    ]

    filtered = rag_runtime.filter_retrieved_docs(
        "how can i create applications in visual builder",
        docs,
    )

    assert len(filtered) == 1
    assert filtered[0].metadata.get("source") == "Visual Builder Guide"
