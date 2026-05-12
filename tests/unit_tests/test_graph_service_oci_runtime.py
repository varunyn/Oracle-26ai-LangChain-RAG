from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from api.services.graph_service import ChatRuntimeService


def test_graph_service_run_chat_direct_mode_uses_oci_llm(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            captured["messages"] = messages
            return AIMessage(content="Direct OCI answer")

    monkeypatch.setattr("api.services.graph_service.get_llm", lambda model_id=None: FakeLLM())

    service = ChatRuntimeService(graph=object())

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


def test_graph_service_run_chat_defaults_to_mixed_when_mcp_enabled(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    captured: dict[str, object] = {}

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    monkeypatch.setattr(
        "api.services.graph_service._resolve_effective_mode",
        lambda mode: "mixed",
    )

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        captured["tool_server_keys"] = server_keys
        _ = run_config
        return [calculator_tool]

    async def fake_get_mcp_answer_async(
        question: str,
        *,
        model_id: str | None = None,
        tools: list[object] | None = None,
        run_config: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[str, list[str], list[object]]:
        captured["question"] = question
        captured["model_id"] = model_id
        captured["tools"] = tools
        captured["run_config"] = run_config
        _ = kwargs
        return ("mixed-default-answer", ["calculator_tool"], [])

    monkeypatch.setattr("api.services.graph_service.get_mcp_tools_async", fake_get_mcp_tools_async)
    monkeypatch.setattr("api.services.graph_service.get_mcp_answer_async", fake_get_mcp_answer_async)
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )

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

    assert result["final_answer"] == "mixed-default-answer"
    assert result["mcp_used"] is True
    tool_names = [tool_obj.name for tool_obj in cast(list[object], captured["tools"])]
    assert "retrieval_tool" in tool_names
    assert "calculator_tool" in tool_names


def test_graph_service_run_chat_mcp_mode_uses_mcp_answer_async(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    captured: dict[str, object] = {}

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    async def fake_get_mcp_answer_async(
        question: str,
        *,
        model_id: str | None = None,
        tools: list[object] | None = None,
        run_config: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[str, list[str], list[object]]:
        captured["require_tool_call"] = kwargs.get("require_tool_call")
        captured["question"] = question
        captured["model_id"] = model_id
        captured["tools"] = tools
        captured["run_config"] = run_config
        return ("The integral is (x^2 - 2x + 2)e^x + C.", ["calculator_tool"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        captured["tool_server_keys"] = server_keys
        _ = run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Calculate the integral of x^2 * e^x."}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-1",
            session_id=None,
            collection_name=None,
            enable_reranker=None,
            enable_tracing=None,
            mode="mcp",
            mcp_server_keys=["calculator"],
            stream=False,
        )
    )

    assert result["final_answer"] == "The integral is (x^2 - 2x + 2)e^x + C."
    assert result["mcp_used"] is True
    assert captured["question"] == "Calculate the integral of x^2 * e^x."
    assert captured["model_id"] == "google.gemini-2.5-pro"
    tool_names = [tool_obj.name for tool_obj in cast(list[object], captured["tools"])]
    assert tool_names == ["calculator_tool"]
    assert captured["tool_server_keys"] == ["calculator"]
    assert captured["require_tool_call"] is None
    assert result["mcp_tools_used"] == ["calculator_tool"]
    assert captured["run_config"] == {
        "configurable": {"thread_id": "thread-1", "mcp_server_keys": ["calculator"]}
    }


def test_graph_service_run_chat_rag_mode_uses_oracle_retrieval(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())

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
            return AIMessage(content="Oracle 23ai introduces AI Vector Search. [1]")

    monkeypatch.setattr("api.services.graph_service.get_pooled_connection", fake_get_pooled_connection)
    monkeypatch.setattr("api.services.graph_service.get_embedding_model", lambda: object())
    monkeypatch.setattr("api.services.graph_service.search_documents", fake_search_documents)
    monkeypatch.setattr("api.services.graph_service.get_llm", lambda model_id=None: FakeLLM())

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

    assert result["final_answer"] == "Oracle 23ai introduces AI Vector Search. [1]"
    assert result["citations"] == [{"source": "Doc1", "page": "1", "link": None}]
    assert result["reranker_docs"] == [
        {
            "page_content": "Oracle Database 23ai introduces AI Vector Search.",
            "metadata": {"source": "Doc1", "page": "1"},
        }
    ]


def test_graph_service_run_chat_mixed_mode_uses_mcp_answer_async(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    captured: dict[str, object] = {}

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    async def fake_get_mcp_answer_async(
        question: str,
        *,
        model_id: str | None = None,
        tools: list[object] | None = None,
        run_config: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[str, list[str], list[object]]:
        captured["require_tool_call"] = kwargs.get("require_tool_call")
        captured["question"] = question
        captured["model_id"] = model_id
        captured["tools"] = tools
        captured["run_config"] = run_config
        return ("The integral is (x^2 - 2x + 2)e^x + C.", ["calculator_tool"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        captured["tool_server_keys"] = server_keys
        _ = run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(ChatRuntimeService, "_build_oracle_retrieval_tool", lambda self, collection_name=None: retrieval_tool)
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Calculate the integral of x^2 * e^x."}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-1",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=["calculator"],
            stream=False,
        )
    )

    assert result["final_answer"] == "The integral is (x^2 - 2x + 2)e^x + C."
    assert captured["question"] == "Calculate the integral of x^2 * e^x."
    assert captured["model_id"] == "google.gemini-2.5-pro"
    tool_names = [tool_obj.name for tool_obj in cast(list[object], captured["tools"])]
    assert "retrieval_tool" in tool_names
    assert "calculator_tool" in tool_names
    assert captured["tool_server_keys"] == ["calculator"]
    assert captured["require_tool_call"] is None
    assert result["mcp_tools_used"] == ["calculator_tool"]


def test_graph_service_run_chat_mcp_mode_uses_all_servers_when_not_specified(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    captured: dict[str, object] = {}

    @tool
    def routed_tool(command: str) -> str:
        """Run routed MCP tool command."""
        return command

    async def fake_get_mcp_answer_async(
        question: str,
        *,
        model_id: str | None = None,
        tools: list[object] | None = None,
        run_config: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[str, list[str], list[object]]:
        _ = question, model_id, tools
        captured["require_tool_call"] = kwargs.get("require_tool_call")
        captured["run_config"] = run_config
        return ("ok", ["routed_tool"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        captured["tool_server_keys"] = server_keys
        _ = run_config
        return [routed_tool]

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Find my tenancy namespace"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-1",
            session_id=None,
            collection_name=None,
            enable_reranker=None,
            enable_tracing=None,
            mode="mcp",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "ok"
    assert captured["tool_server_keys"] is None
    assert captured["require_tool_call"] is None
    assert captured["run_config"]
    configurable = cast(dict[str, object], captured["run_config"])["configurable"]
    assert cast(dict[str, object], configurable)["thread_id"] == "thread-1"
    assert "mcp_server_keys" not in cast(dict[str, object], configurable)


def test_graph_service_mixed_mode_invokes_mcp_answer_per_request(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    call_count = {"value": 0}

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args
        _ = kwargs
        call_count["value"] += 1
        return ("ok", [], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(ChatRuntimeService, "_build_oracle_retrieval_tool", lambda self, collection_name=None: retrieval_tool)
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Calculate the integral of x^2 * e^x."}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-1",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=["calculator"],
            stream=False,
        )
    )
    asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Calculate the integral of x^2 * e^x."}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-2",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=["calculator"],
            stream=False,
        )
    )

    assert call_count["value"] == 2


def test_graph_service_mixed_mode_includes_retrieval_references_when_available(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())

    class _Tool:
        name = "oracle_retrieval"
        description = "Retrieve Oracle documentation context for a question."
        _retrieval_state = {
            "docs": [
                Document(
                    page_content="OCI Namespace docs",
                    metadata={"source": "OCI Doc", "page": "2"},
                )
            ]
        }

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args, kwargs
        return ("namespace is xyz", ["oracle_retrieval"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(ChatRuntimeService, "_build_oracle_retrieval_tool", lambda self, collection_name=None: _Tool())
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Find OCI namespace docs"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-refs",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["citations"] == [{"source": "OCI Doc", "page": "2", "link": None}]
    assert result["reranker_docs"] == [
        {"page_content": "OCI Namespace docs", "metadata": {"source": "OCI Doc", "page": "2"}}
    ]
    assert result["context_usage"] == {"retrieved_docs_count": 1}


def test_graph_service_mixed_mode_falls_back_to_direct_retrieval_for_citations(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())

    class _Tool:
        name = "oracle_retrieval"
        description = "Retrieve Oracle documentation context for a question."
        _retrieval_state = {"docs": []}

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args, kwargs
        return ("namespace is xyz", ["oracle_retrieval"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    docs = [
        Document(page_content="VB doc content", metadata={"source": "VB Guide", "page": "1"}),
    ]

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(ChatRuntimeService, "_build_oracle_retrieval_tool", lambda self, collection_name=None: _Tool())
    monkeypatch.setattr(ChatRuntimeService, "_retrieve_oracle_docs", lambda self, **kwargs: docs)
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "How can I create visual applications?"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-refs-fallback",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["citations"] == [{"source": "VB Guide", "page": "1", "link": None}]
    assert result["reranker_docs"] == [{"page_content": "VB doc content", "metadata": {"source": "VB Guide", "page": "1"}}]


def test_graph_service_mixed_mode_keeps_non_retrieval_mcp_answer_without_rag_override(
    monkeypatch,
) -> None:
    service = ChatRuntimeService(graph=object())

    class _Tool:
        name = "oracle_retrieval"
        description = "Retrieve Oracle documentation context for a question."
        _retrieval_state = {"docs": []}

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args, kwargs
        return ("x = 6", ["calculator_tool"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    def fail_if_retrieval_called(self, **kwargs):
        _ = self, kwargs
        raise AssertionError("RAG fallback should not run when non-retrieval MCP tools were used")

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService, "_build_oracle_retrieval_tool", lambda self, collection_name=None: _Tool()
    )
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(ChatRuntimeService, "_retrieve_oracle_docs", fail_if_retrieval_called)

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Solve: 5(x-2)=20"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-calculator",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=["calculator"],
            stream=False,
        )
    )

    assert result["final_answer"] == "x = 6"
    assert result["mcp_tools_used"] == ["calculator_tool"]
    assert result["citations"] == []
    assert result["reranker_docs"] == []
    assert result["context_usage"] is None


def test_graph_service_mixed_mode_retries_with_required_tool_call_when_mcp_tools_explicitly_referenced(
    monkeypatch,
) -> None:
    service = ChatRuntimeService(graph=object())
    call_kwargs: list[dict[str, object]] = []

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool("oic_LIST_DOCUMENTS")
    def oic_list_documents(folder_name: str) -> str:
        """List invoice documents."""
        return folder_name

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args
        call_kwargs.append(cast(dict[str, object], kwargs))
        require_tool_call = bool(kwargs.get("require_tool_call"))
        if not require_tool_call:
            return ("I don't know.", [], [])
        return ("MCP tool call required but none was produced after retry. Please try again.", [], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [oic_list_documents]

    def fail_if_rag_fallback_called(self, **kwargs):
        _ = self, kwargs
        raise AssertionError("RAG fallback must not run when explicit MCP tool reference is present")

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )
    monkeypatch.setattr(ChatRuntimeService, "_retrieve_oracle_docs", fail_if_rag_fallback_called)

    result = asyncio.run(
        service.run_chat(
            messages=[
                {
                    "role": "user",
                    "content": "Use oic_LIST_DOCUMENTS and process invoices end to end.",
                }
            ],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-explicit-mcp",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert len(call_kwargs) == 2
    assert call_kwargs[0].get("require_tool_call") is None
    assert call_kwargs[1].get("require_tool_call") is True
    assert result["final_answer"] == "MCP tool call required but none was produced after retry. Please try again."
    assert result["mcp_tools_used"] == []


def test_graph_service_mixed_mode_enforces_generic_workflow_policy_when_activated(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())

    class _Settings:
        MCP_WORKFLOW_POLICY = {
            "enabled": True,
            "apply_modes": ["mixed"],
            "activation_terms": ["invoice"],
            "required_capabilities": ["classify", "extract", "create"],
            "tool_capability_map": {
                "oic_CLASSIFY_DOCUMENT": ["classify"],
                "oic_EXTRACT_INVOICE_DATA": ["extract"],
                "oic_CREATE_INVOICE": ["create"],
            },
        }

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool("oic_CLASSIFY_DOCUMENT")
    def oic_classify_document(file_name: str, file_path: str) -> str:
        """Classify document."""
        return f"{file_name}@{file_path}"

    @tool("oic_EXTRACT_INVOICE_DATA")
    def oic_extract_invoice_data(file_name: str, file_path: str) -> str:
        """Extract invoice fields."""
        return f"{file_name}@{file_path}"

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[dict[str, object]]]:
        _ = args, kwargs
        return (
            "partial summary",
            ["oic_CLASSIFY_DOCUMENT", "oic_EXTRACT_INVOICE_DATA"],
            [
                {"tool_name": "oic_CLASSIFY_DOCUMENT", "args": {}, "result": "ok"},
                {"tool_name": "oic_EXTRACT_INVOICE_DATA", "args": {}, "result": "ok"},
            ],
        )

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [oic_classify_document, oic_extract_invoice_data]

    monkeypatch.setattr("api.services.graph_service.get_settings", lambda: _Settings())
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Process this invoice workflow end-to-end."}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-workflow-policy",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert isinstance(result["error"], str)
    assert "missing required steps: create" in str(result["error"]).lower()
    assert result["final_answer"] == result["error"]


def test_graph_service_mixed_mode_workflow_policy_does_not_apply_when_not_activated(
    monkeypatch,
) -> None:
    service = ChatRuntimeService(graph=object())

    class _Settings:
        MCP_WORKFLOW_POLICY = {
            "enabled": True,
            "apply_modes": ["mixed"],
            "activation_terms": ["invoice"],
            "required_capabilities": ["classify", "extract"],
            "tool_capability_map": {
                "calculator_tool": ["calculate"],
            },
        }

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate expression."""
        return expression

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[dict[str, object]]]:
        _ = args, kwargs
        return ("x = 6", ["calculator_tool"], [{"tool_name": "calculator_tool", "args": {}, "result": "x=6"}])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr("api.services.graph_service.get_settings", lambda: _Settings())
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Solve: 5(x-2)=20"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-workflow-policy-inactive",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "x = 6"
    assert result["error"] is None


def test_graph_service_mixed_mode_replaces_trivial_answer_with_tool_failure_summary(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool("oic_CREATE_INVOICE")
    def oic_create_invoice(payload: str) -> str:
        """Create invoice."""
        return payload

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[dict[str, object]]]:
        _ = args, kwargs
        return (
            ".",
            ["oic_CREATE_INVOICE"],
            [
                {
                    "tool_name": "oic_CREATE_INVOICE",
                    "args": {"InvoiceData": {}},
                    "result": "Tool 'oic_CREATE_INVOICE' failed after 2 attempts with ToolException: validation error",
                }
            ],
        )

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [oic_create_invoice]

    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "api.services.graph_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Process invoice in mixed mode."}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-trivial-dot-answer",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert isinstance(result["final_answer"], str)
    assert "tool execution failed" in str(result["final_answer"]).lower()
    assert "oic_CREATE_INVOICE" in str(result["final_answer"])
    assert result["error"] == result["final_answer"]


def test_graph_service_run_chat_does_not_apply_custom_transform_prepass(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    thread_id = "thread-transform"

    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            _ = messages
            return AIMessage(content="Direct response path")

    monkeypatch.setattr("api.services.graph_service.get_llm", lambda model_id=None: FakeLLM())

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
    service = ChatRuntimeService(graph=object())
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

    citations = service._citations_from_docs(docs)

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
    service = ChatRuntimeService(graph=object())
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

    filtered = service._filter_retrieved_docs(
        "how can i create applications in visual builder",
        docs,
    )

    assert len(filtered) == 1
    assert filtered[0].metadata.get("source") == "Visual Builder Guide"
