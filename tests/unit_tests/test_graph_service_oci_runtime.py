from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.rag_agent.runtime import chat_service as mod
from src.rag_agent.runtime.chat_service import ChatRuntimeService


def test_called_tool_names_combines_tools_used_and_invocations() -> None:
    assert mod._called_tool_names(
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


def test_graph_service_direct_mode_hydrates_prior_thread_messages(monkeypatch) -> None:
    captured: dict[str, object] = {}
    service = ChatRuntimeService(graph=object())
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
        "src.rag_agent.runtime.chat_service._resolve_effective_mode",
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

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async", fake_get_mcp_tools_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async", fake_get_mcp_answer_async
    )
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
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type("Settings", (), {"REQUIRE_TOOL_CALL": False, "MCP_WORKFLOW_POLICY": {}})(),
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
    assert captured["require_tool_call"] is False
    assert result["mcp_tools_used"] == ["calculator_tool"]
    assert captured["run_config"] == {
        "configurable": {"thread_id": "thread-1", "mcp_server_keys": ["calculator"]}
    }


def test_graph_service_mcp_mode_passes_prior_thread_history_to_agent(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    thread_id = "thread-mcp-memory"
    captured: dict[str, object] = {}
    service._thread_state[thread_id] = {
        "messages": [
            HumanMessage(content="Use the calculator for arithmetic."),
            AIMessage(content="Understood."),
        ],
        "final_answer": "Understood.",
    }

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    async def fake_get_mcp_answer_async(
        question: str,
        *,
        chat_history: list[object] | None = None,
        **kwargs: object,
    ) -> tuple[str, list[str], list[object]]:
        _ = kwargs
        captured["question"] = question
        captured["chat_history"] = chat_history
        return ("4", ["calculator_tool"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async", fake_get_mcp_tools_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async", fake_get_mcp_answer_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type("Settings", (), {"REQUIRE_TOOL_CALL": False, "MCP_WORKFLOW_POLICY": {}})(),
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            model_id="google.gemini-2.5-pro",
            thread_id=thread_id,
            session_id=None,
            collection_name=None,
            enable_reranker=None,
            enable_tracing=None,
            mode="mcp",
            mcp_server_keys=["calculator"],
            stream=False,
        )
    )

    history = cast(list[object], captured["chat_history"])
    assert captured["question"] == "What is 2+2?"
    assert [getattr(message, "content", "") for message in history] == [
        "Use the calculator for arithmetic.",
        "Understood.",
    ]
    assert result["final_answer"] == "4"


def test_graph_service_mcp_mode_honors_require_tool_call_setting(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    captured: dict[str, object] = {}

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a math expression."""
        return f"calculated: {expression}"

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args
        captured["require_tool_call"] = kwargs.get("require_tool_call")
        return ("ok", ["calculator_tool"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type("Settings", (), {"REQUIRE_TOOL_CALL": True, "MCP_WORKFLOW_POLICY": {}})(),
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async", fake_get_mcp_tools_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async", fake_get_mcp_answer_async
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

    assert result["final_answer"] == "ok"
    assert captured["require_tool_call"] is True


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

    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_pooled_connection", fake_get_pooled_connection
    )
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.get_embedding_model", lambda: object())
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.search_documents", fake_search_documents
    )
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

    assert result["final_answer"] == "Oracle 23ai introduces AI Vector Search. [1]"
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
    service = ChatRuntimeService(graph=object())

    def fake_retrieve_oracle_docs(**kwargs: object) -> list[Document]:
        assert kwargs["query"] == "How can we land on moon?"
        return []

    async def fail_if_synthesizing_without_context(self: object, **kwargs: object):
        _ = self, kwargs
        raise AssertionError("RAG mode must not synthesize an answer without retrieved docs")

    monkeypatch.setattr(
        ChatRuntimeService,
        "_retrieve_oracle_docs",
        lambda self, **kwargs: fake_retrieve_oracle_docs(**kwargs),
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_synthesize_rag_answer",
        fail_if_synthesizing_without_context,
    )

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
    service = ChatRuntimeService(graph=object())
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
            return AIMessage(content="Best matching chunk [1]")

    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_pooled_connection", fake_get_pooled_connection
    )
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.get_embedding_model", lambda: object())
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.search_documents", fake_search_documents
    )
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
    service = ChatRuntimeService(graph=object())
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
            return AIMessage(content="AI Vector Search supports semantic search. [1]")

    fake_llm = FakeLLM()
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_pooled_connection", fake_get_pooled_connection
    )
    monkeypatch.setattr("src.rag_agent.runtime.rag_runtime.get_embedding_model", lambda: object())
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.search_documents", fake_search_documents
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.memory.get_llm", lambda model_id=None: fake_llm
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.rag_runtime.get_llm", lambda model_id=None: fake_llm
    )

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
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type("Settings", (), {"REQUIRE_TOOL_CALL": False, "MCP_WORKFLOW_POLICY": {}})(),
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
    assert captured["require_tool_call"] is False
    assert result["mcp_tools_used"] == ["calculator_tool"]


def test_graph_service_mixed_mode_honors_require_tool_call_setting(monkeypatch) -> None:
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
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args
        captured["require_tool_call"] = kwargs.get("require_tool_call")
        return ("ok", ["calculator_tool"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type("Settings", (), {"REQUIRE_TOOL_CALL": True, "MCP_WORKFLOW_POLICY": {}})(),
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async", fake_get_mcp_tools_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async", fake_get_mcp_answer_async
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

    assert result["final_answer"] == "ok"
    assert captured["require_tool_call"] is True


def test_graph_service_mixed_mode_passes_prior_thread_history_to_agent(monkeypatch) -> None:
    service = ChatRuntimeService(graph=object())
    thread_id = "thread-mixed-memory"
    captured: dict[str, object] = {}
    service._thread_state[thread_id] = {
        "messages": [
            HumanMessage(
                content="When retrieving contracts, use the vendor from the prior invoice."
            ),
            AIMessage(content="I will use the prior invoice vendor for contract lookups."),
        ],
        "final_answer": "I will use the prior invoice vendor for contract lookups.",
    }

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
        chat_history: list[object] | None = None,
        **kwargs: object,
    ) -> tuple[str, list[str], list[object]]:
        _ = kwargs
        captured["question"] = question
        captured["chat_history"] = chat_history
        return ("retrieved prior vendor terms", ["retrieval_tool"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async", fake_get_mcp_tools_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async", fake_get_mcp_answer_async
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type("Settings", (), {"REQUIRE_TOOL_CALL": False, "MCP_WORKFLOW_POLICY": {}})(),
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Now compare it with the contract terms."}],
            model_id="google.gemini-2.5-pro",
            thread_id=thread_id,
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    history = cast(list[object], captured["chat_history"])
    assert captured["question"] == "Now compare it with the contract terms."
    assert [getattr(message, "content", "") for message in history] == [
        "When retrieving contracts, use the vendor from the prior invoice.",
        "I will use the prior invoice vendor for contract lookups.",
    ]
    assert result["final_answer"] == "retrieved prior vendor terms"


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
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type("Settings", (), {"REQUIRE_TOOL_CALL": False, "MCP_WORKFLOW_POLICY": {}})(),
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
    assert captured["require_tool_call"] is False
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
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
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
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: _Tool(),
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
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


def test_graph_service_mixed_mode_uses_only_retrieval_tool_state_for_citations(
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
        return ("namespace is xyz", ["oracle_retrieval"], [])

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: _Tool(),
    )

    def fail_if_direct_retrieval_called(self, **kwargs):
        _ = self, kwargs
        raise AssertionError("mixed mode must not run direct retrieval outside oracle_retrieval")

    monkeypatch.setattr(ChatRuntimeService, "_retrieve_oracle_docs", fail_if_direct_retrieval_called)
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "How can I create visual applications?"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-refs-tool-state",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "I don't know the answer from the selected Oracle collection."
    assert result["mcp_tools_used"] == ["oracle_retrieval"]
    assert result["citations"] == []
    assert result["reranker_docs"] == []
    assert result["context_usage"] is None


def test_graph_service_mixed_mode_does_not_run_direct_retrieval_after_tool_error(
    monkeypatch,
) -> None:
    service = ChatRuntimeService(graph=object())

    class _Tool:
        name = "oracle_retrieval"
        description = "Retrieve Oracle documentation context for a question."
        _retrieval_state = {
            "docs": [],
            "error": "Failed due to a DB error: ORA-22275: invalid LOB locator specified",
        }

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args, kwargs
        return (
            "Oracle retrieval failed while searching the knowledge base.",
            ["oracle_retrieval"],
            [],
        )

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return []

    def fail_if_direct_retrieval_called(self, **kwargs):
        _ = self, kwargs
        raise AssertionError("direct retrieval should not run after tool error")

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: _Tool(),
    )
    monkeypatch.setattr(ChatRuntimeService, "_retrieve_oracle_docs", fail_if_direct_retrieval_called)
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Give me Northway payment terms"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-refs-tool-error",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "Oracle retrieval failed while searching the knowledge base."
    assert result["citations"] == []
    assert result["context_usage"] is None


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
        raise AssertionError("direct retrieval should not run when non-retrieval MCP tools were used")

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: _Tool(),
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
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


def test_graph_service_mixed_mode_keeps_metadata_answer_without_rag_override(
    monkeypatch,
) -> None:
    service = ChatRuntimeService(graph=object())

    class _Tool:
        name = "oracle_retrieval"
        description = "Retrieve Oracle documentation context for a question."
        _retrieval_state = {"docs": []}

    @tool("oic_LIST_DOCUMENTS")
    def list_documents(folder_name: str) -> str:
        """List available documents."""
        return folder_name

    async def fake_get_mcp_answer_async(
        *args: object, **kwargs: object
    ) -> tuple[str, list[str], list[object]]:
        _ = args, kwargs
        return (
            "I have access to document listing, classification, extraction, invoice creation, "
            "approval, email summary, and Oracle retrieval tools.",
            [],
            [],
        )

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [list_documents]

    def fail_if_retrieval_called(self, **kwargs):
        _ = self, kwargs
        raise AssertionError("direct retrieval should not override a substantive MCP answer")

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: _Tool(),
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(ChatRuntimeService, "_retrieve_oracle_docs", fail_if_retrieval_called)

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "What MCP tools can you access?"}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-mcp-metadata-answer",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert "document listing" in result["final_answer"]
    assert result["mcp_tools_used"] == []
    assert result["citations"] == []
    assert result["reranker_docs"] == []


def test_graph_service_mixed_mode_requires_tool_call_when_mcp_tools_explicitly_referenced(
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
        return (
            "MCP tool call required but none was produced after retry. Please try again.",
            [],
            [],
        )

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [oic_list_documents]

    def fail_if_direct_retrieval_called(self, **kwargs):
        _ = self, kwargs
        raise AssertionError(
            "direct retrieval must not run when explicit MCP tool reference is present"
        )

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
        fake_get_mcp_answer_async,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )
    monkeypatch.setattr(ChatRuntimeService, "_retrieve_oracle_docs", fail_if_direct_retrieval_called)
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type("Settings", (), {"REQUIRE_TOOL_CALL": False, "MCP_WORKFLOW_POLICY": {}})(),
    )

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

    assert len(call_kwargs) == 1
    assert call_kwargs[0].get("require_tool_call") is True
    assert (
        result["final_answer"]
        == "MCP tool call required but none was produced after retry. Please try again."
    )
    assert result["mcp_tools_used"] == []


def test_graph_service_mixed_mode_runs_repeated_workflow_before_normal_agent(
    monkeypatch,
) -> None:
    service = ChatRuntimeService(graph=object())

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool("list_work")
    def list_work(folder: str) -> str:
        """List work units."""
        return folder

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [list_work]

    async def fail_if_normal_agent_runs(*args: object, **kwargs: object):
        _ = args, kwargs
        raise AssertionError("normal MCP agent must not run before repeated workflow controller")

    async def fake_repeated_workflow(**kwargs: object):
        assert kwargs["question"] == "Process every item and send a summary."
        assert "discovery_result" not in kwargs
        return (
            "processed all work",
            ["list_work"],
            [
                {
                    "tool_name": "list_work",
                    "args": {},
                    "result": '{"items": [{"id": "a"}, {"id": "b"}]}',
                }
            ],
        )

    async def fake_should_use_repeated_workflow(**kwargs: object) -> bool:
        _ = kwargs
        return True

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async", fake_get_mcp_tools_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async", fail_if_normal_agent_runs
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "ENABLE_PERSISTENT_MEMORY": False,
                "LANGGRAPH_SQLITE_PATH": "",
                "MCP_REPEATED_WORKFLOW_CONTROLLER": True,
                "MCP_WORKFLOW_POLICY": {},
                "REQUIRE_TOOL_CALL": False,
            },
        )(),
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.should_use_repeated_workflow",
        fake_should_use_repeated_workflow,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.run_repeated_mcp_workflow",
        fake_repeated_workflow,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Process every item and send a summary."}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-repeated-first",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert result["final_answer"] == "processed all work"
    assert result["mcp_tools_used"] == ["list_work"]


def test_graph_service_mixed_mode_stops_when_repeated_workflow_has_no_queue(
    monkeypatch,
) -> None:
    service = ChatRuntimeService(graph=object())
    normal_calls = 0

    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve Oracle documentation context for a question."""
        return f"retrieved: {question}"

    @tool("list_work")
    def list_work(folder: str) -> str:
        """List work units."""
        return folder

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [list_work]

    async def fake_get_mcp_answer_async(*args: object, **kwargs: object):
        nonlocal normal_calls
        _ = args, kwargs
        normal_calls += 1
        return ("normal answer", ["list_work"], [{"tool_name": "list_work", "result": "ok"}])

    async def fake_repeated_workflow(**kwargs: object):
        assert "discovery_result" not in kwargs
        return None

    async def fake_should_use_repeated_workflow(**kwargs: object) -> bool:
        _ = kwargs
        return True

    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async", fake_get_mcp_tools_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async", fake_get_mcp_answer_async
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "ENABLE_PERSISTENT_MEMORY": False,
                "LANGGRAPH_SQLITE_PATH": "",
                "MCP_REPEATED_WORKFLOW_CONTROLLER": True,
                "MCP_WORKFLOW_POLICY": {},
                "REQUIRE_TOOL_CALL": False,
            },
        )(),
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.should_use_repeated_workflow",
        fake_should_use_repeated_workflow,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.run_repeated_mcp_workflow",
        fake_repeated_workflow,
    )
    monkeypatch.setattr(
        ChatRuntimeService,
        "_build_oracle_retrieval_tool",
        lambda self, collection_name=None: retrieval_tool,
    )

    result = asyncio.run(
        service.run_chat(
            messages=[{"role": "user", "content": "Process the item."}],
            model_id="google.gemini-2.5-pro",
            thread_id="thread-repeated-fallback",
            session_id=None,
            collection_name="RAG_KNOWLEDGE_BASE",
            enable_reranker=None,
            enable_tracing=None,
            mode="mixed",
            mcp_server_keys=None,
            stream=False,
        )
    )

    assert normal_calls == 0
    assert "could not identify a work queue" in result["final_answer"]


def test_graph_service_mixed_mode_enforces_generic_workflow_policy_when_activated(
    monkeypatch,
) -> None:
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

    monkeypatch.setattr("src.rag_agent.runtime.chat_service.get_settings", lambda: _Settings())
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
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
        return (
            "x = 6",
            ["calculator_tool"],
            [{"tool_name": "calculator_tool", "args": {}, "result": "x=6"}],
        )

    async def fake_get_mcp_tools_async(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [calculator_tool]

    monkeypatch.setattr("src.rag_agent.runtime.chat_service.get_settings", lambda: _Settings())
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
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


def test_graph_service_mixed_mode_replaces_trivial_answer_with_tool_failure_summary(
    monkeypatch,
) -> None:
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
        "src.rag_agent.runtime.chat_service.get_mcp_tools_async",
        fake_get_mcp_tools_async,
    )
    monkeypatch.setattr(
        "src.rag_agent.runtime.chat_service.get_mcp_answer_async",
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
