from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.rag_agent.langgraph.nodes.rewrite_for_retrieval import RewriteForRetrieval
from src.rag_agent.langgraph.state import MixedV2State, RetrievalIntent


def test_rewrite_for_retrieval_emits_standalone_question_and_intent_for_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = RewriteForRetrieval()

    class FakeStructuredModel:
        def invoke(self, messages: list[object]) -> object:
            assert len(messages) == 2
            return {
                "standalone_question": "How do I create an OCI bucket?",
                "search_mode": "semantic",
                "product_area": None,
                "doc_version": None,
                "language": None,
                "top_k": None,
                "decision_reason": "resolved pronoun from history",
            }

    class FakeLLM:
        def with_structured_output(self, _schema: type[BaseModel]) -> FakeStructuredModel:
            return FakeStructuredModel()

    monkeypatch.setattr("src.rag_agent.langgraph.nodes.rewrite_for_retrieval.get_llm", lambda: FakeLLM())
    state: MixedV2State = {
        "user_request": "How do I create one?",
        "messages": [
            HumanMessage(content="Tell me about OCI buckets."),
            AIMessage(content="OCI buckets are object storage containers."),
            HumanMessage(content="How do I create one?"),
        ],
        "intent": "retrieve",
    }

    result = node(state)

    assert result["standalone_question"] == "How do I create an OCI bucket?"
    retrieval_intent = cast(RetrievalIntent, result["retrieval_intent"])
    assert retrieval_intent is not None
    assert retrieval_intent["standalone_question"] == "How do I create an OCI bucket?"
    assert retrieval_intent["search_mode"] == "semantic"


def test_rewrite_for_retrieval_bypasses_non_retrieve_intents() -> None:
    node = RewriteForRetrieval()
    state: MixedV2State = {
        "user_request": "Solve x^2 - 5x + 6 = 0",
        "messages": [HumanMessage(content="Solve x^2 - 5x + 6 = 0")],
        "intent": "tool",
    }

    result = node(state)

    assert result == {}


def test_rewrite_for_retrieval_extracts_top_k_from_generic_request() -> None:
    node = RewriteForRetrieval()
    state: MixedV2State = {
        "user_request": "Show me top 5 examples of bucket lifecycle rules",
        "messages": [HumanMessage(content="Show me top 5 examples of bucket lifecycle rules")],
        "intent": "retrieve",
    }

    result = node(state)

    retrieval_intent = cast(RetrievalIntent, result["retrieval_intent"])
    assert retrieval_intent["top_k"] == 5


def test_rewrite_for_retrieval_empty_input_returns_empty_intent() -> None:
    node = RewriteForRetrieval()
    state: MixedV2State = {
        "user_request": "",
        "messages": [HumanMessage(content="")],
        "intent": "retrieve",
    }

    result = node(state)

    retrieval_intent = cast(RetrievalIntent, result["retrieval_intent"])
    assert result["standalone_question"] == ""
    assert retrieval_intent["standalone_question"] == ""
    assert retrieval_intent["search_mode"] == "semantic"
    assert retrieval_intent.get("metadata_filters") is None


def test_rewrite_for_retrieval_exact_term_query_prefers_keyword_or_hybrid_without_metadata() -> None:
    node = RewriteForRetrieval()
    state: MixedV2State = {
        "user_request": "What does --namespace-name do in os bucket create?",
        "messages": [HumanMessage(content="What does --namespace-name do in os bucket create?")],
        "intent": "retrieve",
    }

    result = node(state)

    retrieval_intent = cast(RetrievalIntent, result["retrieval_intent"])
    assert retrieval_intent["search_mode"] in {"keyword", "hybrid"}
    assert retrieval_intent.get("product_area") is None
    assert retrieval_intent.get("doc_version") is None
    assert retrieval_intent.get("metadata_filters") is None


def test_rewrite_for_retrieval_broad_question_prefers_semantic() -> None:
    node = RewriteForRetrieval()
    state: MixedV2State = {
        "user_request": "How does object storage replication work?",
        "messages": [HumanMessage(content="How does object storage replication work?")],
        "intent": "retrieve",
    }

    result = node(state)

    retrieval_intent = cast(RetrievalIntent, result["retrieval_intent"])
    assert retrieval_intent["search_mode"] == "semantic"
    assert retrieval_intent.get("metadata_filters") is None


def test_rewrite_for_retrieval_contextual_followup_uses_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = RewriteForRetrieval()

    class FakeStructuredModel:
        def invoke(self, messages: list[object]) -> object:
            assert len(messages) == 2
            return {
                "standalone_question": "How do I create an OCI bucket?",
                "search_mode": "semantic",
                "product_area": None,
                "doc_version": None,
                "language": None,
                "top_k": None,
                "decision_reason": "resolved pronoun from history",
            }

    class FakeLLM:
        def with_structured_output(self, _schema: type[BaseModel]) -> FakeStructuredModel:
            return FakeStructuredModel()

    monkeypatch.setattr("src.rag_agent.langgraph.nodes.rewrite_for_retrieval.get_llm", lambda: FakeLLM())

    state: MixedV2State = {
        "user_request": "How do I create one?",
        "messages": [
            HumanMessage(content="Tell me about OCI buckets."),
            AIMessage(content="OCI buckets are object storage containers."),
            HumanMessage(content="How do I create one?"),
        ],
        "intent": "retrieve",
    }

    result = node(state)
    assert result["standalone_question"] == "How do I create an OCI bucket?"


def test_rewrite_for_retrieval_llm_failure_preserves_deterministic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = RewriteForRetrieval()

    class FailingLLM:
        def with_structured_output(self, _schema: type[BaseModel]) -> object:
            raise RuntimeError("structured output unavailable")

    monkeypatch.setattr("src.rag_agent.langgraph.nodes.rewrite_for_retrieval.get_llm", lambda: FailingLLM())

    state: MixedV2State = {
        "user_request": "How do I create one?",
        "messages": [
            HumanMessage(content="Tell me about buckets."),
            HumanMessage(content="How do I create one?"),
        ],
        "intent": "retrieve",
    }

    result = node(state)
    retrieval_intent = cast(RetrievalIntent, result["retrieval_intent"])
    assert retrieval_intent["standalone_question"] == "How do I create one?"
    assert retrieval_intent["search_mode"] == "semantic"
    assert retrieval_intent.get("metadata_filters") is None


def test_rewrite_for_retrieval_json_mode_fallback_path(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RewriteForRetrieval()

    class FakeResponse:
        content = '{"standalone_question": "How do I create an OCI bucket?", "search_mode": "semantic", "product_area": null, "doc_version": null, "language": null, "top_k": null, "decision_reason": "resolved from history"}'

    class FakeLLM:
        def invoke(self, messages: list[object]) -> FakeResponse:
            assert len(messages) == 2
            return FakeResponse()

    monkeypatch.setattr("src.rag_agent.langgraph.nodes.rewrite_for_retrieval.get_llm", lambda: FakeLLM())

    state: MixedV2State = {
        "user_request": "How do I create one?",
        "messages": [
            HumanMessage(content="Tell me about OCI buckets."),
            HumanMessage(content="How do I create one?"),
        ],
        "intent": "retrieve",
    }

    result = node(state)
    assert result["standalone_question"] == "How do I create an OCI bucket?"
