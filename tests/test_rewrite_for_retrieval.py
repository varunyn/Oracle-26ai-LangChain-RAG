from typing import cast

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.langgraph.nodes.rewrite_for_retrieval import RewriteForRetrieval
from src.rag_agent.langgraph.state import MixedV2State, RetrievalIntent


def test_rewrite_for_retrieval_emits_standalone_question_and_intent_for_retrieve() -> None:
    node = RewriteForRetrieval()
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


def test_rewrite_for_retrieval_infers_hybrid_search_from_version_and_product_filters() -> None:
    node = RewriteForRetrieval()
    state: MixedV2State = {
        "user_request": "What changed in OCI CLI v3 for buckets?",
        "messages": [HumanMessage(content="What changed in OCI CLI v3 for buckets?")],
        "intent": "retrieve",
    }

    result = node(state)

    retrieval_intent = cast(RetrievalIntent, result["retrieval_intent"])
    assert retrieval_intent is not None
    assert retrieval_intent["search_mode"] == "hybrid"
    assert retrieval_intent["product_area"] == "oci cli"
    assert retrieval_intent["doc_version"] == "v3"
    assert retrieval_intent["metadata_filters"] == {"product_area": "oci cli", "doc_version": "v3"}


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
