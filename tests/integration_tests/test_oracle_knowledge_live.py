"""Gated live Oracle/OCI verification for the Oracle Knowledge service.

Run explicitly with ``RUN_ORACLE_KNOWLEDGE_LIVE=1``. This test never substitutes
fake providers and reports configuration/service skips rather than claiming a
mocked result is provider evidence.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from api.settings import Settings, get_settings
from src.rag_agent.application.oracle_knowledge import SearchKnowledgeRequest
from src.rag_agent.core.citations import normalize_citation
from src.rag_agent.infrastructure.oracle_knowledge import (
    KnowledgeReadinessProbe,
    build_oracle_knowledge_service,
)


def _live_settings() -> Settings:
    if os.getenv("RUN_ORACLE_KNOWLEDGE_LIVE") != "1":
        pytest.skip("Live Oracle Knowledge verification is gated; set RUN_ORACLE_KNOWLEDGE_LIVE=1")
    settings = get_settings()
    config_path = Path(os.path.expanduser(str(settings.OCI_CONFIG_FILE)))
    missing = [
        name
        for name, value in {
            "VECTOR_DSN": settings.VECTOR_DSN,
            "COMPARTMENT_ID": settings.COMPARTMENT_ID,
            "OCI_CONFIG_FILE": str(config_path),
        }.items()
        if (
            not value
            or "your_" in str(value)
            or (name == "OCI_CONFIG_FILE" and not config_path.is_file())
        )
    ]
    if missing:
        pytest.skip(f"Live Oracle/OCI configuration unavailable: {', '.join(missing)}")
    return settings


@pytest.mark.integration
def test_live_oracle_knowledge_search_has_scores_ids_citations_and_reranking() -> None:
    settings = _live_settings()
    key = settings.ORACLE_KNOWLEDGE_DEFAULT_KEY
    collection = os.getenv("ORACLE_KNOWLEDGE_LIVE_COLLECTION") or settings.DEFAULT_COLLECTION
    if not collection:
        pytest.skip("No ORACLE_KNOWLEDGE_LIVE_COLLECTION or application default collection")

    ready, reason = KnowledgeReadinessProbe(settings).check()
    if not ready:
        pytest.skip(f"Live dependency preflight unavailable: {reason}")
    service = build_oracle_knowledge_service(
        settings,
        knowledge_bases={key: collection},
        default_key=key,
        enable_reranker=True,
    )

    query = os.getenv("ORACLE_KNOWLEDGE_LIVE_QUERY", "Oracle database vector search")
    request = SearchKnowledgeRequest(
        query=query,
        knowledge_base=key,
        limit=min(settings.ORACLE_KNOWLEDGE_MAX_RESULT_LIMIT, 5),
        candidate_limit=min(settings.ORACLE_KNOWLEDGE_MAX_CANDIDATE_LIMIT, 10),
    )
    result = asyncio.run(service.search(request))
    if result.outcome == "no_hits":
        pytest.skip("Configured live collection returned no hits for the verification query")
    assert result.outcome == "success", result.error
    assert result.knowledge_base == key
    assert result.reranking_status == "applied"
    assert result.evidence

    repeated = asyncio.run(service.search(request))
    assert repeated.outcome == "success", repeated.error
    assert [(item.document_id, item.chunk_id) for item in repeated.evidence] == [
        (item.document_id, item.chunk_id) for item in result.evidence
    ]

    ids = {(item.document_id, item.chunk_id) for item in result.evidence}
    assert len(ids) == len(result.evidence)
    assert [item.rank for item in result.evidence] == list(range(1, len(result.evidence) + 1))
    assert all(item.content and item.source for item in result.evidence)
    assert all(item.retrieval_score is not None for item in result.evidence)
    assert all(item.reranking_score is not None for item in result.evidence)
    rerank_scores = [item.reranking_score for item in result.evidence]
    assert rerank_scores == sorted(rerank_scores, reverse=True)
    for item in result.evidence:
        citation = normalize_citation(
            {
                "source": item.source,
                "title": item.title,
                "page": item.page,
                "link": item.link,
            }
        )
        assert citation == {
            "source": item.source,
            "page": item.page,
            "link": item.link,
        }
