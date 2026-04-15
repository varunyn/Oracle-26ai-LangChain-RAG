"""Live integration checks for chat runtime mode behavior and MCP usage.

These tests intentionally hit real configured runtime boundaries when enabled.
Run with: RUN_INTEGRATION_TESTS=1
"""

from __future__ import annotations

import asyncio
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse, urlunparse

import pytest

from api.schemas import ChatMessage
from api.services.graph_service import ChatRuntimeService
from src.rag_agent.infrastructure.mcp_settings import get_mcp_servers_config

VERIFICATION_COLLECTION = "RAG_KNOWLEDGE_BASE_TEST"

BASE_CASES: list[tuple[str, bool]] = [
    ("Solve the following equation: x^2 - 5x + 6 = 0", True),
    ("How do I integrate my visual application with a Git repository?", False),
    ("Calculate the integral of x^2 * e^x.", True),
]

MODE_CASES: list[tuple[str, str, bool]] = [
    ("rag", "how to config oci cli in linux", False),
    ("direct", "Explain what the OCI CLI is in one sentence.", False),
    ("mcp", "Solve the following equation: x^2 - 5x + 6 = 0", True),
    ("mixed", "Calculate the integral of x^2 * e^x.", True),
]


@pytest.fixture(scope="module")
def integration_enabled() -> None:
    if os.getenv("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests")


@pytest.fixture(scope="module")
def graph_service() -> ChatRuntimeService:
    return ChatRuntimeService()


def _check_mcp_available() -> bool:
    config = get_mcp_servers_config()
    if not config:
        return False
    for entry in config.values():
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        urls_to_try = [url]
        parsed = urlparse(url)
        if parsed.hostname == "host.docker.internal":
            urls_to_try.append(urlunparse(parsed._replace(netloc=f"localhost:{parsed.port or 80}")))
        for candidate in urls_to_try:
            try:
                urllib.request.urlopen(candidate, timeout=5)
                return True
            except urllib.error.HTTPError:
                return True
            except Exception:
                continue
    return False


async def _run_case(
    graph_service: ChatRuntimeService,
    question: str,
    *,
    mode: str | None = None,
) -> tuple[str, str | None, bool, list[str]]:
    result = await graph_service.run_chat(
        messages=[ChatMessage(role="user", content=question).model_dump()],
        model_id=None,
        thread_id=None,
        session_id=None,
        collection_name=VERIFICATION_COLLECTION,
        enable_reranker=None,
        enable_tracing=None,
        mode=mode,
        mcp_server_keys=None,
        stream=False,
    )
    answer = str(result.get("final_answer") or "").strip()
    err = result.get("error")
    mcp_used = bool(result.get("mcp_used"))
    mcp_tools_used = [str(t) for t in (result.get("mcp_tools_used") or [])]
    return answer, (str(err) if err is not None else None), mcp_used, mcp_tools_used


@pytest.mark.integration
@pytest.mark.parametrize("question,expect_mcp", BASE_CASES)
def test_verification_queries_cover_rag_and_mcp_paths(
    integration_enabled: None,
    graph_service: ChatRuntimeService,
    question: str,
    expect_mcp: bool,
) -> None:
    if expect_mcp and not _check_mcp_available():
        pytest.skip("MCP servers not reachable for MCP-required verification case")

    answer, err, mcp_used, mcp_tools = asyncio.run(_run_case(graph_service, question))

    assert err is None, err
    assert answer, "Expected non-empty answer"
    if expect_mcp:
        assert mcp_used is True
        assert mcp_tools
        assert "i don't know the answer" not in answer.lower()


@pytest.mark.integration
@pytest.mark.parametrize("mode,question,expect_mcp", MODE_CASES)
def test_mode_specific_verification_queries(
    integration_enabled: None,
    graph_service: ChatRuntimeService,
    mode: str,
    question: str,
    expect_mcp: bool,
) -> None:
    if expect_mcp and not _check_mcp_available():
        pytest.skip("MCP servers not reachable for MCP mode verification case")

    answer, err, mcp_used, mcp_tools = asyncio.run(_run_case(graph_service, question, mode=mode))

    assert err is None, err
    assert answer, "Expected non-empty answer"
    if expect_mcp:
        assert mcp_used is True
        assert mcp_tools
    else:
        assert mcp_used is False
