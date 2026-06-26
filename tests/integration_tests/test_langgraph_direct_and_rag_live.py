from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest
from langgraph_sdk import get_sync_client


def _integration_enabled() -> bool:
    return os.environ.get("RUN_INTEGRATION_TESTS") == "1"


def _langgraph_url() -> str:
    return os.environ.get("LANGGRAPH_API_URL", "http://127.0.0.1:2024")


@pytest.fixture(scope="module")
def configured_langgraph_url() -> str:
    if not _integration_enabled():
        pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests")

    url = _langgraph_url()
    try:
        urllib.request.urlopen(f"{url}/ok", timeout=5)
    except urllib.error.HTTPError:
        pass
    except Exception as exc:  # pragma: no cover - skip path depends on local runtime state
        pytest.skip(f"LangGraph Agent Server not reachable at {url}: {exc}")
    return url


@pytest.mark.integration
def test_chat_agent_direct_mode_live(configured_langgraph_url: str) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={"messages": [{"role": "user", "content": "Reply with the word READY"}]},
        context={"mode": "direct"},
    )
    messages = result["messages"]
    assert any("READY" in str(message.get("content", "")) for message in messages)


@pytest.mark.integration
def test_chat_agent_rag_mode_live(configured_langgraph_url: str) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Use retrieval to answer the configured corpus question",
                }
            ]
        },
        context={"mode": "rag", "collection_name": "default"},
    )
    assert result["references"]["mode"] == "rag"
