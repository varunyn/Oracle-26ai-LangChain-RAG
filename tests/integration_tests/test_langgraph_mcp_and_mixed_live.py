from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest
from langgraph_sdk import get_sync_client

from src.rag_agent.infrastructure.mcp_settings import get_mcp_servers_config


def _integration_enabled() -> bool:
    return os.environ.get("RUN_INTEGRATION_TESTS") == "1"


def _langgraph_url() -> str:
    return os.environ.get("LANGGRAPH_API_URL", "http://127.0.0.1:2024")


def _reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def configured_langgraph_url() -> str:
    if not _integration_enabled():
        pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests")

    url = _langgraph_url()
    if not _reachable(f"{url}/ok"):
        pytest.skip(f"LangGraph Agent Server not reachable at {url}")
    return url


@pytest.fixture(scope="module")
def configured_mcp_server_keys() -> list[str]:
    config = get_mcp_servers_config()
    keys = sorted(key for key in config.keys() if str(key).strip())
    if not keys:
        pytest.skip("No enabled MCP servers are configured for this environment")
    unreachable = [key for key in keys if not _reachable(str(config[key].get("url") or ""))]
    if unreachable:
        pytest.skip(f"Configured MCP servers are unreachable: {', '.join(unreachable)}")
    return keys


@pytest.mark.integration
def test_chat_agent_mcp_mode_live(
    configured_langgraph_url: str, configured_mcp_server_keys: list[str]
) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Use the calculator tool to compute 19 + 23. Return the numeric result.",
                }
            ]
        },
        context={"mode": "mcp", "mcp_server_keys": configured_mcp_server_keys},
    )
    assert result["references"]["mode"] == "mcp"
    assert result["references"].get("mcp_used") is True
    assert "42" in str(result["messages"][-1].get("content", ""))


@pytest.mark.integration
def test_chat_agent_mixed_mode_live(
    configured_langgraph_url: str, configured_mcp_server_keys: list[str]
) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    result = client.runs.wait(
        None,
        "chat_agent",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Using the selected Oracle collection, tell me the net payment terms "
                        "for Summit Technologies, then use the calculator tool to add 5 days "
                        "to that payment term."
                    ),
                }
            ]
        },
        context={
            "mode": "mixed",
            "collection_name": "ORACLE_WEB_EMBEDDINGS",
            "mcp_server_keys": configured_mcp_server_keys,
        },
    )
    assert result["references"]["mode"] == "mixed"
    assert result["references"].get("mcp_used") is True
    citations = result["references"].get("citations")
    assert isinstance(citations, list)
    assert len(citations) > 0
