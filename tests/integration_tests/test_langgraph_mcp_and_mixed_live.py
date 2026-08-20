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


def _thread_messages(client: object, thread_id: str) -> list[dict[str, object]]:
    state = client.threads.get_state(thread_id)  # type: ignore[attr-defined]
    values = state.get("values", {})
    messages = values.get("messages", {}) if isinstance(values, dict) else []
    return messages if isinstance(messages, list) else []


def _message_content(message: dict[str, object]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return repr(content)


def _last_assistant_message(messages: list[dict[str, object]]) -> dict[str, object]:
    for message in reversed(messages):
        if str(message.get("type") or message.get("role")) in {"ai", "assistant"}:
            return message
    raise AssertionError("stream completed without an assistant message")


def _assert_no_internal_payload_fields(parts: list[object]) -> None:
    payload = "\n".join(repr(part) for part in parts)
    for field in ("tool_agent_turn", "lease_owner_id", "recipe_json", "client_secret"):
        assert field not in payload


def _value_message_ids(parts: list[object]) -> set[str]:
    message_ids: set[str] = set()
    for part in parts:
        if getattr(part, "event", None) != "values":
            continue
        data = getattr(part, "data", None)
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            continue
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("id"), str):
                message_ids.add(message["id"])
    return message_ids


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
    thread = client.threads.create()
    result = client.runs.wait(
        str(thread["thread_id"]),
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
    thread = client.threads.create()
    result = client.runs.wait(
        str(thread["thread_id"]),
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


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "context", "question", "requires_citations"),
    [
        (
            "mcp",
            {"mode": "mcp"},
            "Use the calculator tool to compute 19 + 23. Return the numeric result.",
            False,
        ),
        (
            "mixed",
            {
                "mode": "mixed",
                "collection_name": "ORACLE_WEB_EMBEDDINGS",
            },
            (
                "Using the selected Oracle collection, tell me the net payment terms "
                "for Summit Technologies, then use the calculator tool to add 5 days "
                "to that payment term."
            ),
            True,
        ),
    ],
)
def test_chat_agent_mcp_and_mixed_stream_contract_live(
    configured_langgraph_url: str,
    configured_mcp_server_keys: list[str],
    mode: str,
    context: dict[str, object],
    question: str,
    requires_citations: bool,
) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    thread_id = str(client.threads.create()["thread_id"])
    context = {**context, "mcp_server_keys": configured_mcp_server_keys}
    parts = list(
        client.runs.stream(
            thread_id,
            "chat_agent",
            input={"messages": [{"role": "user", "content": question}]},
            context=context,
            stream_mode=["messages", "values"],
        )
    )

    _assert_no_internal_payload_fields(parts)
    messages = _thread_messages(client, thread_id)
    assert messages
    final_message = _last_assistant_message(messages)
    assert str(final_message.get("type") or final_message.get("role")) in {"ai", "assistant"}
    assert _message_content(final_message).strip()

    state = client.threads.get_state(thread_id)
    references = state["values"].get("references", {})
    assert isinstance(references, dict)
    assert references.get("mode") == mode
    assert references.get("mcp_used") is True
    if requires_citations:
        citations = references.get("citations")
        assert isinstance(citations, list)
        assert citations


@pytest.mark.integration
def test_chat_agent_mcp_interrupt_resume_keeps_terminal_message_stable_live(
    configured_langgraph_url: str, configured_mcp_server_keys: list[str]
) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    thread_id = str(client.threads.create()["thread_id"])
    context = {"mode": "mcp", "mcp_server_keys": configured_mcp_server_keys}
    question = "Use the calculator tool to compute 19 + 23. Return the numeric result."

    list(
        client.runs.stream(
            thread_id,
            "chat_agent",
            input={"messages": [{"role": "user", "content": question}]},
            context=context,
            stream_mode=["messages", "values"],
            interrupt_before=["mcp_compose"],
        )
    )
    interrupted_messages = _thread_messages(client, thread_id)
    assert interrupted_messages

    resumed_parts = list(
        client.runs.stream(
            thread_id,
            "chat_agent",
            input=None,
            context=context,
            stream_mode=["messages", "values"],
        )
    )
    _assert_no_internal_payload_fields(resumed_parts)
    resumed_messages = _thread_messages(client, thread_id)
    assert resumed_messages
    terminal_message = _last_assistant_message(resumed_messages)
    terminal_id = terminal_message.get("id")
    assert isinstance(terminal_id, str) and terminal_id
    assert str(terminal_message.get("type") or terminal_message.get("role")) in {
        "ai",
        "assistant",
    }
    assert "42" in _message_content(terminal_message)
    assert terminal_id in _value_message_ids(resumed_parts)

    state = client.threads.get_state(thread_id)
    references = state["values"].get("references", {})
    assert isinstance(references, dict)
    assert references.get("mode") == "mcp"
    assert references.get("mcp_used") is True
    assert _last_assistant_message(_thread_messages(client, thread_id)).get("id") == terminal_id
