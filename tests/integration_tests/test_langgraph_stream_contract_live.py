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


def _message_role(message: dict[str, object]) -> str:
    return str(message.get("role") or message.get("type") or "")


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode,context,questions",
    [
        (
            "direct",
            {"mode": "direct"},
            ["Reply with ALPHA only", "Reply with BETA only"],
        ),
        (
            "rag",
            {"mode": "rag", "collection_name": "ORACLE_WEB_EMBEDDINGS"},
            [
                "Tell me about Summit Technologies policies",
                "What are the payment terms for Summit Technologies?",
            ],
        ),
    ],
)
def test_chat_agent_stream_exposes_one_visible_answer_per_turn(
    configured_langgraph_url: str,
    mode: str,
    context: dict[str, object],
    questions: list[str],
) -> None:
    client = get_sync_client(url=configured_langgraph_url)
    thread = client.threads.create()
    thread_id = str(thread["thread_id"])

    for turn_index, question in enumerate(questions, start=1):
        message_complete_events = 0
        value_message_counts: list[int] = []

        for part in client.runs.stream(
            thread_id,
            "chat_agent",
            input={"messages": [{"role": "user", "content": question}]},
            context=context,
            stream_mode=["messages", "values"],
        ):
            if part.event == "messages/complete":
                message_complete_events += 1
            if part.event == "values" and isinstance(part.data, dict):
                messages = part.data.get("messages")
                if isinstance(messages, list):
                    value_message_counts.append(len(messages))

        state = client.threads.get_state(thread_id)
        messages = state["values"]["messages"]
        expected_count = turn_index * 2

        assert message_complete_events <= 1
        assert len(messages) == expected_count
        assert [_message_role(message) for message in messages] == ["user", "ai"] * turn_index
        assert value_message_counts[-1] == expected_count
