from __future__ import annotations

from src.rag_agent.runtime.mcp_activity import (
    MCP_TOOL_ACTIVITY_NAME,
    mcp_tool_activity_event,
)


def test_mcp_tool_activity_event_normalizes_started_tool() -> None:
    assert mcp_tool_activity_event(
        {
            "phase": "start",
            "tool_name": "lookup",
            "tool_run_id": "call-1",
            "args": {"query": "invoice"},
        }
    ) == {
        "name": MCP_TOOL_ACTIVITY_NAME,
        "payload": {
            "tool_run_id": "call-1",
            "tool_name": "lookup",
            "server_name": None,
            "status": "running",
            "args": {"query": "invoice"},
            "output": None,
            "error": None,
        },
    }


def test_mcp_tool_activity_event_preserves_server_name() -> None:
    event = mcp_tool_activity_event(
        {
            "phase": "start",
            "tool_run_id": "call-1",
            "tool_name": "Calculator_linear_regression",
            "server_name": "calculator",
            "args": {"data": [[1, 2], [2, 3.5]]},
        }
    )

    assert event["payload"]["server_name"] == "calculator"


def test_mcp_tool_activity_event_normalizes_completed_and_failed_tools() -> None:
    assert mcp_tool_activity_event(
        {"phase": "end", "tool_name": "lookup", "tool_run_id": "call-1", "result": "ok"}
    )["payload"]["status"] == "finished"
    assert mcp_tool_activity_event(
        {"phase": "error", "tool_name": "lookup", "tool_run_id": "call-1", "error": "failed"}
    )["payload"] == {
        "tool_run_id": "call-1",
        "tool_name": "lookup",
        "server_name": None,
        "status": "error",
        "args": None,
        "output": None,
        "error": "failed",
    }
