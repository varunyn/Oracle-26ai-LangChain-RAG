from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MCP_TOOL_ACTIVITY_NAME = "mcp_tool_activity"


def mcp_tool_activity_event(event: Mapping[str, Any]) -> dict[str, object]:
    phase = str(event.get("phase") or "start")
    status = "error" if phase == "error" else "finished" if phase == "end" else "running"
    return {
        "name": MCP_TOOL_ACTIVITY_NAME,
        "payload": {
            "tool_run_id": str(event.get("tool_run_id") or ""),
            "tool_name": str(event.get("tool_name") or "unknown_tool"),
            "server_name": str(event.get("server_name") or "").strip() or None,
            "status": status,
            "args": event.get("args"),
            "output": event.get("result"),
            "error": event.get("error"),
        },
    }
