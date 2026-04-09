"""
Structured logging for LangGraph node execution (flow visibility in Loki/Grafana).

Use log_node_start / log_node_end at node entry/exit so logs show which nodes ran
and in what order. Message format: langgraph_node node=X event=start|end key=val ...
Query in Loki: {service_name="rag-api"} |= "langgraph_node" | line_format "{{.message}}"
"""

from __future__ import annotations

import logging
from typing import Any

# Dedicated logger so Grafana/Loki can filter all node flow logs by name
_NODE_LOGGER = logging.getLogger("rag_agent.langgraph_nodes")


def _fmt_kwargs(**kwargs: Any) -> str:
    """Format key=value pairs; skip None, sanitize for one-line log."""
    parts = []
    for k, v in sorted(kwargs.items()):
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            # Keep short: list length or repr truncate
            if isinstance(v, list):
                parts.append(f"{k}={len(v)}")
            else:
                parts.append(f"{k}=<dict>")
        else:
            s = str(v).strip()
            if " " in s or "\n" in s:
                s = s.replace("\n", " ")[:80]
            parts.append(f"{k}={s}")
    return " ".join(parts) if parts else ""


def log_node_start(node_name: str, **kwargs: Any) -> None:
    """Log that a LangGraph node has started. Use at node entry."""
    attributes = {"event_type": "langgraph_node", "node": node_name, "event": "start"}
    attributes.update({k: v for k, v in kwargs.items() if v is not None})
    msg_extra = _fmt_kwargs(**kwargs)
    _NODE_LOGGER.info(
        "langgraph_node node=%s event=start %s",
        node_name,
        msg_extra,
        extra={"otel_attributes": attributes},
    )


def log_node_end(node_name: str, duration_ms: float | None = None, **kwargs: Any) -> None:
    """Log that a LangGraph node has finished. Use at node exit (before return)."""
    attributes: dict[str, Any] = {
        "event_type": "langgraph_node",
        "node": node_name,
        "event": "end",
    }
    if duration_ms is not None:
        attributes["duration_ms"] = duration_ms
    attributes.update({k: v for k, v in kwargs.items() if v is not None})
    msg_extra = _fmt_kwargs(**kwargs)
    suffix = f" {msg_extra}" if msg_extra else ""
    if duration_ms is not None:
        suffix = f" duration_ms={duration_ms:.0f}{suffix}"
    _NODE_LOGGER.info(
        "langgraph_node node=%s event=end%s",
        node_name,
        suffix,
        extra={"otel_attributes": attributes},
    )
