"""Shared config builders and conversation helpers for the API."""

import logging
import uuid
from typing import Any

from api.settings import get_settings
from src.rag_agent.infrastructure.mcp_settings import get_mcp_servers_config

logger = logging.getLogger(__name__)

_warned_about_mcp_server_keys = False
conv_log = logging.getLogger(__name__ + ".conversations")


def generate_request_id() -> str:
    return str(uuid.uuid4())


def build_chat_config(
    model_id: str | None = None,
    thread_id: str | None = None,
    collection_name: str | None = None,
    enable_reranker: bool | None = None,
    enable_tracing: bool | None = None,
    mode: str | None = None,
    mcp_server_keys: list[str] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    server_keys = mcp_server_keys
    if not server_keys or not isinstance(server_keys, list) or len(server_keys) == 0:
        server_keys = None

    # Warn once per process if mcp_server_keys or MCP_SERVER_KEYS is provided
    global _warned_about_mcp_server_keys
    if (
        mcp_server_keys is not None or getattr(settings, "MCP_SERVER_KEYS", None) is not None
    ) and not _warned_about_mcp_server_keys:
        logger.warning(
            "MCP_SERVER_KEYS/mcp_server_keys does not choose the default mode. Mode is determined by ENABLE_MCP_TOOLS and the configured MCP server list, while MCP_SERVER_KEYS still limits which configured MCP servers/tools are loaded."
        )
        _warned_about_mcp_server_keys = True

    if mode is not None:
        effective_mode = mode
    else:
        # New default logic
        enable_mcp_tools = getattr(settings, "ENABLE_MCP_TOOLS", True)
        mcp_servers_config = get_mcp_servers_config()
        if enable_mcp_tools and mcp_servers_config and len(mcp_servers_config) > 0:
            effective_mode = "mixed"
        else:
            effective_mode = "rag"

    out: dict[str, Any] = {
        "configurable": {
            "model_id": model_id or settings.LLM_MODEL_ID,
            "embed_model_type": settings.EMBED_MODEL_TYPE,
            "search_mode": settings.RAG_SEARCH_MODE,
            "enable_reranker": (
                enable_reranker
                if enable_reranker is not None
                else getattr(settings, "ENABLE_RERANKER", True)
            ),
            "enable_tracing": enable_tracing if enable_tracing is not None else False,
            "collection_name": collection_name or settings.DEFAULT_COLLECTION,
            "thread_id": thread_id or generate_request_id(),
            "mode": effective_mode,
            "max_rounds": getattr(settings, "MCP_MAX_ROUNDS", 2),
        }
    }
    if getattr(settings, "ENABLE_MCP_TOOLS", True):
        if server_keys and len(server_keys) > 0:
            out["configurable"]["mcp_server_keys"] = server_keys
            logger.info("MCP: chat config mcp_server_keys=%s", server_keys)
        elif server_keys is None:
            _cfg_raw = get_mcp_servers_config()
            _cfg = _cfg_raw if isinstance(_cfg_raw, dict) else {}
            default_key = "default" if "default" in _cfg else (next(iter(_cfg), None) if _cfg else None)
            if isinstance(default_key, str) and default_key.strip():
                out["configurable"]["mcp_server_keys"] = [default_key]
                logger.debug("MCP: chat config default mcp_server_keys selected")
    return out


def _mcp_tool_names(mcp_tools_used: list[Any] | None) -> list[str]:
    """Return stable, queryable MCP tool names without raw tool payloads."""
    names: list[str] = []
    for item in mcp_tools_used or []:
        name: str | None = None
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            for key in ("name", "tool_name", "id"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    name = value
                    break
        if name is None:
            name = str(item)
        name = name.strip()
        if name:
            names.append(name)
    return names


def log_conversation_out(
    final_answer: str,
    error: str | None,
    mcp_used: bool | None,
    mcp_tools_used: list[Any] | None,
    standalone_question: str | None,
) -> None:
    """Log outcome of a conversation (RAG/MCP) for tracing."""
    answer_len = len(final_answer or "")
    standalone_len = len(standalone_question or "")
    tool_names = _mcp_tool_names(mcp_tools_used)
    attributes = {
        "event_type": "chat_out",
        "answer_len": answer_len,
        "standalone_len": standalone_len,
        "error": error,
        "mcp_used": bool(mcp_used),
        "mcp_tool_count": len(tool_names),
        "mcp_tool_names": ",".join(tool_names),
    }
    conv_log.info(
        "chat_out answer_len=%s standalone_len=%s error=%s mcp_used=%s mcp_tool_count=%s mcp_tool_names=%s",
        answer_len,
        standalone_len,
        error,
        mcp_used,
        len(tool_names),
        ",".join(tool_names),
        extra={"otel_attributes": attributes},
    )
