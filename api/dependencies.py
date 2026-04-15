"""Shared config builders and conversation helpers for the API."""

import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.deps.request import (  # noqa: F401
    get_graph_service,
)
from api.settings import get_settings
from src.rag_agent.infrastructure.mcp_settings import get_mcp_servers_config

from .schemas import ChatMessage

logger = logging.getLogger(__name__)

_warned_about_mcp_server_keys = False
_CONV_LOG_PREVIEW_LEN = 400
conv_log = logging.getLogger(__name__ + ".conversations")
_RAW_TOOL_CALL_PATTERN = re.compile(r"^[\w\.-]+\s*\([^)]*\)$")


def _looks_like_internal_mcp_trace(role: str, content: str) -> bool:
    if role != "assistant":
        return False
    candidate = content.strip()
    if not candidate:
        return False
    if candidate.startswith("<|python_start|>") and candidate.endswith("<|python_end|>"):
        return True
    if _RAW_TOOL_CALL_PATTERN.fullmatch(candidate):
        return True
    if candidate.startswith("[") and candidate.endswith("]") and "(" in candidate:
        return True
    return False


def generate_request_id() -> str:
    return str(uuid.uuid4())


def messages_to_runtime_state(
    messages: list[ChatMessage],
) -> tuple[str, list[AIMessage | HumanMessage | SystemMessage]]:
    if not messages:
        return "", []
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break
    user_request = (
        messages[last_user_idx].content or ""
        if last_user_idx >= 0
        else (messages[-1].content or "" if messages[-1].role == "user" else "")
    )
    chat_history: list[AIMessage | HumanMessage | SystemMessage] = []
    for m in messages[:last_user_idx] if last_user_idx >= 0 else []:
        content = m.content or ""
        if _looks_like_internal_mcp_trace(m.role, content):
            continue
        if m.role == "system":
            chat_history.append(SystemMessage(content=content))
        elif m.role == "user":
            chat_history.append(HumanMessage(content=content))
        elif m.role == "assistant":
            chat_history.append(AIMessage(content=content))
    return user_request, chat_history


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
            "MCP_SERVER_KEYS/mcp_server_keys does not choose the default mode. Mode is determined by ENABLE_MCP_TOOLS and MCP_SERVERS_CONFIG, while MCP_SERVER_KEYS still limits which configured MCP servers/tools are loaded."
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
            _entry = _cfg.get("default") or (next(iter(_cfg.values()), None) if _cfg else None)
            mcp_url = (_entry or {}).get("url", "").strip() or None
            if mcp_url:
                out["configurable"]["mcp_url"] = mcp_url
                logger.debug("MCP: chat config mcp_url set (do not log URL)")
    return out


def _conversation_in_summary(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Build a log-safe summary of messages: role + content preview."""
    out: list[dict[str, Any]] = []
    for m in messages or []:
        role = (m.role or "unknown").strip() or "unknown"
        content = (m.content or "").strip()
        preview = content[:_CONV_LOG_PREVIEW_LEN] + (
            "..." if len(content) > _CONV_LOG_PREVIEW_LEN else ""
        )
        out.append({"role": role, "content_preview": preview, "content_length": len(content)})
    return out


def log_conversation_in(
    stream: bool,
    messages: list[ChatMessage],
    user_request: str,
    chat_history_len: int,
) -> None:
    """Log incoming chat request for conversation tracing."""
    summary = _conversation_in_summary(messages)
    conv_log.info(
        "chat_in stream=%s messages_count=%s history_len=%s user_request_preview=%s messages_summary=%s",
        stream,
        len(messages or []),
        chat_history_len,
        (user_request or "")[:_CONV_LOG_PREVIEW_LEN],
        summary,
    )


def log_conversation_out(
    final_answer: str,
    error: str | None,
    mcp_used: bool | None,
    mcp_tools_used: list[dict[str, Any]] | None,
    standalone_question: str | None,
) -> None:
    """Log outcome of a conversation (RAG/MCP) for tracing."""
    preview = (final_answer or "")[:_CONV_LOG_PREVIEW_LEN]
    preview_one_line = preview.replace("\n", " ").replace("\r", " ").strip()
    answer_len = len(final_answer or "")
    attributes = {
        "event_type": "chat_out",
        "answer_len": answer_len,
        "error": error,
        "mcp_used": bool(mcp_used),
        "mcp_tools_used": mcp_tools_used or [],
    }
    if standalone_question:
        attributes["standalone_preview"] = (standalone_question or "")[:_CONV_LOG_PREVIEW_LEN]
    attributes["final_answer_preview"] = preview_one_line
    conv_log.info(
        "chat_out answer_len=%s error=%s mcp_used=%s mcp_tools_used=%s standalone_preview=%s final_answer_preview=%s",
        answer_len,
        error,
        mcp_used,
        mcp_tools_used or [],
        attributes.get("standalone_preview"),
        preview_one_line,
        extra={"otel_attributes": attributes},
    )
