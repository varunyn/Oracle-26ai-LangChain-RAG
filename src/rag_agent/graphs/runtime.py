from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from api.settings import get_settings
from src.rag_agent.graphs.state import ChatGraphContext
from src.rag_agent.utils.logging_config import get_request_id


def get_runtime_context(runtime: Runtime[ChatGraphContext]) -> ChatGraphContext:
    context = runtime.context
    if isinstance(context, dict):
        return cast(ChatGraphContext, context)
    return {}


def get_thread_id(runtime: Runtime[ChatGraphContext]) -> str | None:
    thread_id = getattr(runtime.execution_info, "thread_id", None)
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id
    return None


def _clean_optional_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null", "-"}:
        return None
    return normalized


def build_run_config(
    *,
    parent_config: RunnableConfig | None = None,
    thread_id: str | None,
    mode: str,
    model_id: str | None,
    session_id: str | None,
    enable_tracing: bool | None,
    mcp_server_keys: list[str] | None,
    request_id: str | None = None,
    user_id: str | None = None,
    release: str | None = None,
) -> RunnableConfig:
    resolved_request_id = request_id or get_request_id()
    if resolved_request_id == "-":
        resolved_request_id = None
    resolved_release = _clean_optional_value(
        release or getattr(get_settings(), "LANGFUSE_RELEASE", None)
    )
    run_config: dict[str, Any] = dict(parent_config or {})
    parent_configurable = run_config.get("configurable")
    configurable: dict[str, Any] = (
        dict(parent_configurable) if isinstance(parent_configurable, dict) else {}
    )
    configurable.update({
        "mode": mode,
        "enable_tracing": bool(enable_tracing),
    })
    if thread_id:
        configurable["thread_id"] = thread_id
    if model_id:
        configurable["model_id"] = model_id
    if session_id:
        configurable["session_id"] = session_id
    if resolved_request_id:
        configurable["request_id"] = resolved_request_id
    if user_id:
        configurable["user_id"] = user_id
    if resolved_release:
        configurable["release"] = resolved_release
    if mcp_server_keys:
        configurable["mcp_server_keys"] = mcp_server_keys
    run_config["configurable"] = configurable
    metadata = dict(run_config.get("metadata") or {})
    for key, value in {
        "request_id": resolved_request_id,
        "thread_id": thread_id,
        "session_id": session_id,
        "user_id": user_id,
        "mode": mode,
        "model_id": model_id,
        "release": resolved_release,
    }.items():
        if isinstance(value, str) and value.strip():
            metadata[key] = value.strip()
    if metadata:
        run_config["metadata"] = metadata
    return cast(RunnableConfig, run_config)


def references_from_result(result: dict[str, object], *, mode: str) -> dict[str, object]:
    references: dict[str, object] = {"mode": mode}
    for key in (
        "standalone_question",
        "citations",
        "reranker_docs",
        "context_usage",
        "trace_id",
        "mcp_used",
        "mcp_tools_used",
        "mcp_tool_invocations",
        "outcome",
        "error",
    ):
        value = result.get(key)
        if value is None and key not in {"citations", "reranker_docs", "mcp_tools_used"}:
            continue
        if key in {"citations", "reranker_docs", "mcp_tools_used"} and not isinstance(value, list):
            references[key] = []
            continue
        references[key] = value
    return references


def result_to_assistant_message(
    mode: str, result: dict[str, object], *, message_id: str | None = None
) -> AIMessage:
    final_answer = result.get("final_answer")
    if isinstance(final_answer, str):
        content = final_answer
    elif isinstance(final_answer, list):
        content = final_answer
    elif final_answer is None:
        content = ""
    else:
        content = str(final_answer)
    references = references_from_result(result, mode=mode)
    return AIMessage(
        content=content,
        id=message_id,
        additional_kwargs=references,
        response_metadata=references,
    )
