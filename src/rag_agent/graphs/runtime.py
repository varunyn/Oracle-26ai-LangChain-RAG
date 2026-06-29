from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from src.rag_agent.graphs.state import ChatGraphContext


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


def build_run_config(
    *,
    thread_id: str | None,
    mode: str,
    model_id: str | None,
    session_id: str | None,
    enable_tracing: bool | None,
    mcp_server_keys: list[str] | None,
    trace_context: dict[str, object] | None = None,
) -> RunnableConfig:
    configurable: dict[str, Any] = {
        "mode": mode,
        "enable_tracing": bool(enable_tracing),
    }
    if thread_id:
        configurable["thread_id"] = thread_id
    if model_id:
        configurable["model_id"] = model_id
    if session_id:
        configurable["session_id"] = session_id
    if mcp_server_keys:
        configurable["mcp_server_keys"] = mcp_server_keys
    if trace_context:
        configurable["langfuse_trace_context"] = trace_context
    return cast(RunnableConfig, {"configurable": configurable})


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


def result_to_assistant_message(mode: str, result: dict[str, object]) -> AIMessage:
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
        additional_kwargs=references,
        response_metadata=references,
    )
