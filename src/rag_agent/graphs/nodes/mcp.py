from __future__ import annotations

from typing import Any, cast

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from src.rag_agent.graphs.mcp_policies import (
    is_trivial_answer,
    repeated_workflow_controller_enabled,
    require_tool_call_enabled,
    workflow_checkpoint_path,
)
from src.rag_agent.graphs.nodes.references import (
    assistant_message_from_exception,
    assistant_message_from_result,
)
from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.infrastructure import oci_models as _oci_models
from src.rag_agent.runtime.mcp_turn import run_mcp_agent_turn, tool_failure_summary
from src.rag_agent.runtime.mcp_activity import mcp_tool_activity_event
from src.rag_agent.runtime.memory import (
    chat_history_before_latest_user,
    langchain_messages_to_dicts,
    latest_user_message,
)
from src.rag_agent.utils.langfuse_tracing import start_langfuse_chat_trace


def get_llm(model_id: str | None = None) -> Any:
    return _oci_models.get_llm(model_id=model_id)


async def run_mcp_node(state: ChatGraphState, runtime: Runtime[ChatGraphContext]) -> ChatGraphState:
    context = get_runtime_context(runtime)
    thread_id = get_thread_id(runtime)
    messages = langchain_messages_to_dicts(state["messages"])
    try:
        with start_langfuse_chat_trace(
            enabled=cast(bool | None, context.get("enable_tracing")),
            mode="mcp",
            model_id=cast(str | None, context.get("model_id")),
            session_id=cast(str | None, context.get("session_id")),
            thread_id=thread_id,
            input_payload={"question": latest_user_message(messages)} if messages else None,
        ) as langfuse_trace:
            question = latest_user_message(messages)
            chat_history = chat_history_before_latest_user(messages)
            resolved_model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
            run_cfg = build_run_config(
                thread_id=thread_id,
                mode="mcp",
                model_id=resolved_model_id,
                session_id=cast(str | None, context.get("session_id")),
                enable_tracing=cast(bool | None, context.get("enable_tracing")),
                mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
                trace_context=langfuse_trace.trace_context if langfuse_trace else None,
            )

            def emit_tool_activity(event: dict[str, object]) -> None:
                get_stream_writer()(mcp_tool_activity_event(event))

            mcp_turn = await run_mcp_agent_turn(
                question=question,
                chat_history=chat_history,
                resolved_model_id=resolved_model_id,
                run_config=run_cfg,
                mode="mcp",
                mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
                require_tool_call=require_tool_call_enabled(),
                repeated_workflow_enabled=repeated_workflow_controller_enabled(),
                workflow_checkpoint_path=workflow_checkpoint_path(),
                tool_progress_callback=emit_tool_activity,
                answer_delta_callback=None,
            )
            result: dict[str, object] = {
                "final_answer": mcp_turn.answer,
                "error": None,
                "standalone_question": question or None,
                "citations": [],
                "reranker_docs": [],
                "context_usage": None,
                "mcp_used": bool(mcp_turn.tools_used),
                "mcp_tools_used": mcp_turn.tools_used,
                "mcp_tool_invocations": mcp_turn.tool_invocations,
                "model_id": mcp_turn.resolved_model_id,
            }
            tool_failure_error = tool_failure_summary(mcp_turn.tool_invocations)
            if is_trivial_answer(str(result.get("final_answer") or "")) and tool_failure_error:
                result["final_answer"] = tool_failure_error
                result["error"] = tool_failure_error
            if langfuse_trace is not None and langfuse_trace.trace_id:
                result["trace_id"] = langfuse_trace.trace_id
        assistant_message = assistant_message_from_result("mcp", result)
    except Exception as exc:
        assistant_message = assistant_message_from_exception("mcp", exc)
    return {
        "messages": [assistant_message],
        "references": assistant_message.additional_kwargs,
    }
