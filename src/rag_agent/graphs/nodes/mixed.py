from __future__ import annotations

from typing import Any, cast

from langgraph.runtime import Runtime

from src.rag_agent.graphs.mcp_policies import (
    NO_ORACLE_CONTEXT_ANSWER,
    ORACLE_RETRIEVAL_FAILED_ANSWER,
    enforce_workflow_policy,
    is_trivial_answer,
    mixed_tool_supplemental_context,
    oracle_retrieval_error,
    oracle_retrieval_used_without_context,
    repeated_workflow_controller_enabled,
    require_tool_call_enabled,
    workflow_checkpoint_path,
    workflow_policy_for_request,
)
from src.rag_agent.graphs.nodes.references import (
    assistant_message_from_exception,
    assistant_message_from_result,
)
from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.infrastructure import oci_models as _oci_models
from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.mcp_turn import run_mcp_agent_turn, tool_failure_summary
from src.rag_agent.runtime.memory import (
    chat_history_before_latest_user,
    langchain_messages_to_dicts,
    latest_user_message,
)
from src.rag_agent.utils.langfuse_tracing import start_langfuse_chat_trace


def get_llm(model_id: str | None = None) -> Any:
    return _oci_models.get_llm(model_id=model_id)


async def run_mixed_node(
    state: ChatGraphState, runtime: Runtime[ChatGraphContext]
) -> ChatGraphState:
    context = get_runtime_context(runtime)
    thread_id = get_thread_id(runtime)
    messages = langchain_messages_to_dicts(state["messages"])
    try:
        with start_langfuse_chat_trace(
            enabled=cast(bool | None, context.get("enable_tracing")),
            mode="mixed",
            model_id=cast(str | None, context.get("model_id")),
            session_id=cast(str | None, context.get("session_id")),
            thread_id=thread_id,
            input_payload={"question": latest_user_message(messages)} if messages else None,
        ) as langfuse_trace:
            question = latest_user_message(messages)
            chat_history = chat_history_before_latest_user(messages)
            retrieval_tool = rag_runtime.build_oracle_retrieval_tool(
                collection_name=cast(str | None, context.get("collection_name")),
                filter_docs=rag_runtime.filter_retrieved_docs,
            )
            resolved_model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
            run_cfg = build_run_config(
                thread_id=thread_id,
                mode="mixed",
                model_id=resolved_model_id,
                session_id=cast(str | None, context.get("session_id")),
                enable_tracing=cast(bool | None, context.get("enable_tracing")),
                mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
                trace_context=langfuse_trace.trace_context if langfuse_trace else None,
            )
            mcp_turn = await run_mcp_agent_turn(
                question=question,
                chat_history=chat_history,
                resolved_model_id=resolved_model_id,
                run_config=run_cfg,
                mode="mixed",
                mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
                require_tool_call=require_tool_call_enabled(),
                repeated_workflow_enabled=repeated_workflow_controller_enabled(),
                workflow_checkpoint_path=workflow_checkpoint_path(),
                tool_progress_callback=None,
                answer_delta_callback=None,
                stop_after_tool_names=None,
                extra_tools=[retrieval_tool],
                require_mcp_tool_call_when_referenced=True,
            )
            final_answer = mcp_turn.answer
            tools_used = mcp_turn.tools_used
            tool_invocations = mcp_turn.tool_invocations
            workflow_policy = workflow_policy_for_request(mode="mixed", question=question)
            policy_applied, missing_capabilities, policy_failure_message = enforce_workflow_policy(
                policy=workflow_policy,
                tools_used=tools_used,
                tool_invocations=cast(list[dict[str, object]], tool_invocations),
            )
            policy_error = policy_failure_message if policy_applied and missing_capabilities else None
            if policy_error:
                final_answer = policy_error
            tool_failure_error = tool_failure_summary(cast(list[dict[str, object]], tool_invocations))
            if not policy_error and is_trivial_answer(final_answer) and tool_failure_error:
                final_answer = tool_failure_error
                policy_error = tool_failure_error
            retrieval_state = getattr(retrieval_tool, "_retrieval_state", None)
            retrieval_docs = (
                cast(list[object], retrieval_state.get("docs", []))
                if isinstance(retrieval_state, dict)
                else []
            )
            retrieval_error = oracle_retrieval_error(
                retrieval_state=retrieval_state,
                tools_used=tools_used,
                tool_invocations=cast(list[dict[str, object]], tool_invocations),
            )
            if retrieval_docs and question:
                retrieval_docs = rag_runtime.rerank_retrieved_docs(
                    question,
                    cast(list[Any], retrieval_docs),
                    enable_reranker=cast(bool | None, context.get("enable_reranker")),
                )
            if not policy_error and retrieval_error:
                final_answer = ORACLE_RETRIEVAL_FAILED_ANSWER
                policy_error = ORACLE_RETRIEVAL_FAILED_ANSWER
            if not policy_error and oracle_retrieval_used_without_context(
                retrieval_state=retrieval_state,
                retrieval_docs=cast(list[Any], retrieval_docs),
                tools_used=tools_used,
                tool_invocations=cast(list[dict[str, object]], tool_invocations),
            ):
                final_answer = NO_ORACLE_CONTEXT_ANSWER
            if not policy_error and retrieval_docs:
                supplemental_context = mixed_tool_supplemental_context(
                    cast(list[dict[str, object]], tool_invocations)
                )
                final_answer, _rag_usage, resolved_model_id = await rag_runtime.synthesize_rag_answer(
                    question=question,
                    docs=cast(list[Any], retrieval_docs),
                    model_id=cast(str | None, context.get("model_id")),
                    run_config=run_cfg,
                    supplemental_context=supplemental_context,
                )
            result: dict[str, object] = {
                "final_answer": final_answer,
                "error": policy_error,
                "standalone_question": question or None,
                "citations": rag_runtime.citations_from_docs(cast(list[Any], retrieval_docs)),
                "reranker_docs": rag_runtime.serialize_docs(cast(list[Any], retrieval_docs)),
                "context_usage": (
                    {"retrieved_docs_count": len(retrieval_docs)} if retrieval_docs else None
                ),
                "mcp_used": bool(tools_used),
                "mcp_tools_used": tools_used,
                "mcp_tool_invocations": tool_invocations,
            }
            if isinstance(resolved_model_id, str) and resolved_model_id.strip():
                result["model_id"] = resolved_model_id.strip()
            if langfuse_trace is not None and langfuse_trace.trace_id:
                result["trace_id"] = langfuse_trace.trace_id
        assistant_message = assistant_message_from_result("mixed", result)
    except Exception as exc:
        assistant_message = assistant_message_from_exception("mixed", exc)
    return {
        "messages": [assistant_message],
        "references": assistant_message.additional_kwargs,
    }
