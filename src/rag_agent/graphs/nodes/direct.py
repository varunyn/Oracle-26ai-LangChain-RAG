from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from src.rag_agent.graphs.nodes.references import (
    assistant_message_from_exception,
    assistant_message_from_result,
)
from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.infrastructure import oci_models as _oci_models
from src.rag_agent.runtime.llm_invocation import invoke_llm_with_optional_config
from src.rag_agent.runtime.memory import langchain_messages_to_dicts, latest_user_message
from src.rag_agent.runtime.observability import emit_usage_observability, extract_usage
from src.rag_agent.utils.langfuse_tracing import start_langfuse_chat_trace


def get_llm(model_id: str | None = None) -> Any:
    return _oci_models.get_llm(model_id=model_id)


async def run_direct_node(
    state: ChatGraphState, runtime: Runtime[ChatGraphContext]
) -> ChatGraphState:
    context = get_runtime_context(runtime)
    thread_id = get_thread_id(runtime)
    messages = langchain_messages_to_dicts(state["messages"])
    try:
        with start_langfuse_chat_trace(
            enabled=cast(bool | None, context.get("enable_tracing")),
            mode="direct",
            model_id=cast(str | None, context.get("model_id")),
            session_id=cast(str | None, context.get("session_id")),
            thread_id=thread_id,
            input_payload={"question": latest_user_message(messages)} if messages else None,
        ) as langfuse_trace:
            history: list[Any] = []
            latest_question = ""
            for item in messages:
                role = str(item.get("role") or "").strip().lower()
                content = str(item.get("content") or "")
                if role == "user":
                    history.append(HumanMessage(content=content))
                    latest_question = content.strip() or latest_question
                elif role == "assistant":
                    history.append(AIMessage(content=content))

            run_cfg = build_run_config(
                thread_id=thread_id,
                mode="direct",
                model_id=cast(str | None, context.get("model_id")),
                session_id=cast(str | None, context.get("session_id")),
                enable_tracing=cast(bool | None, context.get("enable_tracing")),
                mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
                trace_context=langfuse_trace.trace_context if langfuse_trace else None,
            )
            llm = get_llm(model_id=cast(str | None, context.get("model_id")))
            response = await asyncio.to_thread(invoke_llm_with_optional_config, llm, history, run_cfg)
            usage = extract_usage(response)
            resolved_model_id = cast(str | None, getattr(llm, "model_id", None)) or cast(
                str | None, context.get("model_id")
            )
            emitted_usage, cost_usd = emit_usage_observability(
                mode="direct",
                model_id=resolved_model_id,
                session_id=cast(str | None, context.get("session_id")),
                thread_id=thread_id,
                usage=usage,
            )
            result: dict[str, object] = {
                "final_answer": getattr(response, "content", "") or "",
                "error": None,
                "standalone_question": latest_question or None,
                "citations": [],
                "reranker_docs": [],
                "context_usage": None,
                "mcp_used": False,
                "mcp_tools_used": [],
                "model_id": resolved_model_id,
                "usage": emitted_usage,
                "cost_usd": cost_usd,
            }
            if langfuse_trace is not None and langfuse_trace.trace_id:
                result["trace_id"] = langfuse_trace.trace_id
        assistant_message = assistant_message_from_result("direct", result)
    except Exception as exc:
        assistant_message = assistant_message_from_exception("direct", exc)
    return {
        "messages": [assistant_message],
        "references": assistant_message.additional_kwargs,
    }
