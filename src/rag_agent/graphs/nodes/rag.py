from __future__ import annotations

import asyncio
from typing import cast

from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from src.rag_agent.graphs.nodes.references import (
    assistant_message_from_exception,
    assistant_message_from_result,
)
from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.memory import (
    chat_history_before_latest_user,
    contextualize_question,
    latest_user_message,
    latest_user_message_id,
)
from src.rag_agent.runtime.observability import emit_usage_observability

_NO_ORACLE_CONTEXT_ANSWER = "I don't know the answer from the selected Oracle collection."


async def run_rag_node(
    state: ChatGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    context = get_runtime_context(runtime)
    thread_id = get_thread_id(runtime)
    messages = state["messages"]
    assistant_id = latest_user_message_id(messages)
    if assistant_id:
        assistant_id = f"{assistant_id}:assistant"
    try:
        question = latest_user_message(messages)
        chat_history = chat_history_before_latest_user(messages)
        run_cfg = build_run_config(
            parent_config=config,
            thread_id=thread_id,
            mode="rag",
            model_id=cast(str | None, context.get("model_id")),
            session_id=cast(str | None, context.get("session_id")),
            enable_tracing=cast(bool | None, context.get("enable_tracing")),
            mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
        )
        standalone_question = await contextualize_question(
            question=question,
            chat_history=chat_history,
            model_id=cast(str | None, context.get("model_id")),
            run_config=run_cfg,
        )
        docs = await asyncio.to_thread(
            rag_runtime.retrieve_oracle_docs,
            query=standalone_question,
            collection_name=cast(str | None, context.get("collection_name")),
            k=10,
        )
        docs = await asyncio.to_thread(
            rag_runtime.rerank_retrieved_docs,
            standalone_question,
            docs,
            enable_reranker=cast(bool | None, context.get("enable_reranker")),
        )
        if docs:
            rag_answer, rag_usage, resolved_model_id = await rag_runtime.synthesize_rag_answer(
                question=standalone_question,
                docs=docs,
                model_id=cast(str | None, context.get("model_id")),
                run_config=run_cfg,
            )
        else:
            rag_answer = _NO_ORACLE_CONTEXT_ANSWER
            rag_usage = None
            resolved_model_id = cast(str | None, context.get("model_id")) or "unknown"
        emitted_usage, cost_usd = emit_usage_observability(
            mode="rag",
            model_id=resolved_model_id,
            session_id=cast(str | None, context.get("session_id")),
            thread_id=thread_id,
            usage=rag_usage,
        )
        result: dict[str, object] = {
            "final_answer": rag_answer,
            "error": None,
            "standalone_question": standalone_question or None,
            "citations": rag_runtime.citations_from_docs(docs),
            "reranker_docs": rag_runtime.serialize_docs(docs),
            "context_usage": None,
            "mcp_used": False,
            "mcp_tools_used": [],
            "model_id": resolved_model_id,
            "usage": emitted_usage,
            "cost_usd": cost_usd,
        }
        assistant_message = assistant_message_from_result(
            "rag", result, message_id=assistant_id
        )
    except Exception as exc:
        assistant_message = assistant_message_from_exception(
            "rag", exc, message_id=assistant_id
        )
    return {
        "messages": [assistant_message],
        "references": assistant_message.additional_kwargs,
    }
