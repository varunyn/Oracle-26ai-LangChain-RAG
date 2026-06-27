from __future__ import annotations

from langgraph.runtime import Runtime

from src.rag_agent.graphs.nodes.references import (
    assistant_message_from_exception,
    assistant_message_from_result,
)
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService
from src.rag_agent.runtime.memory import langchain_messages_to_dicts


def _runtime_context(runtime: Runtime[ChatGraphContext]) -> ChatGraphContext:
    context = runtime.context
    if isinstance(context, dict):
        return context
    return {}


async def run_mixed_node(
    state: ChatGraphState, runtime: Runtime[ChatGraphContext]
) -> ChatGraphState:
    context = _runtime_context(runtime)
    thread_id = getattr(runtime.execution_info, "thread_id", None)
    messages = langchain_messages_to_dicts(state["messages"])
    try:
        result = await ChatRuntimeService().run_chat(
            messages=messages,
            model_id=context.get("model_id"),
            thread_id=thread_id,
            session_id=None,
            collection_name=context.get("collection_name"),
            enable_reranker=context.get("enable_reranker"),
            enable_tracing=context.get("enable_tracing"),
            mode="mixed",
            mcp_server_keys=context.get("mcp_server_keys"),
            stream=False,
        )
        assistant_message = assistant_message_from_result("mixed", result)
    except Exception as exc:
        assistant_message = assistant_message_from_exception("mixed", exc)
    return {
        "messages": [assistant_message],
        "references": assistant_message.additional_kwargs,
    }
