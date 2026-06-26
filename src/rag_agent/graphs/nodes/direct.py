from __future__ import annotations

from langgraph.runtime import Runtime

from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService
from src.rag_agent.runtime.memory import collapse_replayed_messages, langchain_messages_to_dicts


def _runtime_context(runtime: Runtime[ChatGraphContext]) -> ChatGraphContext:
    context = runtime.context
    if isinstance(context, dict):
        return context
    return {}


async def run_direct_node(
    state: ChatGraphState, runtime: Runtime[ChatGraphContext]
) -> ChatGraphState:
    context = _runtime_context(runtime)
    thread_id = getattr(runtime.execution_info, "thread_id", None)
    messages = langchain_messages_to_dicts(collapse_replayed_messages(state["messages"]))
    service = ChatRuntimeService()
    result = await service.run_chat(
        messages=messages,
        model_id=context.get("model_id"),
        thread_id=thread_id,
        session_id=None,
        collection_name=None,
        enable_reranker=False,
        enable_tracing=context.get("enable_tracing"),
        mode="direct",
        mcp_server_keys=context.get("mcp_server_keys"),
        stream=False,
    )
    return {
        "messages": [{"role": "assistant", "content": result["final_answer"]}],
        "references": {"mode": "direct", **{k: v for k, v in result.items() if k != "final_answer"}},
    }
