from __future__ import annotations

from langgraph.runtime import Runtime

from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService


async def run_rag_node(
    state: ChatGraphState, runtime: Runtime[ChatGraphContext]
) -> ChatGraphState:
    context = runtime.context
    thread_id = getattr(runtime.execution_info, "thread_id", None)
    service = ChatRuntimeService()
    result = await service.run_chat(
        messages=state["messages"],
        model_id=context.get("model_id"),
        thread_id=thread_id,
        session_id=None,
        collection_name=context.get("collection_name"),
        enable_reranker=context.get("enable_reranker"),
        enable_tracing=context.get("enable_tracing"),
        mode="rag",
        mcp_server_keys=context.get("mcp_server_keys"),
        stream=False,
    )
    return {
        "messages": [*state["messages"], {"role": "assistant", "content": result["final_answer"]}],
        "context": context,
        "references": {"mode": "rag", **{k: v for k, v in result.items() if k != "final_answer"}},
    }
