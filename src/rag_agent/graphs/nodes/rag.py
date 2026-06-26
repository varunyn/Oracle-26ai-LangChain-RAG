from __future__ import annotations

from src.rag_agent.graphs.state import ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService


async def run_rag_node(state: ChatGraphState) -> ChatGraphState:
    context = state.get("context", {})
    service = ChatRuntimeService()
    result = await service.run_chat(
        messages=state["messages"],
        model_id=context.get("model_id"),
        thread_id=None,
        session_id=None,
        collection_name=context.get("collection_name"),
        enable_reranker=context.get("enable_reranker"),
        enable_tracing=context.get("enable_tracing"),
        mode="rag",
        mcp_server_keys=None,
        stream=False,
    )
    return {
        "messages": [*state["messages"], {"role": "assistant", "content": result["final_answer"]}],
        "context": context,
        "references": {"mode": "rag", **{k: v for k, v in result.items() if k != "final_answer"}},
    }
