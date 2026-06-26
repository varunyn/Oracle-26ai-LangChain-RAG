from __future__ import annotations

from src.rag_agent.graphs.state import ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService


async def run_direct_node(state: ChatGraphState) -> ChatGraphState:
    service = ChatRuntimeService()
    result = await service.run_chat(
        messages=state["messages"],
        model_id=state.get("context", {}).get("model_id"),
        thread_id=None,
        session_id=None,
        collection_name=None,
        enable_reranker=False,
        enable_tracing=state.get("context", {}).get("enable_tracing"),
        mode="direct",
        mcp_server_keys=None,
        stream=False,
    )
    return {
        "messages": [*state["messages"], {"role": "assistant", "content": result["final_answer"]}],
        "context": state.get("context", {}),
        "references": {"mode": "direct", **{k: v for k, v in result.items() if k != "final_answer"}},
    }
