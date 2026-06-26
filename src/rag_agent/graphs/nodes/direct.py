from __future__ import annotations

from langgraph.runtime import Runtime

from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.runtime.chat_service import ChatRuntimeService


async def run_direct_node(
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
        collection_name=None,
        enable_reranker=False,
        enable_tracing=context.get("enable_tracing"),
        mode="direct",
        mcp_server_keys=context.get("mcp_server_keys"),
        stream=False,
    )
    return {
        "messages": [*state["messages"], {"role": "assistant", "content": result["final_answer"]}],
        "context": context,
        "references": {"mode": "direct", **{k: v for k, v in result.items() if k != "final_answer"}},
    }
