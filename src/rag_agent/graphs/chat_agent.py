from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolCallTransformer
from langgraph.runtime import Runtime

from src.rag_agent.graphs.nodes.direct import run_direct_node
from src.rag_agent.graphs.nodes.mcp import run_mcp_node
from src.rag_agent.graphs.nodes.mixed import run_mixed_node
from src.rag_agent.graphs.nodes.rag import run_rag_node
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState


def _bootstrap_node(state: ChatGraphState) -> ChatGraphState:
    return state


def _mixed_progress_node(_state: ChatGraphState) -> ChatGraphState:
    return {"progress": "Searching your collection and tools…"}


def _rag_progress_node(_state: ChatGraphState) -> ChatGraphState:
    return {"progress": "Searching your collection…"}


def _runtime_context(runtime: Runtime[ChatGraphContext]) -> ChatGraphContext:
    context = runtime.context
    if isinstance(context, dict):
        return context
    return {}


def route_mode(_state: ChatGraphState, runtime: Runtime[ChatGraphContext]) -> str:
    mode = _runtime_context(runtime).get("mode", "direct")
    if mode in {"direct", "rag", "mcp", "mixed"}:
        return mode
    raise NotImplementedError(f"Graph mode '{mode}' is not implemented yet.")


def build_chat_agent(*, checkpointer: Any | None = None) -> CompiledStateGraph:
    graph = StateGraph(ChatGraphState, context_schema=ChatGraphContext)
    graph.add_node("bootstrap", _bootstrap_node)
    graph.add_node("mixed_progress", _mixed_progress_node)
    graph.add_node("rag_progress", _rag_progress_node)
    graph.add_node("direct", run_direct_node)
    graph.add_node("mcp", run_mcp_node)
    graph.add_node("mixed", run_mixed_node)
    graph.add_node("rag", run_rag_node)
    graph.set_entry_point("bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        route_mode,
        {
            "direct": "direct",
            "mcp": "mcp",
            "mixed": "mixed_progress",
            "rag": "rag_progress",
        },
    )
    graph.add_edge("mixed_progress", "mixed")
    graph.add_edge("rag_progress", "rag")
    graph.add_edge("direct", END)
    graph.add_edge("mcp", END)
    graph.add_edge("mixed", END)
    graph.add_edge("rag", END)
    return graph.compile(checkpointer=checkpointer, transformers=[ToolCallTransformer])


chat_agent = build_chat_agent()
