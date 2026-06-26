from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.rag_agent.graphs.nodes.direct import run_direct_node
from src.rag_agent.graphs.nodes.rag import run_rag_node
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState


def _bootstrap_node(state: ChatGraphState) -> ChatGraphState:
    return state


def route_mode(state: ChatGraphState) -> str:
    mode = state.get("context", {}).get("mode", "direct")
    if mode == "rag":
        return "rag"
    return "direct"


def build_chat_agent() -> CompiledStateGraph:
    graph = StateGraph(ChatGraphState, context_schema=ChatGraphContext)
    graph.add_node("bootstrap", _bootstrap_node)
    graph.add_node("direct", run_direct_node)
    graph.add_node("rag", run_rag_node)
    graph.set_entry_point("bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        route_mode,
        {
            "direct": "direct",
            "rag": "rag",
        },
    )
    graph.add_edge("direct", END)
    graph.add_edge("rag", END)
    return graph.compile()


chat_agent = build_chat_agent()
