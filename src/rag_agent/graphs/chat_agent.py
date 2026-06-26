from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState


def _bootstrap_node(state: ChatGraphState) -> ChatGraphState:
    return state


def build_chat_agent() -> CompiledStateGraph:
    graph = StateGraph(ChatGraphState, context_schema=ChatGraphContext)
    graph.add_node("bootstrap", _bootstrap_node)
    graph.set_entry_point("bootstrap")
    graph.add_edge("bootstrap", END)
    return graph.compile()


chat_agent = build_chat_agent()
