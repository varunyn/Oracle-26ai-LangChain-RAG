from __future__ import annotations

from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolCallTransformer
from langgraph.runtime import Runtime

from src.rag_agent.graphs.nodes.direct import run_direct_node
from src.rag_agent.graphs.nodes.mcp import run_mcp_node
from src.rag_agent.graphs.nodes.mixed import (
    run_mixed_compose_node,
    run_mixed_mcp_node,
)
from src.rag_agent.graphs.nodes.rag import run_rag_node
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.utils.langfuse_tracing import add_langfuse_callbacks


def _bootstrap_node(state: ChatGraphState) -> ChatGraphState:
    return state


def _mixed_route_node(state: ChatGraphState) -> ChatGraphState:
    return {"progress": "Planning collection and tool search…"}


def _mixed_retrieval_node(state: ChatGraphState) -> ChatGraphState:
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
    graph.add_node("mixed_route", _mixed_route_node)
    graph.add_node("mixed_retrieval", _mixed_retrieval_node)
    graph.add_node("rag_progress", _rag_progress_node)
    graph.add_node("direct", run_direct_node)
    graph.add_node("mcp", run_mcp_node)
    graph.add_node("mixed_mcp", run_mixed_mcp_node)
    graph.add_node("mixed_compose", run_mixed_compose_node)
    graph.add_node("rag", run_rag_node)
    graph.set_entry_point("bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        route_mode,
        {
            "direct": "direct",
            "mcp": "mcp",
            "mixed": "mixed_route",
            "rag": "rag_progress",
        },
    )
    graph.add_edge("mixed_route", "mixed_retrieval")
    graph.add_edge("mixed_retrieval", "mixed_mcp")
    graph.add_edge("mixed_mcp", "mixed_compose")
    graph.add_edge("rag_progress", "rag")
    graph.add_edge("direct", END)
    graph.add_edge("mcp", END)
    graph.add_edge("mixed_compose", END)
    graph.add_edge("rag", END)
    return graph.compile(checkpointer=checkpointer, transformers=[ToolCallTransformer])


def make_chat_agent(config: RunnableConfig | None = None) -> Any:
    """Build the Agent Server graph with one callback-owned tracing boundary."""
    run_config: dict[str, Any] = dict(config or {})
    configurable = run_config.get("configurable")
    if not isinstance(configurable, dict):
        configurable = {}
        run_config["configurable"] = configurable

    if configurable.get("enable_tracing") is True:
        add_langfuse_callbacks(
            run_config,
            session_id=configurable.get("session_id"),
            user_id=configurable.get("user_id"),
            request_id=configurable.get("request_id"),
            release=configurable.get("release"),
            trace_name="chat.request",
            tags=[
                tag
                for tag in (
                    "chat",
                    f"mode:{configurable.get('mode')}" if configurable.get("mode") else None,
                    f"model:{configurable.get('model_id')}"
                    if configurable.get("model_id")
                    else None,
                )
                if tag is not None
            ],
        )

    return build_chat_agent().with_config(run_config)


chat_agent = build_chat_agent()
