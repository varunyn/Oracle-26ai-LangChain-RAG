from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables.config import RunnableConfig
from langgraph.config import get_config
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph, StateNode
from langgraph.prebuilt import ToolCallTransformer
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from src.rag_agent.graphs.nodes.direct import run_direct_node
from src.rag_agent.graphs.nodes.mcp import run_mcp_compose, run_mcp_setup
from src.rag_agent.graphs.nodes.mixed import run_mixed_compose_node, run_mixed_mcp_setup
from src.rag_agent.graphs.nodes.rag import run_rag_node
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.graphs.tool_agent_execution import build_tool_agent_sub_graph
from src.rag_agent.utils.langfuse_tracing import add_langfuse_callbacks


def _bootstrap_node(state: ChatGraphState) -> ChatGraphState:
    return state


def _mixed_route_node(state: ChatGraphState) -> ChatGraphState:
    return {"progress": "Planning collection and tool search…"}


def _mixed_retrieval_node(state: ChatGraphState) -> ChatGraphState:
    return {"progress": "Searching your collection and tools…"}


def _rag_progress_node(_state: ChatGraphState) -> dict[str, str]:
    return {"progress": "Searching your collection…"}


async def _run_direct_graph_node(
    state: ChatGraphState,
    *,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    return await run_direct_node(state, get_config(), runtime)


async def _run_mcp_setup_graph_node(
    state: ChatGraphState,
    *,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    return await run_mcp_setup(state, get_config(), runtime)


async def _run_mcp_compose_graph_node(
    state: ChatGraphState,
    *,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    return await run_mcp_compose(state, get_config(), runtime)


async def _run_mixed_setup_graph_node(
    state: ChatGraphState,
    *,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    return await run_mixed_mcp_setup(state, get_config(), runtime)


async def _run_mixed_compose_graph_node(
    state: ChatGraphState,
    *,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    return await run_mixed_compose_node(state, get_config(), runtime)


async def _run_rag_graph_node(
    state: ChatGraphState,
    *,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    return await run_rag_node(state, get_config(), runtime)


def _runtime_context(runtime: Runtime[ChatGraphContext]) -> ChatGraphContext:
    context = runtime.context
    if isinstance(context, dict):
        return context
    return {}


def route_mode(_state: ChatGraphState, runtime: Runtime[ChatGraphContext]) -> str:
    mode = _runtime_context(runtime).get("mode", "direct")
    if isinstance(mode, str) and mode in {"direct", "rag", "mcp", "mixed"}:
        return mode
    raise NotImplementedError(f"Graph mode '{mode}' is not implemented yet.")


def build_chat_agent(
    *, checkpointer: Checkpointer = None
) -> CompiledStateGraph[ChatGraphState, ChatGraphContext, ChatGraphState, ChatGraphState]:
    graph: StateGraph[ChatGraphState, ChatGraphContext, ChatGraphState, ChatGraphState] = (
        StateGraph(ChatGraphState, context_schema=ChatGraphContext)
    )
    graph.add_node("bootstrap", _bootstrap_node)
    graph.add_node("mixed_route", _mixed_route_node)
    graph.add_node("mixed_retrieval", _mixed_retrieval_node)
    graph.add_node(
        "rag_progress",
        cast(StateNode[ChatGraphState, ChatGraphContext], _rag_progress_node),
    )
    graph.add_node("direct", _run_direct_graph_node)
    graph.add_node("mcp_setup", _run_mcp_setup_graph_node)
    mcp_agent = build_tool_agent_sub_graph()
    graph.add_node("mcp_agent", mcp_agent)
    graph.add_node("mcp_compose", _run_mcp_compose_graph_node)
    graph.add_node("mixed_setup", _run_mixed_setup_graph_node)
    mixed_agent = build_tool_agent_sub_graph()
    graph.add_node("mixed_agent", mixed_agent)
    graph.add_node("mixed_compose", _run_mixed_compose_graph_node)
    graph.add_node("rag", _run_rag_graph_node)
    graph.set_entry_point("bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        route_mode,
        {
            "direct": "direct",
            "mcp": "mcp_setup",
            "mixed": "mixed_route",
            "rag": "rag_progress",
        },
    )
    graph.add_edge("mixed_route", "mixed_retrieval")
    graph.add_edge("mixed_retrieval", "mixed_setup")
    graph.add_edge("mixed_setup", "mixed_agent")
    graph.add_edge("mixed_agent", "mixed_compose")
    graph.add_edge("mcp_setup", "mcp_agent")
    graph.add_edge("mcp_agent", "mcp_compose")
    graph.add_edge("rag_progress", "rag")
    graph.add_edge("direct", END)
    graph.add_edge("mcp_compose", END)
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
                    (
                        f"model:{configurable.get('model_id')}"
                        if configurable.get("model_id")
                        else None
                    ),
                )
                if tag is not None
            ],
        )

    return build_chat_agent().with_config(cast(RunnableConfig, run_config))


chat_agent = build_chat_agent()
