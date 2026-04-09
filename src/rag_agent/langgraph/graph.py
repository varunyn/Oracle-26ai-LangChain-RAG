"""LangGraph-based orchestration for the RAG agent (graph composition entrypoint)."""

# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false, reportExplicitAny=false, reportMissingTypeArgument=false, reportConstantRedefinition=false

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Literal, cast

from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore[import-not-found]
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.rag_agent.agent_state import State
from src.rag_agent.core.node_logging import log_node_end, log_node_start
from src.rag_agent.langgraph.nodes.answer_generator import (
    STRUCTURED_FAILED_MESSAGE,
    AnswerFromDocs,
    DraftAnswer,
)
from src.rag_agent.langgraph.nodes.followup_interpreter import (
    FollowUpInterpreter,
    GroundedReformatAnswer,
)
from src.rag_agent.langgraph.nodes.mixed import CallMCPTools, DirectAnswer, SelectMCPTools
from src.rag_agent.langgraph.nodes.reranker import Reranker
from src.rag_agent.langgraph.nodes.router import Router
from src.rag_agent.langgraph.nodes.vector_search import SemanticSearch

# User-facing message when search (e.g. database) fails so we show it early instead of waiting for Answer.
SEARCH_ERROR_MESSAGE = (
    "Search is temporarily unavailable (e.g. database connection error). "
    "Please check your connection and try again later."
)


def route_after_router(state: State) -> Literal["followup", "select_mcp", "direct"] | str:
    """Route after Router node based on state['route']."""
    route = (state.get("route") or "followup").strip()
    return "followup" if route == "search" else route


def route_after_followup_interpreter(state: State) -> Literal["search", "reformat"]:
    return "reformat" if (state.get("followup_intent") or "").strip() == "reformat" else "search"


def route_after_search(state: State) -> Literal["error", "rerank"]:
    """Route after Search node based on error presence."""
    return "error" if state.get("error") else "rerank"


def _rag_failed(state: State) -> bool:
    rag_answer = (state.get("rag_answer") or "").strip().lower()
    if not rag_answer:
        return True
    if "i don't know" in rag_answer:
        return True
    if (state.get("rag_answer") or "") == STRUCTURED_FAILED_MESSAGE:
        return True
    docs = state.get("reranker_docs") or state.get("retriever_docs") or []
    return len(docs) == 0


def route_after_answer_from_docs(state: State) -> Literal["select_mcp", "draft"]:
    if (state.get("mode") or "").lower() == "mixed" and state.get("mcp_tool_match"):
        if _rag_failed(state) or not state.get("rag_has_citations"):
            return "select_mcp"
    return "draft"


def search_error_response(state: State) -> dict[str, str]:
    """Return a short final_answer so the UI can show the error early without running Rerank/Answer."""
    log_node_start("SearchErrorResponse")
    err = state.get("error") or "Unknown error"
    log_node_end("SearchErrorResponse", next="END")
    return {"final_answer": f"{SEARCH_ERROR_MESSAGE}\n\nDetails: {err}"}


# Lazy, process-wide default checkpointer (SQLite)
_DEFAULT_SQLITE_CHECKPOINTER: SqliteSaver | None = None


def _default_sqlite_path() -> Path:
    """Resolve default SQLite path under repo root unless overridden by env.

    Env: LANGGRAPH_SQLITE_PATH
    Default: <repo-root>/.local-data/langgraph-checkpoints.sqlite
    """
    env_path = os.getenv("LANGGRAPH_SQLITE_PATH")
    if env_path:
        return Path(env_path).expanduser()
    # Derive repo root from this file location: src/rag_agent/langgraph/graph.py -> repo root = parents[3]
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / ".local-data" / "langgraph-checkpoints.sqlite"


def get_default_checkpointer() -> SqliteSaver:
    """Return a process-wide SqliteSaver, creating parent dir if needed."""
    global _DEFAULT_SQLITE_CHECKPOINTER
    if _DEFAULT_SQLITE_CHECKPOINTER is None:
        db_path = _default_sqlite_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _DEFAULT_SQLITE_CHECKPOINTER = SqliteSaver(conn)
    return _DEFAULT_SQLITE_CHECKPOINTER


async def create_async_checkpointer() -> tuple[AsyncSqliteSaver, Any]:
    """Create an AsyncSqliteSaver and its aiosqlite connection for use in async contexts.

    Caller must keep the connection open for the lifetime of the graph and close it
    on shutdown (e.g. await conn.close()). Requires the aiosqlite package.
    """
    import aiosqlite

    db_path = _default_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    return AsyncSqliteSaver(conn), conn


def create_workflow(
    checkpointer: object | None = None,
) -> CompiledStateGraph:
    """
    Create the entire workflow with optional checkpointer for state persistence.

    Args:
        checkpointer: Optional checkpointer instance (e.g., SqliteSaver, PostgresSaver).
                 If None, uses a process-wide SqliteSaver at LANGGRAPH_SQLITE_PATH
                 (default: .local-data/langgraph-checkpoints.sqlite).

    Returns:
        Compiled StateGraph workflow with checkpointer support.
    """
    workflow = StateGraph(State)

    # create nodes (each is a Runnable)
    router = Router()
    semantic_search = SemanticSearch()
    reranker = Reranker()
    answer_from_docs = AnswerFromDocs()
    draft_answer = DraftAnswer()
    followup_interpreter = FollowUpInterpreter()
    grounded_reformat_answer = GroundedReformatAnswer()
    select_mcp_tools = SelectMCPTools()
    call_mcp_tools = CallMCPTools()
    direct_answer = DirectAnswer()

    workflow.add_node("Router", router)
    workflow.add_node("Search", semantic_search)
    workflow.add_node("SearchErrorResponse", search_error_response)
    workflow.add_node("Rerank", reranker)
    workflow.add_node("AnswerFromDocs", answer_from_docs)
    workflow.add_node("DraftAnswer", draft_answer)
    workflow.add_node("FollowUpInterpreter", followup_interpreter)
    workflow.add_node("GroundedReformatAnswer", grounded_reformat_answer)
    workflow.add_node("SelectMCPTools", select_mcp_tools)
    workflow.add_node("CallMCPTools", call_mcp_tools)
    workflow.add_node("DirectAnswer", direct_answer)

    workflow.add_edge(START, "Router")
    workflow.add_conditional_edges(
        "Router",
        route_after_router,
        {
            "followup": "FollowUpInterpreter",
            "select_mcp": "SelectMCPTools",
            "direct": "DirectAnswer",
        },
    )
    workflow.add_conditional_edges(
        "FollowUpInterpreter",
        route_after_followup_interpreter,
        {"search": "Search", "reformat": "GroundedReformatAnswer"},
    )
    workflow.add_edge("SelectMCPTools", "CallMCPTools")
    workflow.add_edge("CallMCPTools", "DraftAnswer")
    workflow.add_edge("DirectAnswer", "DraftAnswer")
    workflow.add_edge("GroundedReformatAnswer", "DraftAnswer")
    workflow.add_conditional_edges(
        "Search",
        route_after_search,
        {"error": "SearchErrorResponse", "rerank": "Rerank"},
    )
    workflow.add_edge("SearchErrorResponse", END)
    workflow.add_edge("Rerank", "AnswerFromDocs")
    workflow.add_conditional_edges(
        "AnswerFromDocs",
        route_after_answer_from_docs,
        {"select_mcp": "SelectMCPTools", "draft": "DraftAnswer"},
    )
    workflow.add_edge("DraftAnswer", END)

    memory = cast(Any, checkpointer) if checkpointer is not None else get_default_checkpointer()
    workflow_app = workflow.compile(checkpointer=memory)

    return workflow_app
