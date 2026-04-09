"""Agent state definition for the RAG workflow.

LangChain Best Practices (from langchain-docs/docs/src/oss/langgraph/memory.mdx):
- State is persisted via checkpointers, accumulating across thread invocations
- Chat history grows indefinitely and should be trimmed to prevent context overflow
- "messages alternate between human inputs and model responses, resulting in a list
  of messages that grows longer over time... many applications can benefit from
  using techniques to manually remove or forget stale information"

Task 14: State contract clarity
- Explicitly document required vs optional fields
- Classify persistent vs per-turn fields
- Provide lightweight helpers for validation/normalization used at node boundaries
"""

# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, NotRequired, Required, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from src.rag_agent.utils.context_window import messages_to_text as _messages_to_text

# ------------------------------
# Supporting entry types
# ------------------------------


class DocSerializable(TypedDict, total=False):
    page_content: str
    metadata: dict[str, object]


class CitationEntry(TypedDict, total=False):
    source: str
    page: int | str | None


class ContextUsage(TypedDict, total=False):
    tokens: int
    max: int
    percent: float
    model_id: str


class FollowUpInterpretation(TypedDict):
    intent: str
    standalone_question: str | None
    response_instruction: str | None
    reasoning: str


# ------------------------------
# State schema (TypedDict)
# ------------------------------


class State(TypedDict, total=False):
    """
    The state of the graph.

    IMPORTANT
    - messages accumulates across invocations within the same thread_id (persisted by the checkpointer)
    - Per-turn fields MUST be reset for each new user turn (the API resets them per request)
    - Nodes should only write atomic updates (return a partial dict with changed keys)

    """

    # Persistent across turns (conversation memory)
    user_request: Required[str]  # required for every turn
    # Canonical conversation memory (LangGraph best practice)
    messages: Required[
        Annotated[list[AnyMessage], add_messages]
    ]  # grows over time; trimmed upstream when needed

    # Produced during the turn
    standalone_question: NotRequired[str]
    history_text: NotRequired[str | None]
    followup_intent: NotRequired[str | None]
    response_instruction: NotRequired[str | None]
    latest_answer: NotRequired[str | None]

    # Search/Rerank artifacts (docs are dicts with page_content+metadata)
    retriever_docs: NotRequired[list[DocSerializable] | None]
    reranker_docs: NotRequired[list[DocSerializable] | None]

    # Final merged answer for the turn
    final_answer: NotRequired[str]

    # Citations corresponding 1:1 with reranker_docs order
    citations: NotRequired[list[CitationEntry]]

    # Context window usage snapshot for the turn
    context_usage: NotRequired[ContextUsage | None]

    # MCP usage flags for the turn
    mcp_used: NotRequired[bool | None]
    mcp_tools_used: NotRequired[list[str] | None]
    mcp_tool_match: NotRequired[bool | None]
    selected_mcp_tool_names: NotRequired[list[str] | None]
    selected_mcp_tool_descriptions: NotRequired[list[str] | None]

    # Routing
    mode: NotRequired[str | None]  # rag | mcp | mixed | direct
    route: NotRequired[str | None]  # rewrite | select_mcp | direct

    # Branch-specific answers
    rag_answer: NotRequired[str | None]
    rag_context: NotRequired[str | None]
    rag_has_citations: NotRequired[bool | None]
    mcp_answer: NotRequired[str | None]
    direct_answer: NotRequired[str | None]

    # MoreInfo loop
    round: NotRequired[int | None]
    max_rounds: NotRequired[int | None]

    # Error propagated to UI
    error: NotRequired[str | None]


# ------------------------------
# State contract metadata
# ------------------------------

# Immutable classification used by helpers and documentation
PERSISTENT_KEYS: tuple[str, ...] = (
    "user_request",
    "messages",
)

PER_TURN_KEYS: tuple[str, ...] = (
    # routing/flow
    "mode",
    "route",
    # rewrite/search/rerank
    "standalone_question",
    "history_text",
    "followup_intent",
    "response_instruction",
    "latest_answer",
    "retriever_docs",
    "reranker_docs",
    "citations",
    # answers
    "rag_answer",
    "rag_context",
    "rag_has_citations",
    "mcp_answer",
    "direct_answer",
    "final_answer",
    # tools/usage
    "mcp_used",
    "mcp_tools_used",
    "mcp_tool_match",
    "selected_mcp_tool_names",
    "selected_mcp_tool_descriptions",
    "context_usage",
    # control + errors
    "round",
    "max_rounds",
    "error",
)

# Minimal required keys per node (read-before-write)
REQUIRED_BY_NODE: dict[str, tuple[str, ...]] = {
    "Router": ("user_request", "messages"),
    "Search": tuple(),
    "Rerank": tuple(),  # accepts empty docs
    "AnswerFromDocs": tuple(),  # accepts empty docs
    "SelectMCPTools": tuple(),
    "CallMCPTools": ("user_request", "messages"),
    "DirectAnswer": ("user_request",),
    "DraftAnswer": tuple(),
    "SearchErrorResponse": tuple(),
}


# ------------------------------
# Lightweight helpers
# ------------------------------


def ensure_required(state: State, node: str) -> None:
    """Validate required keys for a node.

    O(1) checks; raises ValueError with a clear message (mirrors KeyError behavior
    you'd get from direct indexing but is easier to surface in logs/tests).
    """
    required = REQUIRED_BY_NODE.get(node, tuple())
    for key in required:
        if key not in state or state.get(key) in (None, ""):
            raise ValueError(f"State missing required key '{key}' for node '{node}'")


def per_turn_reset(keys: Iterable[str] | None = None) -> dict[str, object]:
    """Return an atomic update that clears per-turn fields.

    The API already clears critical per-turn fields at the start of each turn;
    this helper is provided for internal flows or tests that wish to reset state
    explicitly. It does not modify persistent keys.
    """
    to_clear = set(keys or PER_TURN_KEYS) - set(PERSISTENT_KEYS)
    out: dict[str, object] = {}
    for k in to_clear:
        if k in ("citations", "retriever_docs", "reranker_docs", "mcp_tools_used"):
            out[k] = []
        else:
            out[k] = None
    return out


def messages_text_from_state(state: State) -> str:
    """Serialize state messages to text using existing messages_to_text helper."""
    msgs = state.get("messages") or []
    try:
        return _messages_to_text(list(msgs))
    except Exception:
        return ""
