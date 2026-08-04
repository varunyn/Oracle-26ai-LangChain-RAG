from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.infrastructure.mcp_adapter_runtime import load_adapter_tools
from src.rag_agent.infrastructure.mcp_agent_executor import _build_tool_summary
from src.rag_agent.infrastructure.oci_models import get_llm
from src.rag_agent.prompts.mcp_agent_prompts import (
    SYSTEM_PROMPT_MIXED,
    TOOL_SUMMARY_PLACEHOLDER,
)
from src.rag_agent.runtime.memory import (
    chat_history_before_latest_user,
    latest_user_message,
)
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore


class ToolAgentTurn(TypedDict):
    chat_history: list[BaseMessage]
    model_id: str
    question: str
    run_config: RunnableConfig
    system_prompt: str
    tools: list[object]
    oracle_retrieval_evidence: OracleRetrievalEvidenceStore | None


def get_tool_agent_turn(runtime: Runtime[ChatGraphContext]) -> ToolAgentTurn:
    turn = get_runtime_context(runtime).get("tool_agent_turn")
    if not isinstance(turn, dict):
        raise RuntimeError("Tool-agent execution was not prepared.")
    return cast(ToolAgentTurn, turn)


def build_tool_agent_system_prompt(tools: Sequence[object]) -> str:
    return SYSTEM_PROMPT_MIXED.replace(
        TOOL_SUMMARY_PLACEHOLDER,
        _build_tool_summary(cast(Sequence, tools)),
    )


async def prepare_tool_agent_turn(
    *,
    state: ChatGraphState,
    parent_config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
    mode: Literal["mcp", "mixed"],
    extra_tools: Sequence[object] = (),
    oracle_retrieval_evidence: OracleRetrievalEvidenceStore | None = None,
) -> ToolAgentTurn:
    context = get_runtime_context(runtime)
    messages = state.get("messages", [])
    question = latest_user_message(messages)
    chat_history = chat_history_before_latest_user(messages)
    model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
    run_config = build_run_config(
        parent_config=parent_config,
        thread_id=get_thread_id(runtime),
        mode=mode,
        model_id=model_id,
        session_id=cast(str | None, context.get("session_id")),
        enable_tracing=cast(bool | None, context.get("enable_tracing")),
        mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
    )
    mcp_tools = await load_adapter_tools(
        server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
        run_config=run_config,
    )
    tools = [*extra_tools, *mcp_tools]
    return {
        "chat_history": chat_history,
        "model_id": model_id,
        "question": question,
        "run_config": run_config,
        "system_prompt": build_tool_agent_system_prompt(tools),
        "tools": tools,
        "oracle_retrieval_evidence": oracle_retrieval_evidence,
    }


__all__ = [
    "ToolAgentTurn",
    "build_tool_agent_system_prompt",
    "get_tool_agent_turn",
    "prepare_tool_agent_turn",
]
