from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from langchain_core.messages import AnyMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from src.rag_agent.graphs.mcp_policies import (
    NO_ORACLE_CONTEXT_ANSWER,
    ORACLE_RETRIEVAL_FAILED_ANSWER,
    enforce_workflow_policy,
    is_trivial_answer,
    mixed_tool_supplemental_context,
    oracle_retrieval_error,
    oracle_retrieval_used_without_context,
    workflow_policy_for_request,
)
from src.rag_agent.graphs.nodes.references import messages_from_result
from src.rag_agent.graphs.runtime import get_runtime_context, stable_terminal_message_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.graphs.tool_agent_execution import (
    analyze_tool_execution,
    messages_since_latest_user,
)
from src.rag_agent.graphs.tool_agent_turn import (
    ToolAgentTurn,
    mark_tool_agent_turn_terminal,
    prepare_tool_agent_turn,
    reconstruct_tool_agent_turn,
    release_tool_agent_turn,
    release_tool_agent_turn_after_failure,
    run_with_lease_heartbeat,
)
from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.mcp_turn import tool_failure_summary
from src.rag_agent.runtime.memory import latest_user_message
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore

logger = logging.getLogger(__name__)


async def run_mixed_mcp_setup(
    state: ChatGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    context = get_runtime_context(runtime)
    retrieval_evidence = OracleRetrievalEvidenceStore()
    retrieval_tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name=cast(str | None, context.get("collection_name")),
        filter_docs=rag_runtime.filter_retrieved_docs,
        evidence=retrieval_evidence,
    )
    turn = await prepare_tool_agent_turn(
        state=state,
        parent_config=config,
        runtime=runtime,
        mode="mixed",
        extra_tools=[retrieval_tool] if retrieval_tool else [],
        oracle_retrieval_evidence=retrieval_evidence,
    )
    await release_tool_agent_turn(config, turn)
    return {"messages": [], "progress": "Planning collection and tool search…"}


async def run_mixed_compose_node(
    state: ChatGraphState,
    _config: RunnableConfig | None = None,
    _runtime: Runtime[ChatGraphContext] | None = None,
) -> ChatGraphState:
    messages = state.get("messages", [])
    turn = None
    if messages and _runtime and _config:
        turn = await reconstruct_tool_agent_turn(
            state=state, parent_config=_config, runtime=_runtime, mode="mixed"
        )
    try:
        result = await _compose_mixed_result(state, _config, _runtime, turn)
    except BaseException:
        if turn and _config:
            await _release_after_failure(_config, turn)
        raise
    if turn and _config:
        try:
            await _mark_terminal(_config, turn)
            await release_tool_agent_turn(_config, turn)
        except BaseException:
            await _release_after_failure(_config, turn)
            raise
    return result


async def _compose_mixed_result(
    state: ChatGraphState,
    _config: RunnableConfig | None = None,
    _runtime: Runtime[ChatGraphContext] | None = None,
    turn: ToolAgentTurn | None = None,
) -> ChatGraphState:
    messages = state.get("messages", [])
    message_id = _terminal_message_id("mixed", turn)
    if not messages:
        empty_result: dict[str, object] = {
            "final_answer": "Mixed-mode execution did not produce a result."
        }
        return {
            "messages": cast(
                list[AnyMessage],
                messages_from_result("mixed", empty_result, [], message_id=message_id),
            ),
            "references": {},
        }

    context = get_runtime_context(_runtime) if _runtime else {}
    question = turn["question"] if turn else latest_user_message(messages) or ""
    transcript = analyze_tool_execution(messages_since_latest_user(cast(list[object], messages)))
    tool_invocations = transcript["tool_invocations"]
    tools_used = transcript["tools_used"]
    final_answer = transcript["final_answer"]

    evidence_store = turn["oracle_retrieval_evidence"] if turn else None
    retrieval_evidence = evidence_store.read() if evidence_store else None
    retrieval_docs = list(retrieval_evidence.documents) if retrieval_evidence else []

    workflow_policy = workflow_policy_for_request(mode="mixed", question=question)
    policy_applied, missing_capabilities, policy_failure_message = enforce_workflow_policy(
        policy=workflow_policy,
        tools_used=tools_used,
        tool_invocations=cast(list[dict[str, object]], tool_invocations),
    )
    policy_error = policy_failure_message if policy_applied and missing_capabilities else None
    if policy_error:
        final_answer = policy_error
    tool_failure_error = tool_failure_summary(cast(list[dict[str, object]], tool_invocations))
    if not policy_error and is_trivial_answer(final_answer) and tool_failure_error:
        final_answer = tool_failure_error
        policy_error = tool_failure_error
    if retrieval_docs and question:

        def rerank_operation():
            return asyncio.to_thread(
                rag_runtime.rerank_retrieved_docs,
                question,
                cast(list[Any], retrieval_docs),
                enable_reranker=(
                    turn.get("enable_reranker")
                    if turn is not None
                    else cast(bool | None, context.get("enable_reranker"))
                ),
            )

        if turn and _config:
            retrieval_docs = await run_with_lease_heartbeat(
                _config,
                turn,
                rerank_operation,
                runtime=_runtime,
            )
        else:
            retrieval_docs = await rerank_operation()
    retrieval_error = oracle_retrieval_error(
        retrieval_evidence=retrieval_evidence,
        tools_used=tools_used,
        tool_invocations=cast(list[dict[str, object]], tool_invocations),
    )
    if not policy_error and retrieval_error:
        final_answer = ORACLE_RETRIEVAL_FAILED_ANSWER
        policy_error = ORACLE_RETRIEVAL_FAILED_ANSWER
    if not policy_error and oracle_retrieval_used_without_context(
        retrieval_evidence=retrieval_evidence,
        tools_used=tools_used,
        tool_invocations=cast(list[dict[str, object]], tool_invocations),
    ):
        final_answer = NO_ORACLE_CONTEXT_ANSWER
    if not policy_error and retrieval_docs and not transcript["has_terminal_answer"]:
        supplemental_context = mixed_tool_supplemental_context(
            cast(list[dict[str, object]], tool_invocations)
        )
        synthesis_kwargs: Any = {
            "question": question,
            "docs": cast(list[Any], retrieval_docs),
            "model_id": turn["model_id"] if turn else cast(str | None, context.get("model_id")),
            "run_config": turn["run_config"] if turn else None,
            "supplemental_context": supplemental_context,
        }
        if turn and _config:
            final_answer, _rag_usage, _ = await run_with_lease_heartbeat(
                _config,
                turn,
                lambda: rag_runtime.synthesize_rag_answer(**synthesis_kwargs),
                runtime=_runtime,
            )
        else:
            final_answer, _rag_usage, _ = await rag_runtime.synthesize_rag_answer(
                **synthesis_kwargs
            )

    result: dict[str, object] = {
        "final_answer": final_answer,
        "error": policy_error,
        "outcome": "error" if policy_error else "success",
        "standalone_question": question or None,
        "citations": rag_runtime.citations_from_docs(cast(list[Any], retrieval_docs)),
        "reranker_docs": rag_runtime.serialize_docs(cast(list[Any], retrieval_docs)),
        "context_usage": {"retrieved_docs_count": len(retrieval_docs)} if retrieval_docs else None,
        "mcp_used": bool(tools_used),
        "mcp_tools_used": tools_used,
        "mcp_tool_invocations": tool_invocations,
    }
    messages_out = messages_from_result("mixed", result, messages, message_id=message_id)
    references = cast(dict[str, object], getattr(messages_out[-1], "additional_kwargs", {}) or {})
    return {
        "messages": cast(list[AnyMessage], messages_out),
        "references": references,
    }


def _terminal_message_id(mode: str, turn: ToolAgentTurn | None) -> str | None:
    lease = turn.get("lease") if turn else None
    thread_id = getattr(lease, "thread_id", None)
    turn_id = getattr(lease, "turn_id", None)
    if isinstance(thread_id, str) and isinstance(turn_id, str):
        return cast(str, stable_terminal_message_id(mode, thread_id, turn_id))
    return None


async def _mark_terminal(config: RunnableConfig, turn: ToolAgentTurn) -> None:
    message_id = _terminal_message_id("mixed", turn)
    if message_id is None:
        return
    await mark_tool_agent_turn_terminal(config, turn, message_id)


async def _release_after_failure(config: RunnableConfig, turn: ToolAgentTurn) -> None:
    try:
        await release_tool_agent_turn_after_failure(config, turn)
    except BaseException:
        logger.warning("Failed stale tool-agent lease cleanup", exc_info=True)
