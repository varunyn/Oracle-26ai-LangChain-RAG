from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
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
from src.rag_agent.graphs.nodes.references import (
    messages_from_result,
)
from src.rag_agent.graphs.runtime import get_runtime_context
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState, MCPSubGraphState
from src.rag_agent.graphs.tool_agent_turn import get_tool_agent_turn, prepare_tool_agent_turn
from src.rag_agent.infrastructure.oci_models import get_llm
from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.mcp_turn import tool_failure_summary
from src.rag_agent.runtime.memory import latest_user_message
from src.rag_agent.runtime.oracle_retrieval_evidence import OracleRetrievalEvidenceStore

MCP_MAX_ROUNDS = 10


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            parts.append(str(item))
        return "".join(parts).strip()
    return str(content or "").strip()


def _latest_agent_final_answer(state_messages: list[object]) -> str | None:
    for message in reversed(state_messages):
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            continue
        content = _content_text(message.content)
        if content and not is_trivial_answer(content):
            return content
    return None


def message_to_langchain(m: object) -> BaseMessage | None:
    if m is None:
        return None
    if isinstance(m, Mapping):
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or "")
        if not content:
            return None
        if role in {"assistant", "ai"}:
            return AIMessage(content=content)
        return HumanMessage(content=content)
    msg_type = str(getattr(m, "type", "") or getattr(m, "role", "") or "").strip().lower()
    content = str(getattr(m, "content", "") or "")
    if not content:
        return None
    if msg_type in {"assistant", "ai"}:
        return AIMessage(content=content)
    return HumanMessage(content=content)


_OCI_SUPPORTED_CONTENT_TYPES = {
    "text",
    "image_url",
    "document_url",
    "document",
    "file",
    "video_url",
    "video",
    "audio_url",
    "audio",
    "media",
}


def _sanitize_for_oci(message: object) -> object:
    if not isinstance(message, AIMessage) or not isinstance(message.content, list):
        return message
    content: list[str | dict] = []
    for item in message.content:
        if isinstance(item, str):
            if item:
                content.append(item)
            continue
        if not isinstance(item, dict):
            continue
        ctype = item.get("type")
        if ctype == "tool_call":
            continue
        if ctype in _OCI_SUPPORTED_CONTENT_TYPES:
            content.append(dict(item))
            continue
        if "text" in item and ctype is None:
            text = item.get("text")
            if isinstance(text, str) and text:
                content.append({"type": "text", "text": text})
    if not content and (message.tool_calls or message.additional_kwargs.get("tool_calls")):
        content = [{"type": "text", "text": "."}]
    if content == message.content:
        return message
    copy = getattr(message, "model_copy", None)
    if callable(copy):
        return copy(update={"content": content})
    return AIMessage(
        content=content,
        additional_kwargs=dict(message.additional_kwargs),
        response_metadata=dict(message.response_metadata),
        tool_calls=list(message.tool_calls),
        id=message.id,
        name=message.name,
    )


async def call_llm_node(
    state: MCPSubGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    turn = get_tool_agent_turn(runtime)
    tools = turn["tools"]
    model_id = turn["model_id"]
    model = get_llm(model_id=model_id)
    if tools:
        model = model.bind_tools(list(tools))
    remaining = state.get("remaining_steps", MCP_MAX_ROUNDS)
    if remaining <= 0:
        return {"messages": [AIMessage(content="Tool call limit reached.")]}

    full_messages: list[object] = []
    full_messages.append(SystemMessage(content=turn["system_prompt"]))
    for item in turn["chat_history"]:
        converted = message_to_langchain(item)
        if converted is not None:
            full_messages.append(converted)
    full_messages.append(HumanMessage(content=turn["question"]))
    full_messages.extend(state.get("messages", []))

    sanitized = [_sanitize_for_oci(m) for m in full_messages]
    response = await model.ainvoke(sanitized, config=config)
    if (
        isinstance(response, AIMessage)
        and response.tool_calls
        and _content_text(response.content) == "."
    ):
        response = AIMessage(
            content="",
            additional_kwargs=dict(response.additional_kwargs),
            response_metadata=dict(response.response_metadata),
            tool_calls=list(response.tool_calls),
            id=response.id,
            name=response.name,
        )
    return {"messages": [response], "remaining_steps": remaining - 1}


async def run_tools_node(
    state: MCPSubGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    tools = get_tool_agent_turn(runtime)["tools"]
    if not tools:
        return {"messages": []}
    tool_node = ToolNode(list(tools))
    result = await tool_node.ainvoke({"messages": [state["messages"][-1]]}, config=config)
    return {"messages": cast(list, result.get("messages", []))}


def route(state: MCPSubGraphState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "run_tools"
    return "__end__"


def extract_tool_invocations_from_messages(messages: list[object]) -> list[dict[str, object]]:
    pending: dict[str, dict[str, object]] = {}
    invocations: list[dict[str, object]] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in cast(list[dict], msg.tool_calls or []):
                tc_id = str(tc.get("id", "") or "")
                name = str(tc.get("name", "") or "")
                if name:
                    pending[tc_id] = {
                        "invocation_id": tc_id,
                        "tool_name": name,
                        "args": tc.get("args", {}),
                    }
            continue
        if isinstance(msg, ToolMessage):
            tc_id = str(getattr(msg, "tool_call_id", "") or "")
            content = str(getattr(msg, "content", "") or "")
            name = str(getattr(msg, "name", "") or "")
            error_text = content if getattr(msg, "status", "") == "error" else None
            rec = pending.pop(
                tc_id,
                {"invocation_id": tc_id, "tool_name": name, "args": None},
            )
            rec["result"] = content
            if error_text:
                rec["error"] = error_text
            invocations.append(rec)
            continue
    return invocations


def build_tool_agent_sub_graph() -> StateGraph:
    sub_graph = StateGraph(MCPSubGraphState, context_schema=ChatGraphContext)
    sub_graph.add_node("call_llm", call_llm_node)
    sub_graph.add_node("run_tools", run_tools_node)
    sub_graph.add_conditional_edges("call_llm", route, {"run_tools": "run_tools", "__end__": END})
    sub_graph.add_edge("run_tools", "call_llm")
    sub_graph.set_entry_point("call_llm")
    return sub_graph.compile()


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
    runtime.context["tool_agent_turn"] = turn

    return {
        "messages": [],
        "progress": "Planning collection and tool search…",
    }


async def run_mixed_compose_node(
    state: ChatGraphState,
    _config: RunnableConfig | None = None,
    _runtime: Runtime[ChatGraphContext] | None = None,
) -> ChatGraphState:
    messages = state.get("messages", [])
    if not messages:
        result = {"final_answer": "Mixed-mode execution did not produce a result."}
        return {"messages": messages_from_result("mixed", result, []), "references": {}}

    context = get_runtime_context(_runtime) if _runtime else {}
    turn = get_tool_agent_turn(_runtime) if _runtime else None
    question = turn["question"] if turn else latest_user_message(messages) or ""
    tool_invocations = extract_tool_invocations_from_messages(messages)
    tools_used = list({inv["tool_name"] for inv in tool_invocations})
    final_answer = _latest_agent_final_answer(messages) or ""

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
        retrieval_docs = rag_runtime.rerank_retrieved_docs(
            question,
            cast(list[Any], retrieval_docs),
            enable_reranker=cast(bool | None, context.get("enable_reranker")),
        )
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
    if not policy_error and retrieval_docs and _latest_agent_final_answer(messages) is None:
        supplemental_context = mixed_tool_supplemental_context(
            cast(list[dict[str, object]], tool_invocations)
        )
        final_answer, _rag_usage, _ = await rag_runtime.synthesize_rag_answer(
            question=question,
            docs=cast(list[Any], retrieval_docs),
            model_id=cast(str | None, context.get("model_id")),
            run_config=turn["run_config"] if turn else None,
            supplemental_context=supplemental_context,
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
    messages_out = messages_from_result("mixed", result, messages)
    references = cast(dict[str, object], getattr(messages_out[-1], "additional_kwargs", {}) or {})
    return {
        "messages": messages_out,
        "references": references,
    }
