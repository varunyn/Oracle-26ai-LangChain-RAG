from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
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
    repeated_workflow_controller_enabled,
    require_tool_call_enabled,
    workflow_checkpoint_path,
    workflow_policy_for_request,
)
from src.rag_agent.graphs.nodes.references import (
    assistant_message_from_exception,
    messages_from_result,
)
from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState, MCPSubGraphState
from src.rag_agent.infrastructure import oci_models as _oci_models
from src.rag_agent.runtime import rag_runtime
from src.rag_agent.runtime.mcp_turn import run_mcp_agent_turn, tool_failure_summary
from src.rag_agent.runtime.memory import (
    chat_history_before_latest_user,
    latest_user_message,
)


def get_llm(model_id: str | None = None) -> Any:
    return _oci_models.get_llm(model_id=model_id)


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


def _message_to_langchain(m: object) -> BaseMessage | None:
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


def _build_system_prompt_tools(question: str, tools: list[object]) -> str:
    from src.rag_agent.prompts.mcp_agent_prompts import SYSTEM_PROMPT_MIXED, TOOL_SUMMARY_PLACEHOLDER
    from src.rag_agent.infrastructure.mcp_agent_executor import _build_tool_summary
    return SYSTEM_PROMPT_MIXED.replace(TOOL_SUMMARY_PLACEHOLDER, _build_tool_summary(tools))


async def call_llm_node(
    state: MCPSubGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    context = get_runtime_context(runtime)
    tools = cast(list | None, context.get("mcp_subgraph_tools"))
    model_id = cast(str | None, context.get("mcp_subgraph_model_id")) or get_llm().model_id
    model = get_llm(model_id=model_id)
    if tools:
        model = model.bind_tools(list(tools))
    remaining = state.get("remaining_steps", MCP_MAX_ROUNDS)
    if remaining <= 0:
        return {"messages": [AIMessage(content="Tool call limit reached.")]}
    response = await model.ainvoke(state["messages"], config=config)
    return {"messages": [response], "remaining_steps": remaining - 1}


async def run_tools_node(
    state: MCPSubGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    context = get_runtime_context(runtime)
    tools = cast(list | None, context.get("mcp_subgraph_tools"))
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
                    pending[tc_id] = {"tool_name": name, "args": tc.get("args", {})}
            continue
        if isinstance(msg, ToolMessage):
            tc_id = str(getattr(msg, "tool_call_id", "") or "")
            content = str(getattr(msg, "content", "") or "")
            name = str(getattr(msg, "name", "") or "")
            error_text = content if getattr(msg, "status", "") == "error" else None
            rec = pending.pop(tc_id, {"tool_name": name, "args": None})
            rec["result"] = content
            if error_text:
                rec["error"] = error_text
            invocations.append(rec)
            continue
    return invocations


def build_mcp_sub_graph() -> StateGraph:
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
    thread_id = get_thread_id(runtime)
    messages = state.get("messages", [])
    question = latest_user_message(messages)
    chat_history = chat_history_before_latest_user(messages)
    retrieval_tool = rag_runtime.build_oracle_retrieval_tool(
        collection_name=cast(str | None, context.get("collection_name")),
        filter_docs=rag_runtime.filter_retrieved_docs,
    )
    resolved_model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
    run_cfg = build_run_config(
        parent_config=config,
        thread_id=thread_id,
        mode="mixed",
        model_id=resolved_model_id,
        session_id=cast(str | None, context.get("session_id")),
        enable_tracing=cast(bool | None, context.get("enable_tracing")),
        mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
    )
    from src.rag_agent.infrastructure.mcp_adapter_runtime import load_adapter_tools
    mcp_tools = await load_adapter_tools(
        server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
        run_config=run_cfg,
    )
    agent_tools = [retrieval_tool, *mcp_tools] if retrieval_tool else list(mcp_tools)
    system_prompt_text = _build_system_prompt_tools(question, agent_tools)
    input_messages: list[BaseMessage] = []
    for item in chat_history or []:
        converted = _message_to_langchain(item)
        if converted is not None:
            input_messages.append(converted)
    input_messages.append(HumanMessage(content=question))

    runtime.context["mcp_subgraph_tools"] = agent_tools
    runtime.context["mcp_subgraph_model_id"] = resolved_model_id
    runtime.context["mcp_subgraph_question"] = question
    runtime.context["mcp_subgraph_run_cfg"] = run_cfg

    return {
        "messages": [SystemMessage(content=system_prompt_text), *input_messages],
        "progress": "Planning collection and tool search…",
    }


async def run_mixed_mcp_node(
    state: ChatGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    context = get_runtime_context(runtime)
    thread_id = get_thread_id(runtime)
    messages = state["messages"]
    try:
        question = latest_user_message(messages)
        chat_history = chat_history_before_latest_user(messages)
        retrieval_tool = rag_runtime.build_oracle_retrieval_tool(
            collection_name=cast(str | None, context.get("collection_name")),
            filter_docs=rag_runtime.filter_retrieved_docs,
        )
        resolved_model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
        run_cfg = build_run_config(
            parent_config=config,
            thread_id=thread_id,
            mode="mixed",
            model_id=resolved_model_id,
            session_id=cast(str | None, context.get("session_id")),
            enable_tracing=cast(bool | None, context.get("enable_tracing")),
            mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
        )

        mcp_turn = await run_mcp_agent_turn(
            question=question,
            chat_history=chat_history,
            resolved_model_id=resolved_model_id,
            run_config=run_cfg,
            mode="mixed",
            mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
            require_tool_call=require_tool_call_enabled(),
            repeated_workflow_enabled=repeated_workflow_controller_enabled(),
            workflow_checkpoint_path=workflow_checkpoint_path(),
            extra_tools=[retrieval_tool],
            require_mcp_tool_call_when_referenced=True,
        )
        final_answer = mcp_turn.answer
        state_messages = cast(list[object], getattr(mcp_turn, "state_messages", []) or [])
        agent_final_answer = _latest_agent_final_answer(state_messages)
        if agent_final_answer:
            final_answer = agent_final_answer
        tools_used = mcp_turn.tools_used
        tool_invocations = mcp_turn.tool_invocations
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
        retrieval_state = getattr(retrieval_tool, "_retrieval_state", None)
        retrieval_docs = (
            cast(list[object], retrieval_state.get("docs", []))
            if isinstance(retrieval_state, dict)
            else []
        )
        retrieval_error = oracle_retrieval_error(
            retrieval_state=retrieval_state,
            tools_used=tools_used,
            tool_invocations=cast(list[dict[str, object]], tool_invocations),
        )
        if retrieval_docs and question:
            retrieval_docs = rag_runtime.rerank_retrieved_docs(
                question,
                cast(list[Any], retrieval_docs),
                enable_reranker=cast(bool | None, context.get("enable_reranker")),
            )
        if not policy_error and retrieval_error:
            final_answer = ORACLE_RETRIEVAL_FAILED_ANSWER
            policy_error = ORACLE_RETRIEVAL_FAILED_ANSWER
        if not policy_error and oracle_retrieval_used_without_context(
            retrieval_state=retrieval_state,
            retrieval_docs=cast(list[Any], retrieval_docs),
            tools_used=tools_used,
            tool_invocations=cast(list[dict[str, object]], tool_invocations),
        ):
            final_answer = NO_ORACLE_CONTEXT_ANSWER
        if not policy_error and retrieval_docs and agent_final_answer is None:
            supplemental_context = mixed_tool_supplemental_context(
                cast(list[dict[str, object]], tool_invocations)
            )
            final_answer, _rag_usage, resolved_model_id = await rag_runtime.synthesize_rag_answer(
                question=question,
                docs=cast(list[Any], retrieval_docs),
                model_id=cast(str | None, context.get("model_id")),
                run_config=run_cfg,
                supplemental_context=supplemental_context,
            )
        result: dict[str, object] = {
            "final_answer": final_answer,
            "error": policy_error,
            "outcome": "error" if policy_error else "success",
            "standalone_question": question or None,
            "citations": rag_runtime.citations_from_docs(cast(list[Any], retrieval_docs)),
            "reranker_docs": rag_runtime.serialize_docs(cast(list[Any], retrieval_docs)),
            "context_usage": {"retrieved_docs_count": len(retrieval_docs)}
            if retrieval_docs
            else None,
            "mcp_used": bool(tools_used),
            "mcp_tools_used": tools_used,
            "mcp_tool_invocations": tool_invocations,
        }
        if isinstance(resolved_model_id, str) and resolved_model_id.strip():
            result["model_id"] = resolved_model_id.strip()
        return {
            "mixed_result": result,
            "mixed_state_messages": state_messages,
        }
    except Exception as exc:
        assistant_message = assistant_message_from_exception("mixed", exc)
        return {
            "mixed_result": {
                "final_answer": assistant_message.content,
                "error": str(exc),
            },
            "mixed_state_messages": [],
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
    question = cast(str | None, context.get("mcp_subgraph_question")) or latest_user_message(messages) or ""
    tool_invocations = extract_tool_invocations_from_messages(messages)
    tools_used = list({inv["tool_name"] for inv in tool_invocations})
    final_answer = _latest_agent_final_answer(messages) or ""

    retrieval_state = None
    raw_tools = context.get("mcp_subgraph_tools")
    if isinstance(raw_tools, list):
        for tool in raw_tools:
            if hasattr(tool, "_retrieval_state"):
                retrieval_state = getattr(tool, "_retrieval_state", None)
                break
    retrieval_docs = (
        cast(list[object], retrieval_state.get("docs", []))
        if isinstance(retrieval_state, dict)
        else []
    )

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
        retrieval_state=retrieval_state,
        tools_used=tools_used,
        tool_invocations=cast(list[dict[str, object]], tool_invocations),
    )
    if not policy_error and retrieval_error:
        final_answer = ORACLE_RETRIEVAL_FAILED_ANSWER
        policy_error = ORACLE_RETRIEVAL_FAILED_ANSWER
    if not policy_error and oracle_retrieval_used_without_context(
        retrieval_state=retrieval_state,
        retrieval_docs=cast(list[Any], retrieval_docs),
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
            run_config=cast(RunnableConfig | None, context.get("mcp_subgraph_run_cfg")),
            supplemental_context=supplemental_context,
        )

    result: dict[str, object] = {
        "final_answer": final_answer,
        "error": policy_error,
        "outcome": "error" if policy_error else "success",
        "standalone_question": question or None,
        "citations": rag_runtime.citations_from_docs(cast(list[Any], retrieval_docs)),
        "reranker_docs": rag_runtime.serialize_docs(cast(list[Any], retrieval_docs)),
        "context_usage": {"retrieved_docs_count": len(retrieval_docs)}
        if retrieval_docs
        else None,
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
