from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

from src.rag_agent.graphs.mcp_policies import is_trivial_answer
from src.rag_agent.graphs.nodes.mixed import (
    _build_system_prompt_tools,
    _latest_agent_final_answer,
    extract_tool_invocations_from_messages,
    message_to_langchain,
)
from src.rag_agent.graphs.nodes.references import messages_from_result
from src.rag_agent.graphs.runtime import build_run_config, get_runtime_context, get_thread_id
from src.rag_agent.graphs.state import ChatGraphContext, ChatGraphState
from src.rag_agent.infrastructure import oci_models as _oci_models
from src.rag_agent.infrastructure.mcp_adapter_runtime import load_adapter_tools
from src.rag_agent.runtime.mcp_turn import tool_failure_summary
from src.rag_agent.runtime.memory import (
    chat_history_before_latest_user,
    latest_user_message,
)


def get_llm(model_id: str | None = None) -> Any:
    return _oci_models.get_llm(model_id=model_id)


async def run_mcp_setup(
    state: ChatGraphState,
    config: RunnableConfig,
    runtime: Runtime[ChatGraphContext],
) -> ChatGraphState:
    context = get_runtime_context(runtime)
    thread_id = get_thread_id(runtime)
    messages = state.get("messages", [])
    question = latest_user_message(messages)
    chat_history = chat_history_before_latest_user(messages)
    resolved_model_id = cast(str | None, context.get("model_id")) or get_llm().model_id
    run_cfg = build_run_config(
        parent_config=config,
        thread_id=thread_id,
        mode="mcp",
        model_id=resolved_model_id,
        session_id=cast(str | None, context.get("session_id")),
        enable_tracing=cast(bool | None, context.get("enable_tracing")),
        mcp_server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
    )
    mcp_tools = await load_adapter_tools(
        server_keys=cast(list[str] | None, context.get("mcp_server_keys")),
        run_config=run_cfg,
    )
    system_prompt_text = _build_system_prompt_tools(question, mcp_tools)
    input_messages: list[BaseMessage] = []
    for item in chat_history or []:
        converted = message_to_langchain(item)
        if converted is not None:
            input_messages.append(converted)
    input_messages.append(HumanMessage(content=question))

    runtime.context["mcp_subgraph_tools"] = mcp_tools
    runtime.context["mcp_subgraph_model_id"] = resolved_model_id
    runtime.context["mcp_subgraph_question"] = question
    runtime.context["mcp_subgraph_run_cfg"] = run_cfg
    runtime.context["mcp_subgraph_input_count"] = 1 + len(input_messages)

    return {
        "messages": [SystemMessage(content=system_prompt_text), *input_messages],
    }


async def run_mcp_compose(
    state: ChatGraphState,
    _config: RunnableConfig | None = None,
    _runtime: Runtime[ChatGraphContext] | None = None,
) -> ChatGraphState:
    messages = state.get("messages", [])
    if not messages:
        result = {"final_answer": "MCP execution did not produce a result."}
        return {"messages": messages_from_result("mcp", result, []), "references": {}}

    context = get_runtime_context(_runtime) if _runtime else {}
    input_count = cast(int, context.get("mcp_subgraph_input_count", 0))
    tool_messages = messages[input_count:] if input_count > 0 and len(messages) > input_count else messages

    context = get_runtime_context(_runtime) if _runtime else {}
    tool_invocations = extract_tool_invocations_from_messages(tool_messages)
    tools_used = list({inv["tool_name"] for inv in tool_invocations})
    final_answer = _latest_agent_final_answer(tool_messages) or ""

    tool_failure_error = tool_failure_summary(cast(list[dict[str, object]], tool_invocations))
    if is_trivial_answer(final_answer) and tool_failure_error:
        final_answer = tool_failure_error

    result: dict[str, object] = {
        "final_answer": final_answer,
        "error": None,
        "standalone_question": cast(str | None, context.get("mcp_subgraph_question")),
        "citations": [],
        "reranker_docs": [],
        "context_usage": None,
        "mcp_used": bool(tools_used),
        "mcp_tools_used": tools_used,
        "mcp_tool_invocations": tool_invocations,
    }
    messages_out = messages_from_result("mcp", result, tool_messages)
    references = cast(dict[str, object], getattr(messages_out[-1], "additional_kwargs", {}) or {})
    return {
        "messages": messages_out,
        "references": references,
    }



