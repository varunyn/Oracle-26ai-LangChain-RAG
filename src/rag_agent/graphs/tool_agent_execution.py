from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias, TypedDict, TypeGuard

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from src.rag_agent.graphs.mcp_policies import is_trivial_answer
from src.rag_agent.graphs.state import ChatGraphContext, MCPSubGraphState
from src.rag_agent.graphs.tool_agent_turn import (
    reconstruct_tool_agent_turn,
    release_tool_agent_turn,
    release_tool_agent_turn_after_failure,
    run_with_lease_heartbeat,
)
from src.rag_agent.infrastructure.oci_models import get_llm


class ToolInvocation(TypedDict, total=False):
    invocation_id: str
    tool_name: str
    args: object
    result: str
    error: str


class ToolExecutionTranscript(TypedDict):
    final_answer: str
    has_terminal_answer: bool
    tool_invocations: list[ToolInvocation]
    tools_used: list[str]


INCOMPLETE_TOOL_CALL_ERROR = "Tool execution did not return a result."
ToolLike: TypeAlias = BaseTool | Callable[..., object]


def _is_any_message(message: object) -> TypeGuard[AnyMessage]:
    return isinstance(message, BaseMessage)


def _tool_sequence(tools: Sequence[object]) -> list[ToolLike]:
    typed_tools: list[ToolLike] = []
    for tool in tools:
        if isinstance(tool, BaseTool) or callable(tool):
            typed_tools.append(tool)
        else:
            raise TypeError(f"Unsupported tool type: {type(tool).__name__}")
    return typed_tools


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
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        content = _content_text(message.content)
        if content and not is_trivial_answer(content):
            return content
    return None


def message_to_langchain(message: object) -> BaseMessage | None:
    if message is None:
        return None
    if isinstance(message, Mapping):
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "")
        if not content:
            return None
        if role in {"assistant", "ai"}:
            return AIMessage(content=content)
        return HumanMessage(content=content)
    msg_type = str(getattr(message, "type", "") or getattr(message, "role", "")).strip().lower()
    content = str(getattr(message, "content", "") or "")
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
        content_type = item.get("type")
        if content_type == "tool_call":
            continue
        if content_type in _OCI_SUPPORTED_CONTENT_TYPES:
            content.append(dict(item))
            continue
        if "text" in item and content_type is None:
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
    turn = await reconstruct_tool_agent_turn(
        state={"messages": state.get("messages", [])},
        parent_config=config,
        runtime=runtime,
        mode="mcp",
    )
    try:
        tools = turn["tools"]
        model = get_llm(model_id=turn["model_id"])
        if tools:
            model = model.bind_tools(_tool_sequence(tools))
        remaining = state.get("remaining_steps", turn["tool_round_limit"])
        if remaining <= 0:
            response = AIMessage(content="Tool call limit reached.")
        else:
            full_messages: list[object] = [SystemMessage(content=turn["system_prompt"])]
            for item in turn["chat_history"]:
                converted = message_to_langchain(item)
                if converted is not None:
                    full_messages.append(converted)
            full_messages.append(HumanMessage(content=turn["question"]))
            full_messages.extend(state.get("messages", []))
            response = await run_with_lease_heartbeat(
                config,
                turn,
                lambda: model.ainvoke(
                    [_sanitize_for_oci(message) for message in full_messages], config=config
                ),
                runtime=runtime,
            )
    except BaseException:
        await release_tool_agent_turn_after_failure(config, turn)
        raise
    else:
        await release_tool_agent_turn(config, turn)
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
    turn = await reconstruct_tool_agent_turn(
        state={"messages": state.get("messages", [])},
        parent_config=config,
        runtime=runtime,
        mode="mcp",
    )
    try:
        tools = turn["tools"]
        if not tools:
            messages: list[AnyMessage] = []
        else:
            result = await run_with_lease_heartbeat(
                config,
                turn,
                lambda: ToolNode(_tool_sequence(tools)).ainvoke(
                    {"messages": [state["messages"][-1]]}, config=config
                ),
                runtime=runtime,
            )
            messages = [
                message for message in result.get("messages", []) if _is_any_message(message)
            ]
    except BaseException:
        await release_tool_agent_turn_after_failure(config, turn)
        raise
    else:
        await release_tool_agent_turn(config, turn)
    return {"messages": messages}


def route(state: MCPSubGraphState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "run_tools"
    return "__end__"


def messages_since_latest_user(messages: list[object]) -> list[object]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            return messages[index + 1 :]
    return messages


def analyze_tool_execution(messages: list[object]) -> ToolExecutionTranscript:
    pending: dict[str, ToolInvocation] = {}
    tool_invocations: list[ToolInvocation] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                invocation_id = str(tool_call.get("id", "") or "")
                tool_name = str(tool_call.get("name", "") or "")
                if tool_name:
                    invocation = ToolInvocation(
                        invocation_id=invocation_id,
                        tool_name=tool_name,
                        args=tool_call.get("args", {}),
                    )
                    pending[invocation_id] = invocation
                    tool_invocations.append(invocation)
            continue
        if isinstance(message, ToolMessage):
            invocation_id = str(getattr(message, "tool_call_id", "") or "")
            content = str(getattr(message, "content", "") or "")
            tool_name = str(getattr(message, "name", "") or "")
            if invocation_id in pending:
                invocation = pending.pop(invocation_id)
            else:
                invocation = ToolInvocation(
                    invocation_id=invocation_id,
                    tool_name=tool_name,
                    args=None,
                )
                tool_invocations.append(invocation)
            invocation["result"] = content
            if getattr(message, "status", "") == "error":
                invocation["error"] = content

    for invocation in pending.values():
        invocation["error"] = INCOMPLETE_TOOL_CALL_ERROR

    final_answer = _latest_agent_final_answer(messages)
    tools_used = list(dict.fromkeys(item["tool_name"] for item in tool_invocations))
    return {
        "final_answer": final_answer or "",
        "has_terminal_answer": final_answer is not None,
        "tool_invocations": tool_invocations,
        "tools_used": tools_used,
    }


async def _call_llm_graph_node(
    state: MCPSubGraphState,
    *,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    return await call_llm_node(state, get_config(), runtime=runtime)


async def _run_tools_graph_node(
    state: MCPSubGraphState,
    *,
    runtime: Runtime[ChatGraphContext],
) -> MCPSubGraphState:
    return await run_tools_node(state, get_config(), runtime=runtime)


def build_tool_agent_sub_graph(*, checkpointer: Checkpointer = None):
    sub_graph = StateGraph(MCPSubGraphState, context_schema=ChatGraphContext)
    sub_graph.add_node("call_llm", _call_llm_graph_node)
    sub_graph.add_node("run_tools", _run_tools_graph_node)
    sub_graph.add_conditional_edges("call_llm", route, {"run_tools": "run_tools", "__end__": END})
    sub_graph.add_edge("run_tools", "call_llm")
    sub_graph.set_entry_point("call_llm")
    return sub_graph.compile(checkpointer=checkpointer)


__all__ = [
    "ToolExecutionTranscript",
    "ToolInvocation",
    "INCOMPLETE_TOOL_CALL_ERROR",
    "analyze_tool_execution",
    "build_tool_agent_sub_graph",
    "call_llm_node",
    "messages_since_latest_user",
    "route",
    "run_tools_node",
]
