from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from .generic_loop import create_repeated_workflow_graph
from .generic_state import RepeatedWorkflowState, WorkOutcome, WorkUnit
from .work_unit_extraction import extract_work_units_from_tool_invocations
from .workflow_intent import (
    select_repeated_workflow_context_tools,
    select_repeated_workflow_discovery_tools,
    select_repeated_workflow_finalization_tools,
    select_repeated_workflow_processing_tools,
)

AnswerCallable = Callable[..., Awaitable[tuple[str, list[str], list[dict[str, object]]]]]


async def run_repeated_mcp_workflow(
    *,
    question: str,
    model_id: str | None,
    tools: Sequence[BaseTool],
    run_config: RunnableConfig | None,
    require_tool_call: bool,
    get_answer: AnswerCallable,
    checkpoint_path: str | Path | None = None,
    discovery_result: tuple[str, list[str], list[dict[str, object]]] | None = None,
    chat_history: Sequence[object] | None = None,
) -> tuple[str, list[str], list[dict[str, object]]] | None:
    """Run a generic discover/process/finalize MCP workflow when a queue is found."""

    if discovery_result is None:
        discovery_tools = await select_repeated_workflow_discovery_tools(
            question=question,
            tools=tools,
            model_id=model_id,
            run_config=run_config,
        )
        processing_tools = await select_repeated_workflow_processing_tools(
            question=question,
            tools=tools,
            model_id=model_id,
            run_config=run_config,
        )
        context_tools = await select_repeated_workflow_context_tools(
            question=question,
            tools=tools,
            model_id=model_id,
            run_config=run_config,
        )
        processing_tools = _with_cached_context_tools(processing_tools, context_tools)
        finalization_tools = await select_repeated_workflow_finalization_tools(
            question=question,
            tools=tools,
            model_id=model_id,
            run_config=run_config,
        )
        discovery_result = await get_answer(
            question=_discovery_prompt(question),
            chat_history=chat_history,
            model_id=model_id,
            tools=list(discovery_tools),
            run_config=run_config,
            require_tool_call=require_tool_call,
        )
    else:
        processing_tools = list(tools)
        finalization_tools = list(tools)
    discovery_answer, discovery_tool_names, discovery_invocations = discovery_result
    work_units = extract_work_units_from_tool_invocations(discovery_invocations)
    if not work_units:
        return (
            _no_work_queue_answer(discovery_answer),
            _dedupe(discovery_tool_names),
            list(discovery_invocations),
        )

    per_unit_invocations: list[dict[str, object]] = []
    per_unit_tools: list[str] = []
    workflow_thread_id = _new_workflow_thread_id(run_config)

    async def _process(unit: WorkUnit, _state: RepeatedWorkflowState) -> WorkOutcome:
        answer, used, invocations = await get_answer(
            question=_per_unit_prompt(question, unit),
            chat_history=chat_history,
            model_id=model_id,
            tools=list(processing_tools),
            run_config=run_config,
            require_tool_call=require_tool_call,
        )
        per_unit_tools.extend(used)
        per_unit_invocations.extend(invocations)
        failed = _contains_tool_failure(invocations)
        if not invocations and processing_tools:
            failed = True
        return {
            "unit_id": unit["id"],
            "status": "failed" if failed else "completed",
            "reason": (
                "No processing tool was called for this work unit."
                if not invocations and processing_tools
                else answer[:500]
            ),
            "result": {"answer": answer},
        }

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
            await checkpointer.setup()
            graph = create_repeated_workflow_graph(
                process_work_unit=_process,
                checkpointer=checkpointer,
            )
            state = cast(
                RepeatedWorkflowState,
                await cast(Any, graph).ainvoke(
                    {"workflow_id": workflow_thread_id, "work_units": work_units},
                    _workflow_config(workflow_thread_id),
                ),
            )
    else:
        graph = create_repeated_workflow_graph(
            process_work_unit=_process,
        )
        state = cast(
            RepeatedWorkflowState,
            await cast(Any, graph).ainvoke(
                {"workflow_id": workflow_thread_id, "work_units": work_units},
                _workflow_config(workflow_thread_id),
            ),
        )

    final_answer, final_used, final_invocations = await get_answer(
        question=_finalization_prompt(question, state),
        chat_history=chat_history,
        model_id=model_id,
        tools=list(finalization_tools),
        run_config=run_config,
        require_tool_call=require_tool_call,
    )
    tools_used = _dedupe([*discovery_tool_names, *per_unit_tools, *final_used])
    invocations = [*discovery_invocations, *per_unit_invocations, *final_invocations]
    return final_answer or discovery_answer, tools_used, invocations


def _discovery_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "First establish the complete work queue using the available tools. "
        "Do not process individual work units yet. Do not use examples, samples, "
        "or reference inputs as the work queue. If queue discovery fails, retry "
        "with corrected arguments when possible. Use native structured tool calls only; "
        "never print tool_name(...) as text. After the queue is discovered, stop."
    )


def _no_work_queue_answer(discovery_answer: str) -> str:
    answer = discovery_answer.strip()
    if answer:
        return (
            "I could not identify a work queue from the discovery tool results, "
            f"so I stopped before processing individual work units. Discovery answer: {answer}"
        )
    return (
        "I could not identify a work queue from the discovery tool results, "
        "so I stopped before processing individual work units."
    )


def _per_unit_prompt(question: str, unit: WorkUnit) -> str:
    return (
        f"{question}\n\n"
        "Process exactly one work unit from the discovered queue. Do not call final "
        "summary, notification, report, or communication tools in this step. "
        "Do not perform any final completion action in this step. Use a native "
        "structured tool call for the appropriate per-unit action; never print "
        "tool_name(...) as text. If no per-unit action applies, reply only with "
        "SKIPPED and a brief reason.\n"
        f"Work unit:\n{json.dumps(unit['payload'], ensure_ascii=True, default=str)}"
    )


def _finalization_prompt(question: str, state: RepeatedWorkflowState) -> str:
    total = len(state.get("work_units", []))
    completed = len(state.get("completed", []))
    skipped = len(state.get("skipped", []))
    failed = len(state.get("failed", []))
    return (
        f"{question}\n\n"
        "All discovered work units now have terminal outcomes. Perform only the "
        "user-requested final summary, report, notification, or communication step. "
        "When the user requested a final action tool, use a native structured tool "
        "call; never print tool_name(...) as text.\n"
        f"Counts: total={total}, completed={completed}, skipped={skipped}, failed={failed}."
    )


def _contains_tool_failure(invocations: Sequence[dict[str, object]]) -> bool:
    for invocation in invocations:
        text = str(invocation.get("result") or "").lower()
        if "error" in text or "failed after" in text or "toolexception" in text:
            return True
    return False


def _with_cached_context_tools(
    processing_tools: Sequence[BaseTool],
    context_tools: Sequence[BaseTool],
) -> list[BaseTool]:
    context_names = {str(tool.name) for tool in context_tools}
    if not context_names:
        return list(processing_tools)

    cache: dict[str, object] = {}
    wrapped: list[BaseTool] = []
    for tool in processing_tools:
        if str(tool.name) not in context_names:
            wrapped.append(tool)
            continue
        wrapped.append(_cached_tool(tool, cache))
    return wrapped


def _cached_tool(tool: BaseTool, cache: dict[str, object]) -> BaseTool:
    async def _call(**kwargs: Any) -> Any:
        cache_key = json.dumps(
            {"tool": tool.name, "args": kwargs},
            sort_keys=True,
            default=str,
        )
        if cache_key not in cache:
            cache[cache_key] = await _invoke_tool_for_cache(tool, kwargs, cache_key)
        return cache[cache_key]

    args_schema = tool.args_schema or tool.get_input_schema()
    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=cast(type[BaseModel] | dict[str, Any], args_schema),
        coroutine=_call,
        response_format=getattr(tool, "response_format", "content"),
        metadata=getattr(tool, "metadata", None),
    )


async def _invoke_tool_for_cache(tool: BaseTool, kwargs: dict[str, Any], cache_key: str) -> Any:
    if getattr(tool, "response_format", "content") != "content_and_artifact":
        return await tool.ainvoke(kwargs)

    tool_call = {
        "type": "tool_call",
        "id": f"cached_{abs(hash(cache_key))}",
        "name": tool.name,
        "args": kwargs,
    }
    result = await tool.ainvoke(tool_call)
    if isinstance(result, ToolMessage):
        return (result.content, result.artifact)
    return result


def _workflow_config(workflow_thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": workflow_thread_id}}


def _new_workflow_thread_id(run_config: RunnableConfig | None) -> str:
    configurable = {}
    if isinstance(run_config, dict) and isinstance(run_config.get("configurable"), dict):
        configurable = cast(dict[str, object], run_config["configurable"])
    thread_id = str(configurable.get("thread_id") or "repeated-workflow")
    workflow_run_id = str(configurable.get("workflow_run_id") or uuid4())
    return f"{thread_id}:workflow:{workflow_run_id}"


def _dedupe(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
