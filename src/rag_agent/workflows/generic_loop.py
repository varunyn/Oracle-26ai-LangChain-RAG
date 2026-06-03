from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .generic_state import RepeatedWorkflowState, WorkOutcome, WorkUnit, is_complete

ProcessWorkUnit = Callable[[WorkUnit, RepeatedWorkflowState], WorkOutcome | Awaitable[WorkOutcome]]


def create_repeated_workflow_graph(
    *,
    process_work_unit: ProcessWorkUnit,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> object:
    """Build a generic LangGraph loop for repeated work-unit processing."""

    def initialize(state: RepeatedWorkflowState) -> RepeatedWorkflowState:
        return {
            **state,
            "current_index": int(state.get("current_index", 0) or 0),
            "completed": list(state.get("completed", [])),
            "skipped": list(state.get("skipped", [])),
            "failed": list(state.get("failed", [])),
            "finalized": bool(state.get("finalized", False)),
        }

    def process_current(state: RepeatedWorkflowState) -> RepeatedWorkflowState:
        work_units = state.get("work_units", [])
        current_index = int(state.get("current_index", 0) or 0)
        if current_index >= len(work_units):
            return state

        outcome = process_work_unit(work_units[current_index], state)
        if inspect.isawaitable(outcome):
            raise TypeError("Async work-unit processors require graph.ainvoke().")
        return _apply_outcome(state, cast(WorkOutcome, outcome))

    async def aprocess_current(state: RepeatedWorkflowState) -> RepeatedWorkflowState:
        work_units = state.get("work_units", [])
        current_index = int(state.get("current_index", 0) or 0)
        if current_index >= len(work_units):
            return state

        outcome = process_work_unit(work_units[current_index], state)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return _apply_outcome(state, cast(WorkOutcome, outcome))

    def _apply_outcome(
        state: RepeatedWorkflowState,
        outcome: WorkOutcome,
    ) -> RepeatedWorkflowState:
        work_units = state.get("work_units", [])
        current_index = int(state.get("current_index", 0) or 0)
        if current_index >= len(work_units):
            return state
        status = outcome.get("status")
        if status not in {"completed", "skipped", "failed"}:
            outcome = {
                "unit_id": work_units[current_index]["id"],
                "status": "failed",
                "reason": f"Invalid terminal status: {status}",
            }
            status = "failed"

        updated: RepeatedWorkflowState = {
            **state,
            "current_index": current_index + 1,
            "completed": list(state.get("completed", [])),
            "skipped": list(state.get("skipped", [])),
            "failed": list(state.get("failed", [])),
        }
        updated[status].append(outcome)
        return updated

    def finalize(state: RepeatedWorkflowState) -> RepeatedWorkflowState:
        return {**state, "finalized": is_complete(state)}

    def route_after_initialize(
        state: RepeatedWorkflowState,
    ) -> Literal["process_current", "finalize"]:
        return "finalize" if is_complete(state) else "process_current"

    def route_after_process(state: RepeatedWorkflowState) -> Literal["process_current", "finalize"]:
        return "finalize" if is_complete(state) else "process_current"

    builder = StateGraph(RepeatedWorkflowState)
    builder.add_node("initialize", initialize)
    builder.add_node(
        "process_current",
        RunnableLambda(process_current, afunc=aprocess_current),
    )
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        route_after_initialize,
        {"process_current": "process_current", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "process_current",
        route_after_process,
        {"process_current": "process_current", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
