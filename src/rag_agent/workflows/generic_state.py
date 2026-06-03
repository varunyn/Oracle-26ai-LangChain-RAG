from __future__ import annotations

from typing import Any, Literal, TypedDict

TerminalStatus = Literal["completed", "skipped", "failed"]


class WorkUnit(TypedDict):
    """One independently addressable unit in a repeated workflow."""

    id: str
    payload: dict[str, Any]


class WorkOutcome(TypedDict, total=False):
    """Terminal result for one work unit."""

    unit_id: str
    status: TerminalStatus
    reason: str
    result: dict[str, Any]


class RepeatedWorkflowState(TypedDict, total=False):
    """Generic loop state for workflows that process many work units."""

    workflow_id: str
    work_units: list[WorkUnit]
    current_index: int
    completed: list[WorkOutcome]
    skipped: list[WorkOutcome]
    failed: list[WorkOutcome]
    finalized: bool


def terminal_count(state: RepeatedWorkflowState) -> int:
    return (
        len(state.get("completed", []))
        + len(state.get("skipped", []))
        + len(state.get("failed", []))
    )


def is_complete(state: RepeatedWorkflowState) -> bool:
    return terminal_count(state) >= len(state.get("work_units", []))
