from __future__ import annotations

import asyncio
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from src.rag_agent.workflows import WorkOutcome, WorkUnit, create_repeated_workflow_graph


def test_repeated_workflow_graph_processes_each_work_unit_until_terminal() -> None:
    calls: list[str] = []

    def process(unit: WorkUnit, _state: object) -> WorkOutcome:
        calls.append(unit["id"])
        if unit["payload"].get("skip"):
            return {"unit_id": unit["id"], "status": "skipped", "reason": "not applicable"}
        if unit["payload"].get("fail"):
            return {"unit_id": unit["id"], "status": "failed", "reason": "tool failed"}
        return {"unit_id": unit["id"], "status": "completed", "result": {"ok": True}}

    graph = create_repeated_workflow_graph(process_work_unit=process)

    result = graph.invoke(
        {
            "workflow_id": "wf-1",
            "work_units": [
                {"id": "unit-1", "payload": {}},
                {"id": "unit-2", "payload": {"skip": True}},
                {"id": "unit-3", "payload": {"fail": True}},
            ],
        }
    )

    assert calls == ["unit-1", "unit-2", "unit-3"]
    assert result["current_index"] == 3
    assert [item["unit_id"] for item in result["completed"]] == ["unit-1"]
    assert [item["unit_id"] for item in result["skipped"]] == ["unit-2"]
    assert [item["unit_id"] for item in result["failed"]] == ["unit-3"]
    assert result["finalized"] is True


def test_repeated_workflow_graph_supports_async_work_unit_processing() -> None:
    calls: list[str] = []

    async def process(unit: WorkUnit, _state: object) -> WorkOutcome:
        calls.append(unit["id"])
        return {"unit_id": unit["id"], "status": "completed"}

    graph = create_repeated_workflow_graph(process_work_unit=process)

    result = asyncio.run(
        graph.ainvoke(
            {
                "workflow_id": "wf-async",
                "work_units": [
                    {"id": "unit-1", "payload": {}},
                    {"id": "unit-2", "payload": {}},
                ],
            }
        )
    )

    assert calls == ["unit-1", "unit-2"]
    assert result["current_index"] == 2
    assert [item["unit_id"] for item in result["completed"]] == ["unit-1", "unit-2"]


def test_repeated_workflow_graph_persists_progress_with_langgraph_checkpointer(tmp_path) -> None:
    db_path = tmp_path / "workflow.sqlite"
    calls: list[str] = []

    def process(unit: WorkUnit, _state: object) -> WorkOutcome:
        calls.append(unit["id"])
        return {"unit_id": unit["id"], "status": "completed"}

    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        graph = create_repeated_workflow_graph(
            process_work_unit=process,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "workflow-thread"}}

        graph.invoke(
            {
                "workflow_id": "wf-2",
                "work_units": [
                    {"id": "unit-1", "payload": {}},
                    {"id": "unit-2", "payload": {}},
                ],
            },
            config,
        )

        snapshot = graph.get_state(config)
        assert snapshot.values["finalized"] is True
        assert snapshot.values["current_index"] == 2
        assert [item["unit_id"] for item in snapshot.values["completed"]] == [
            "unit-1",
            "unit-2",
        ]
    finally:
        conn.close()
