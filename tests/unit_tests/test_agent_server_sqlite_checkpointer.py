import asyncio
import json
from pathlib import Path

import src.rag_agent.runtime.agent_server_checkpointer as checkpointer_module
from src.rag_agent.runtime.agent_server_checkpointer import (
    LocalAsyncSqliteSaver,
    generate_checkpointer,
    resolve_sqlite_checkpointer_path,
)


def test_langgraph_json_uses_agent_server_sqlite_checkpointer() -> None:
    config = json.loads(Path("langgraph.json").read_text())

    assert config["checkpointer"] == {
        "backend": "custom",
        "path": "./src/rag_agent/runtime/agent_server_checkpointer.py:generate_checkpointer",
        "ttl": {
            "strategy": "delete",
            "sweep_interval_minutes": 60,
            "default_ttl": 10080,
        },
    }


def test_resolve_sqlite_checkpointer_path_creates_parent_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nested" / "langgraph.sqlite"
    monkeypatch.setenv("LANGGRAPH_SQLITE_PATH", str(db_path))

    assert resolve_sqlite_checkpointer_path() == db_path
    assert db_path.parent.is_dir()


def test_generate_checkpointer_yields_async_sqlite_saver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "langgraph.sqlite"
    monkeypatch.setenv("LANGGRAPH_SQLITE_PATH", str(db_path))

    async def _run() -> None:
        async with generate_checkpointer() as checkpointer:
            assert isinstance(checkpointer, LocalAsyncSqliteSaver)

    asyncio.run(_run())

    assert db_path.exists()


def test_generate_checkpointer_cancels_and_awaits_periodic_reconciliation_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "langgraph.sqlite"
    monkeypatch.setenv("LANGGRAPH_SQLITE_PATH", str(db_path))
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def block_until_cancelled(_saver: LocalAsyncSqliteSaver) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(
        checkpointer_module,
        "_reconcile_recipes_periodically",
        block_until_cancelled,
    )

    async def _run() -> None:
        async with generate_checkpointer():
            await asyncio.wait_for(started.wait(), timeout=0.1)
        assert cancelled.is_set()

    asyncio.run(_run())


def test_local_sqlite_saver_prunes_old_checkpoints(tmp_path: Path) -> None:
    async def _run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "x.sqlite")) as saver:
            await saver.setup()
            await _insert_checkpoint(saver, "thread-a", "", "001", run_id="run-1")
            await _insert_checkpoint(saver, "thread-a", "", "002", run_id="run-2")
            await _insert_checkpoint(saver, "thread-a", "tools", "001", run_id="run-3")
            await _insert_checkpoint(saver, "thread-a", "tools", "003", run_id="run-4")

            await saver.aprune(["thread-a"], strategy="keep_latest")

            assert await _checkpoint_ids(saver, "thread-a") == [("empty", "002"), ("tools", "003")]

    asyncio.run(_run())


def test_local_sqlite_saver_copies_thread(tmp_path: Path) -> None:
    async def _run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "x.sqlite")) as saver:
            await saver.setup()
            await _insert_checkpoint(saver, "source", "", "001", run_id="run-1")
            await _insert_write(saver, "source", "", "001", "task-a")

            await saver.acopy_thread("source", "target")

            assert await _checkpoint_ids(saver, "target") == [("empty", "001")]
            assert await _write_count(saver, "target") == 1

    asyncio.run(_run())


def test_local_sqlite_saver_deletes_checkpoints_for_runs(tmp_path: Path) -> None:
    async def _run() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "x.sqlite")) as saver:
            await saver.setup()
            await _insert_checkpoint(saver, "thread-a", "", "001", run_id="run-delete")
            await _insert_write(saver, "thread-a", "", "001", "task-a")
            await _insert_checkpoint(saver, "thread-a", "", "002", run_id="run-keep")

            await saver.adelete_for_runs(["run-delete"])

            assert await _checkpoint_ids(saver, "thread-a") == [("empty", "002")]
            assert await _write_count(saver, "thread-a") == 0

    asyncio.run(_run())


async def _insert_checkpoint(
    saver: LocalAsyncSqliteSaver,
    thread_id: str,
    namespace: str,
    checkpoint_id: str,
    *,
    run_id: str,
) -> None:
    await saver.conn.execute(
        """
        INSERT INTO checkpoints (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            type,
            checkpoint,
            metadata
        ) VALUES (?, ?, ?, NULL, 'json', X'7B7D', ?)
        """,
        (
            thread_id,
            namespace,
            checkpoint_id,
            json.dumps({"run_id": run_id}).encode("utf-8"),
        ),
    )
    await saver.conn.commit()


async def _insert_write(
    saver: LocalAsyncSqliteSaver,
    thread_id: str,
    namespace: str,
    checkpoint_id: str,
    task_id: str,
) -> None:
    await saver.conn.execute(
        """
        INSERT INTO writes (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            idx,
            channel,
            type,
            value
        ) VALUES (?, ?, ?, ?, 0, 'messages', 'json', X'7B7D')
        """,
        (thread_id, namespace, checkpoint_id, task_id),
    )
    await saver.conn.commit()


async def _checkpoint_ids(
    saver: LocalAsyncSqliteSaver,
    thread_id: str,
) -> list[tuple[str, str]]:
    async with saver.conn.execute(
        """
        SELECT checkpoint_ns, checkpoint_id
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_ns, checkpoint_id
        """,
        (thread_id,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [(str(namespace or "empty"), str(checkpoint_id)) for namespace, checkpoint_id in rows]


async def _write_count(saver: LocalAsyncSqliteSaver, thread_id: str) -> int:
    async with saver.conn.execute(
        "SELECT COUNT(*) FROM writes WHERE thread_id = ?",
        (thread_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return int(row[0])
