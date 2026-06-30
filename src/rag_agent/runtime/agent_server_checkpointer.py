from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DEFAULT_SQLITE_CHECKPOINT_PATH = ".local-data/langgraph-checkpoints.sqlite"
RUN_ID_METADATA_KEYS = ("run_id", "checkpoint_id", "langgraph_run_id")


class LocalAsyncSqliteSaver(AsyncSqliteSaver):
    """SQLite checkpointer with Agent Server optional operations implemented."""

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        await self.setup()
        normalized_thread_ids = [str(thread_id) for thread_id in thread_ids]
        if not normalized_thread_ids:
            return
        if strategy == "delete":
            for thread_id in normalized_thread_ids:
                await self.adelete_thread(thread_id)
            return
        if strategy != "keep_latest":
            raise ValueError(f"Unsupported SQLite checkpoint prune strategy: {strategy}")

        async with self.lock:
            for thread_id in normalized_thread_ids:
                async with self.conn.execute(
                    """
                    SELECT checkpoint_ns, checkpoint_id
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY checkpoint_ns, checkpoint_id DESC
                    """,
                    (thread_id,),
                ) as cursor:
                    rows = await cursor.fetchall()
                latest_by_namespace: dict[str, str] = {}
                for checkpoint_ns, checkpoint_id in rows:
                    latest_by_namespace.setdefault(str(checkpoint_ns), str(checkpoint_id))
                stale_rows = [
                    (thread_id, str(checkpoint_ns), str(checkpoint_id))
                    for checkpoint_ns, checkpoint_id in rows
                    if latest_by_namespace[str(checkpoint_ns)] != str(checkpoint_id)
                ]
                await self._delete_checkpoint_rows(stale_rows)
            await self.conn.commit()

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await self.setup()
        source_thread_id = str(source_thread_id)
        target_thread_id = str(target_thread_id)
        if source_thread_id == target_thread_id:
            return
        async with self.lock:
            await self._delete_thread_rows(target_thread_id)
            await self.conn.execute(
                """
                INSERT INTO checkpoints (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata
                )
                SELECT
                    ?,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata
                FROM checkpoints
                WHERE thread_id = ?
                """,
                (target_thread_id, source_thread_id),
            )
            await self.conn.execute(
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
                )
                SELECT
                    ?,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    idx,
                    channel,
                    type,
                    value
                FROM writes
                WHERE thread_id = ?
                """,
                (target_thread_id, source_thread_id),
            )
            await self.conn.commit()

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await self.setup()
        normalized_run_ids = {str(run_id) for run_id in run_ids if str(run_id)}
        if not normalized_run_ids:
            return
        async with self.lock:
            async with self.conn.execute(
                """
                SELECT thread_id, checkpoint_ns, checkpoint_id, metadata
                FROM checkpoints
                """
            ) as cursor:
                rows = await cursor.fetchall()
            matching_rows = [
                (str(thread_id), str(checkpoint_ns), str(checkpoint_id))
                for thread_id, checkpoint_ns, checkpoint_id, metadata in rows
                if _metadata_matches_run_id(metadata, normalized_run_ids)
            ]
            await self._delete_checkpoint_rows(matching_rows)
            await self.conn.commit()

    async def _delete_thread_rows(self, thread_id: str) -> None:
        await self.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        await self.conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))

    async def _delete_checkpoint_rows(
        self,
        rows: Sequence[tuple[str, str, str]],
    ) -> None:
        for thread_id, checkpoint_ns, checkpoint_id in rows:
            params = (thread_id, checkpoint_ns, checkpoint_id)
            await self.conn.execute(
                """
                DELETE FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """,
                params,
            )
            await self.conn.execute(
                """
                DELETE FROM writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """,
                params,
            )


def _metadata_matches_run_id(metadata: bytes | str | None, run_ids: set[str]) -> bool:
    if metadata is None:
        return False
    if isinstance(metadata, bytes):
        metadata_text = metadata.decode("utf-8", "ignore")
    else:
        metadata_text = metadata
    try:
        parsed = json.loads(metadata_text)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    return any(str(parsed.get(key) or "") in run_ids for key in RUN_ID_METADATA_KEYS)


def resolve_sqlite_checkpointer_path() -> Path:
    raw_path = os.environ.get("LANGGRAPH_SQLITE_PATH", DEFAULT_SQLITE_CHECKPOINT_PATH).strip()
    path = Path(raw_path or DEFAULT_SQLITE_CHECKPOINT_PATH).expanduser()
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


@asynccontextmanager
async def generate_checkpointer() -> AsyncIterator[LocalAsyncSqliteSaver]:
    """Yield the local SQLite checkpointer used by LangGraph Agent Server."""

    checkpoint_path = resolve_sqlite_checkpointer_path()
    async with LocalAsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        await saver.setup()
        yield saver


__all__ = [
    "LocalAsyncSqliteSaver",
    "generate_checkpointer",
    "resolve_sqlite_checkpointer_path",
]
