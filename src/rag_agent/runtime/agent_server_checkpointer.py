from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.rag_agent.runtime.tool_agent_recipe_store import ToolAgentTurnRecipeStore

DEFAULT_SQLITE_CHECKPOINT_PATH = ".local-data/langgraph-checkpoints.sqlite"
RUN_ID_METADATA_KEYS = ("run_id", "checkpoint_id", "langgraph_run_id")
DEFAULT_RECIPE_RECONCILE_INTERVAL_SECONDS = 60 * 60

logger = logging.getLogger(__name__)


class LocalAsyncSqliteSaver(AsyncSqliteSaver):
    """SQLite checkpointer with Agent Server optional operations implemented."""

    def __init__(self, conn, *args, **kwargs) -> None:
        super().__init__(conn, *args, **kwargs)
        self._recipe_schema_ready = False
        self._recipe_store = ToolAgentTurnRecipeStore(self)

    @property
    def recipe_store(self) -> ToolAgentTurnRecipeStore:
        return self._recipe_store

    async def setup(self) -> None:
        await super().setup()
        if self._recipe_schema_ready:
            return
        async with self.lock:
            if self._recipe_schema_ready:
                return
            await self.conn.execute("PRAGMA foreign_keys = ON")
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS tool_agent_turn_recipes (
                    thread_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    origin_run_id TEXT,
                    recipe_json BLOB NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at_epoch_ms INTEGER NOT NULL,
                    lease_owner_id TEXT,
                    lease_fence INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at_epoch_ms INTEGER,
                    terminal_message_id TEXT,
                    terminal_marked_at_epoch_ms INTEGER,
                    PRIMARY KEY (thread_id, turn_id)
                );
                CREATE INDEX IF NOT EXISTS tool_agent_turn_recipes_origin_run_idx
                    ON tool_agent_turn_recipes(origin_run_id);
                CREATE TABLE IF NOT EXISTS tool_agent_turn_run_links (
                    thread_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    created_at_epoch_ms INTEGER NOT NULL,
                    PRIMARY KEY (thread_id, run_id, turn_id),
                    FOREIGN KEY (thread_id, turn_id)
                        REFERENCES tool_agent_turn_recipes(thread_id, turn_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS tool_agent_turn_run_links_run_idx
                    ON tool_agent_turn_run_links(run_id);
                CREATE TABLE IF NOT EXISTS tool_agent_turn_checkpoint_links (
                    thread_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    PRIMARY KEY (thread_id, turn_id, checkpoint_ns, checkpoint_id),
                    FOREIGN KEY (thread_id, turn_id)
                        REFERENCES tool_agent_turn_recipes(thread_id, turn_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (thread_id, checkpoint_ns, checkpoint_id)
                        REFERENCES checkpoints(thread_id, checkpoint_ns, checkpoint_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS tool_agent_turn_checkpoint_links_checkpoint_idx
                    ON tool_agent_turn_checkpoint_links(thread_id, checkpoint_ns, checkpoint_id);
                CREATE TABLE IF NOT EXISTS tool_agent_turn_recipe_copy_provenance (
                    target_thread_id TEXT NOT NULL,
                    target_turn_id TEXT NOT NULL,
                    source_thread_id TEXT NOT NULL,
                    source_turn_id TEXT NOT NULL,
                    copied_at_epoch_ms INTEGER NOT NULL,
                    PRIMARY KEY (target_thread_id, target_turn_id),
                    FOREIGN KEY (target_thread_id, target_turn_id)
                        REFERENCES tool_agent_turn_recipes(thread_id, turn_id)
                        ON DELETE CASCADE
                );
                """)
            await self.conn.commit()
            self._recipe_schema_ready = True

    @asynccontextmanager
    async def transaction_locked(self):
        """Run one saver-owned transaction and always recover the connection."""

        await self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            await self._rollback_safely()
            raise
        else:
            try:
                await self.conn.commit()
            except BaseException:
                await self._rollback_safely()
                raise

    async def _rollback_safely(self) -> None:
        try:
            await self.conn.rollback()
        except BaseException:
            pass

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

        async with self.lock, self.transaction_locked():
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
                await self.recipe_store._delete_unreachable_recipes_locked()

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await self.setup()
        source_thread_id = str(source_thread_id)
        target_thread_id = str(target_thread_id)
        if source_thread_id == target_thread_id:
            return
        async with self.lock, self.transaction_locked():
            await self._assert_copyable_thread_locked(source_thread_id)
            # Fence the target before its state is destroyed.  This shares the
            # same SQLite transaction as the replacement, so another saver
            # connection cannot slip a live recipe into either side.
            await self._assert_copyable_thread_locked(target_thread_id)
            await self._delete_thread_rows(target_thread_id)
            await self.recipe_store._delete_for_thread_locked(target_thread_id)
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
            await self._copy_recipe_rows_locked(source_thread_id, target_thread_id)
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

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await self.setup()
        normalized_run_ids = {str(run_id) for run_id in run_ids if str(run_id)}
        if not normalized_run_ids:
            return
        async with self.lock, self.transaction_locked():
            async with self.conn.execute("""
                SELECT thread_id, checkpoint_ns, checkpoint_id, metadata
                FROM checkpoints
                """) as cursor:
                rows = await cursor.fetchall()
            matching_rows = [
                (str(thread_id), str(checkpoint_ns), str(checkpoint_id))
                for thread_id, checkpoint_ns, checkpoint_id, metadata in rows
                if _metadata_matches_run_id(metadata, normalized_run_ids)
            ]
            await self._delete_checkpoint_rows(matching_rows)
            await self.recipe_store._delete_checkpoint_links_locked(matching_rows)
            await self.recipe_store._delete_for_origin_runs_locked(list(normalized_run_ids))

    async def adelete_thread(self, thread_id: str) -> None:
        await self.setup()
        async with self.lock, self.transaction_locked():
            await self._delete_thread_rows(thread_id)
            await self.recipe_store._delete_for_thread_locked(str(thread_id))

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

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Pinned sqlite-saver 3.1.0 checkpoint write with atomic recipe linking."""

        del new_versions  # The sqlite 3.1.0 base saver does not persist versions separately.
        await self.setup()
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable["checkpoint_ns"])
        checkpoint_id = str(checkpoint["id"])
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        ).encode("utf-8", "ignore")
        turn_id = _latest_human_message_id(checkpoint)
        async with self.lock, self.transaction_locked():
            await self.conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints(
                    thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    type, checkpoint, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    configurable.get("checkpoint_id"),
                    type_,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            )
            if turn_id is not None:
                await self.recipe_store._link_checkpoint_locked(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def _assert_copyable_thread_locked(self, thread_id: str) -> None:
        async with self.conn.execute(
            "SELECT CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        now = int(row[0])
        async with self.conn.execute(
            """
            SELECT 1 FROM tool_agent_turn_recipes
            WHERE thread_id = ? AND lease_expires_at_epoch_ms > ?
            LIMIT 1
            """,
            (thread_id, now),
        ) as cursor:
            if await cursor.fetchone() is not None:
                raise RuntimeError("Cannot copy a thread with an active tool-agent recipe lease.")

    async def _copy_recipe_rows_locked(self, source_thread_id: str, target_thread_id: str) -> None:
        """Copy recipes only when a copied checkpoint can still require them."""

        async with self.conn.execute(
            """
            SELECT DISTINCT recipe.recipe_json, recipe.schema_version, recipe.created_at_epoch_ms,
                   recipe.turn_id
            FROM tool_agent_turn_recipes AS recipe
            JOIN tool_agent_turn_checkpoint_links AS link
              ON link.thread_id = recipe.thread_id AND link.turn_id = recipe.turn_id
            WHERE recipe.thread_id = ?
            """,
            (source_thread_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        for recipe_json, schema_version, _created_at, turn_id in rows:
            payload = json.loads(
                recipe_json.decode() if isinstance(recipe_json, bytes) else recipe_json
            )
            payload["thread_id"] = target_thread_id
            payload["origin_run_id"] = None
            canonical = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            await self.conn.execute(
                """
                INSERT INTO tool_agent_turn_recipes(
                    thread_id, turn_id, origin_run_id, recipe_json, schema_version, created_at_epoch_ms,
                    lease_owner_id, lease_fence, lease_expires_at_epoch_ms,
                    terminal_message_id, terminal_marked_at_epoch_ms
                ) VALUES (?, ?, NULL, ?, ?, CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER),
                          NULL, 0, NULL, NULL, NULL)
                """,
                (target_thread_id, str(turn_id), canonical, int(schema_version)),
            )
            await self.conn.execute(
                """
                INSERT INTO tool_agent_turn_checkpoint_links(thread_id, turn_id, checkpoint_ns, checkpoint_id)
                SELECT ?, turn_id, checkpoint_ns, checkpoint_id
                FROM tool_agent_turn_checkpoint_links
                WHERE thread_id = ? AND turn_id = ?
                """,
                (target_thread_id, source_thread_id, str(turn_id)),
            )
            await self.conn.execute(
                """
                INSERT INTO tool_agent_turn_recipe_copy_provenance(
                    target_thread_id, target_turn_id, source_thread_id, source_turn_id, copied_at_epoch_ms
                ) VALUES (?, ?, ?, ?, CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER))
                """,
                (target_thread_id, str(turn_id), source_thread_id, str(turn_id)),
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


def _latest_human_message_id(checkpoint: Checkpoint) -> str | None:
    messages = checkpoint.get("channel_values", {}).get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        message_id = getattr(message, "id", None)
        if isinstance(message_id, str) and message_id.strip():
            return message_id.strip()
    return None


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
        typed_saver = cast(LocalAsyncSqliteSaver, saver)
        # Reconciliation is saver-owned: startup repairs crash leftovers and a
        # single bounded periodic task handles long-running Agent Server
        # processes. BEGIN IMMEDIATE serializes independent saver connections.
        await _reconcile_recipes(typed_saver)
        task = asyncio.create_task(_reconcile_recipes_periodically(typed_saver))
        try:
            yield typed_saver
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _reconcile_recipes(saver: LocalAsyncSqliteSaver) -> None:
    try:
        await saver.recipe_store.reconcile()
    except Exception:
        # Recipe payloads are deliberately not logged; the store logs only
        # identifier/reason pairs for removals.
        logger.exception("tool_agent_recipe_reconciliation_failed")


async def _reconcile_recipes_periodically(saver: LocalAsyncSqliteSaver) -> None:
    interval = max(
        1,
        int(
            os.environ.get(
                "TOOL_AGENT_RECIPE_RECONCILE_INTERVAL_SECONDS",
                DEFAULT_RECIPE_RECONCILE_INTERVAL_SECONDS,
            )
        ),
    )
    while True:
        await asyncio.sleep(interval)
        await _reconcile_recipes(saver)


__all__ = [
    "LocalAsyncSqliteSaver",
    "generate_checkpointer",
    "resolve_sqlite_checkpointer_path",
]
