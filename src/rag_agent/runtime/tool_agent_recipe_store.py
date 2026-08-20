from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

RECIPE_SCHEMA_VERSION = 1
DEFAULT_LEASE_DURATION_MS = 5 * 60 * 1000
ORPHAN_RECIPE_GRACE_MS = 15 * 60 * 1000
TERMINAL_RECIPE_RETENTION_MS = 7 * 24 * 60 * 60 * 1000
# Run links are crash/rollback provenance, not permanent reachability.  Keep
# them for at least the terminal/checkpoint retention window before allowing a
# genuinely uncheckpointed and inactive recipe to age out.
RUN_LINK_RETENTION_MS = TERMINAL_RECIPE_RETENTION_MS
_DB_EPOCH_MS = "CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)"

logger = logging.getLogger(__name__)

RecipeKey = tuple[str, str]


class CreateOrLoadStatus(StrEnum):
    CREATED = "created"
    EXISTING_IDENTICAL = "existing_identical"


class LeaseStatus(StrEnum):
    CLAIMED = "claimed"
    ALREADY_ACTIVE = "already_active"
    MISSING = "missing"
    RENEWED = "renewed"
    RELEASED = "released"
    MARKED = "marked"


class RecipeConflictError(ValueError):
    """Raised when an immutable recipe key receives a different payload."""


class MissingRecipeError(LookupError):
    """Raised when a lease mutation targets a missing recipe."""


class StaleLeaseError(RuntimeError):
    """Raised when a lease token no longer owns its recipe."""


@dataclass(frozen=True, slots=True)
class ToolAgentTurnRecipe:
    thread_id: str
    turn_id: str
    origin_run_id: str | None
    request_id: str | None
    session_id: str | None
    mode: Literal["mcp", "mixed"]
    model_key: str
    collection_key: str | None
    mcp_server_keys: tuple[str, ...]
    mcp_config_digest: str | None
    enable_tracing: bool
    tool_round_limit: int
    enable_reranker: bool = False
    schema_version: int = RECIPE_SCHEMA_VERSION
    created_at_epoch_ms: int | None = None
    lease_owner_id: str | None = None
    lease_fence: int = 0
    lease_expires_at_epoch_ms: int | None = None
    terminal_message_id: str | None = None
    terminal_marked_at_epoch_ms: int | None = None

    def canonical_json(self) -> bytes:
        payload = {
            "schema_version": self.schema_version,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "origin_run_id": self.origin_run_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "mode": self.mode,
            "model_key": self.model_key,
            "collection_key": self.collection_key,
            "mcp_server_keys": sorted(self.mcp_server_keys),
            "mcp_config_digest": self.mcp_config_digest,
            "enable_tracing": self.enable_tracing,
            "tool_round_limit": self.tool_round_limit,
            "enable_reranker": self.enable_reranker,
        }
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()

    @classmethod
    def from_storage(
        cls,
        recipe_json: bytes | str,
        created_at_epoch_ms: int,
        *,
        lease_owner_id: str | None = None,
        lease_fence: int = 0,
        lease_expires_at_epoch_ms: int | None = None,
        terminal_message_id: str | None = None,
        terminal_marked_at_epoch_ms: int | None = None,
    ) -> ToolAgentTurnRecipe:
        payload = json.loads(
            recipe_json.decode() if isinstance(recipe_json, bytes) else recipe_json
        )
        if not isinstance(payload, dict) or payload.get("schema_version") != RECIPE_SCHEMA_VERSION:
            raise ValueError("Unsupported tool-agent recipe schema")
        return cls(
            thread_id=str(payload["thread_id"]),
            turn_id=str(payload["turn_id"]),
            origin_run_id=_optional_string(payload.get("origin_run_id")),
            request_id=_optional_string(payload.get("request_id")),
            session_id=_optional_string(payload.get("session_id")),
            mode=payload["mode"],
            model_key=str(payload["model_key"]),
            collection_key=_optional_string(payload.get("collection_key")),
            mcp_server_keys=tuple(str(key) for key in payload.get("mcp_server_keys", [])),
            mcp_config_digest=_optional_string(payload.get("mcp_config_digest")),
            enable_tracing=bool(payload["enable_tracing"]),
            tool_round_limit=int(payload["tool_round_limit"]),
            # Additive field: old rows retain the historical non-reranked behavior.
            enable_reranker=bool(payload.get("enable_reranker", False)),
            schema_version=int(payload["schema_version"]),
            created_at_epoch_ms=int(created_at_epoch_ms),
            lease_owner_id=lease_owner_id,
            lease_fence=int(lease_fence),
            lease_expires_at_epoch_ms=lease_expires_at_epoch_ms,
            terminal_message_id=terminal_message_id,
            terminal_marked_at_epoch_ms=terminal_marked_at_epoch_ms,
        )


@dataclass(frozen=True, slots=True)
class CreateOrLoadResult:
    status: CreateOrLoadStatus
    recipe: ToolAgentTurnRecipe


@dataclass(frozen=True, slots=True)
class LeaseToken:
    thread_id: str
    turn_id: str
    owner_id: str
    fence: int


@dataclass(frozen=True, slots=True)
class LeaseResult:
    status: LeaseStatus
    recipe: ToolAgentTurnRecipe | None = None
    lease: LeaseToken | None = None


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    removed_recipe_count: int = 0


class ToolAgentTurnRecipeStore:
    """Recipe operations sharing a LocalAsyncSqliteSaver connection and lock."""

    def __init__(self, saver: Any) -> None:
        self._saver = saver

    async def create_or_load(self, recipe: ToolAgentTurnRecipe) -> CreateOrLoadResult:
        await self._saver.setup()
        canonical = recipe.canonical_json()
        async with self._saver.lock, self._saver.transaction_locked():
            created_at = await self._database_now_locked()
            try:
                await self._saver.conn.execute(
                    """
                    INSERT INTO tool_agent_turn_recipes (
                        thread_id, turn_id, origin_run_id, recipe_json,
                        schema_version, created_at_epoch_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recipe.thread_id,
                        recipe.turn_id,
                        recipe.origin_run_id,
                        canonical,
                        recipe.schema_version,
                        created_at,
                    ),
                )
            except sqlite3.IntegrityError:
                existing = await self._load_locked((recipe.thread_id, recipe.turn_id))
                if existing is None or existing.canonical_json() != canonical:
                    raise RecipeConflictError(
                        f"Recipe key already contains a different payload: "
                        f"{recipe.thread_id}/{recipe.turn_id}"
                    )
                await self._record_origin_link_locked(existing)
                return CreateOrLoadResult(CreateOrLoadStatus.EXISTING_IDENTICAL, existing)
            stored = ToolAgentTurnRecipe.from_storage(canonical, created_at)
            await self._record_origin_link_locked(stored)
            return CreateOrLoadResult(CreateOrLoadStatus.CREATED, stored)

    async def load(self, key: RecipeKey) -> ToolAgentTurnRecipe | None:
        await self._saver.setup()
        async with self._saver.lock:
            return await self._load_locked(key)

    async def claim(
        self,
        key: RecipeKey,
        owner_id: str,
        *,
        lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS,
    ) -> LeaseResult:
        if not owner_id.strip() or lease_duration_ms <= 0:
            raise ValueError("owner_id and lease_duration_ms must be valid")
        await self._saver.setup()
        async with self._saver.lock, self._saver.transaction_locked():
            recipe = await self._load_locked(key)
            if recipe is None:
                return LeaseResult(LeaseStatus.MISSING)
            now = await self._database_now_locked()
            async with self._saver.conn.execute(
                """
                UPDATE tool_agent_turn_recipes
                SET lease_owner_id = ?,
                    lease_fence = lease_fence + 1,
                    lease_expires_at_epoch_ms = ? + ?
                WHERE thread_id = ? AND turn_id = ?
                  AND (lease_owner_id IS NULL OR lease_expires_at_epoch_ms <= ?)
                """,
                (owner_id, now, lease_duration_ms, key[0], key[1], now),
            ) as cursor:
                claimed = cursor.rowcount == 1
            if not claimed:
                return LeaseResult(LeaseStatus.ALREADY_ACTIVE, recipe=recipe)
            async with self._saver.conn.execute(
                "SELECT lease_fence FROM tool_agent_turn_recipes WHERE thread_id = ? AND turn_id = ?",
                key,
            ) as cursor:
                fence = int((await cursor.fetchone())[0])
            return LeaseResult(
                LeaseStatus.CLAIMED,
                recipe=recipe,
                lease=LeaseToken(key[0], key[1], owner_id, fence),
            )

    async def renew(
        self,
        lease: LeaseToken,
        *,
        lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS,
    ) -> LeaseResult:
        return await self._mutate_lease(
            lease, LeaseStatus.RENEWED, lease_duration_ms=lease_duration_ms
        )

    async def release(self, lease: LeaseToken) -> LeaseResult:
        await self._saver.setup()
        async with self._saver.lock, self._saver.transaction_locked():
            now = await self._database_now_locked()
            async with self._saver.conn.execute(
                """
                UPDATE tool_agent_turn_recipes
                SET lease_owner_id = NULL, lease_expires_at_epoch_ms = NULL
                WHERE thread_id = ? AND turn_id = ? AND lease_owner_id = ? AND lease_fence = ?
                  AND lease_expires_at_epoch_ms > ?
                """,
                (lease.thread_id, lease.turn_id, lease.owner_id, lease.fence, now),
            ) as cursor:
                changed = cursor.rowcount == 1
            await self._finish_lease_mutation(changed, lease)
            return LeaseResult(LeaseStatus.RELEASED, lease=lease)

    async def mark_terminal(self, lease: LeaseToken, terminal_message_id: str) -> LeaseResult:
        if not terminal_message_id.strip():
            raise ValueError("terminal_message_id must not be empty")
        await self._saver.setup()
        async with self._saver.lock, self._saver.transaction_locked():
            async with self._saver.conn.execute(
                f"""
                UPDATE tool_agent_turn_recipes
                SET terminal_message_id = ?, terminal_marked_at_epoch_ms = {_DB_EPOCH_MS}
                WHERE thread_id = ? AND turn_id = ? AND lease_owner_id = ? AND lease_fence = ?
                  AND lease_expires_at_epoch_ms > {_DB_EPOCH_MS}
                """,
                (terminal_message_id, lease.thread_id, lease.turn_id, lease.owner_id, lease.fence),
            ) as cursor:
                changed = cursor.rowcount == 1
            await self._finish_lease_mutation(changed, lease)
            return LeaseResult(LeaseStatus.MARKED, lease=lease)

    async def delete_for_origin_runs(self, run_ids: list[str]) -> None:
        await self._saver.setup()
        normalized = [run_id for run_id in run_ids if run_id]
        if not normalized:
            return
        async with self._saver.lock, self._saver.transaction_locked():
            await self._delete_for_origin_runs_locked(normalized)

    async def record_run_link(self, key: RecipeKey, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        await self._saver.setup()
        async with self._saver.lock, self._saver.transaction_locked():
            recipe = await self._load_locked(key)
            if recipe is None:
                raise MissingRecipeError(f"Missing recipe: {key[0]}/{key[1]}")
            await self._record_run_link_locked(recipe, run_id)

    async def delete_for_thread(self, thread_id: str) -> None:
        await self._saver.setup()
        async with self._saver.lock, self._saver.transaction_locked():
            await self._delete_for_thread_locked(thread_id)

    async def reconcile(self) -> ReconciliationResult:
        await self._saver.setup()
        async with self._saver.lock, self._saver.transaction_locked():
            now = await self._database_now_locked()
            removed = await self._reconcile_locked(now)
        return ReconciliationResult(removed_recipe_count=removed)

    async def _delete_for_thread_locked(self, thread_id: str) -> None:
        await self._saver.conn.execute(
            "DELETE FROM tool_agent_turn_checkpoint_links WHERE thread_id = ?", (thread_id,)
        )
        await self._saver.conn.execute(
            "DELETE FROM tool_agent_turn_run_links WHERE thread_id = ?", (thread_id,)
        )
        await self._saver.conn.execute(
            "DELETE FROM tool_agent_turn_recipes WHERE thread_id = ?", (thread_id,)
        )

    async def _delete_for_origin_runs_locked(self, run_ids: list[str]) -> None:
        placeholders = ", ".join("?" for _ in run_ids)
        await self._saver.conn.execute(
            f"""
            DELETE FROM tool_agent_turn_run_links
            WHERE run_id IN ({placeholders})
            """,
            run_ids,
        )
        await self._saver.conn.execute(
            f"""
            UPDATE tool_agent_turn_recipes
            SET origin_run_id = NULL
            WHERE origin_run_id IN ({placeholders})
            """,
            run_ids,
        )
        await self._delete_unreachable_recipes_locked()

    async def _link_checkpoint_locked(
        self,
        *,
        thread_id: str,
        turn_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> None:
        """Record a committed checkpoint that can reconstruct a recipe turn."""

        async with self._saver.conn.execute(
            """
            SELECT 1 FROM tool_agent_turn_recipes
            WHERE thread_id = ? AND turn_id = ?
            """,
            (thread_id, turn_id),
        ) as cursor:
            if await cursor.fetchone() is None:
                return
        await self._saver.conn.execute(
            """
            INSERT OR REPLACE INTO tool_agent_turn_checkpoint_links(
                thread_id, turn_id, checkpoint_ns, checkpoint_id
            ) VALUES (?, ?, ?, ?)
            """,
            (thread_id, turn_id, checkpoint_ns, checkpoint_id),
        )

    async def _delete_checkpoint_links_locked(self, rows: list[tuple[str, str, str]]) -> None:
        for thread_id, checkpoint_ns, checkpoint_id in rows:
            await self._saver.conn.execute(
                """
                DELETE FROM tool_agent_turn_checkpoint_links
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            )

    async def _delete_unreachable_recipes_locked(self) -> int:
        """Remove only lifecycle-orphaned recipes that no live worker owns."""

        now = await self._database_now_locked()
        async with self._saver.conn.execute(
            """
            DELETE FROM tool_agent_turn_recipes AS recipe
            WHERE NOT EXISTS (
                SELECT 1 FROM tool_agent_turn_checkpoint_links AS checkpoint_link
                WHERE checkpoint_link.thread_id = recipe.thread_id
                  AND checkpoint_link.turn_id = recipe.turn_id
            )
              AND NOT EXISTS (
                SELECT 1 FROM tool_agent_turn_run_links AS run_link
                WHERE run_link.thread_id = recipe.thread_id
                  AND run_link.turn_id = recipe.turn_id
            )
              AND (recipe.lease_expires_at_epoch_ms IS NULL
                   OR recipe.lease_expires_at_epoch_ms <= ?)
            """,
            (now,),
        ) as cursor:
            return int(cursor.rowcount)

    async def _reconcile_locked(self, now: int) -> int:
        """Conservatively expire only unlinked, inactive recipes.

        A retained checkpoint can still be time-travelled or resumed, so its
        recipe is never expired here.  The timestamps deliberately use SQLite
        time rather than a worker clock.
        """

        await self._saver.conn.execute("""
            DELETE FROM tool_agent_turn_checkpoint_links
            WHERE NOT EXISTS (
                SELECT 1 FROM checkpoints
                WHERE checkpoints.thread_id = tool_agent_turn_checkpoint_links.thread_id
                  AND checkpoints.checkpoint_ns = tool_agent_turn_checkpoint_links.checkpoint_ns
                  AND checkpoints.checkpoint_id = tool_agent_turn_checkpoint_links.checkpoint_id
            )
            """)
        # A run link protects a recipe during rollback/replay, but must not
        # make a pre-setup crash orphan immortal.  Expire it using database
        # time only after the full retention window, and never while a worker
        # still owns a live lease.  Checkpoint links remain authoritative.
        await self._saver.conn.execute(
            """
            DELETE FROM tool_agent_turn_run_links AS run_link
            WHERE run_link.created_at_epoch_ms <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM tool_agent_turn_recipes AS recipe
                  WHERE recipe.thread_id = run_link.thread_id
                    AND recipe.turn_id = run_link.turn_id
                    AND recipe.lease_expires_at_epoch_ms > ?
              )
            """,
            (now - RUN_LINK_RETENTION_MS, now),
        )
        async with self._saver.conn.execute(
            """
            SELECT recipe.thread_id, recipe.turn_id,
                   CASE WHEN recipe.terminal_message_id IS NULL
                        THEN 'orphan_grace_elapsed'
                        ELSE 'terminal_retention_elapsed'
                   END AS reason
            FROM tool_agent_turn_recipes AS recipe
            WHERE NOT EXISTS (
                SELECT 1 FROM tool_agent_turn_checkpoint_links AS checkpoint_link
                WHERE checkpoint_link.thread_id = recipe.thread_id
                  AND checkpoint_link.turn_id = recipe.turn_id
            )
              AND NOT EXISTS (
                SELECT 1 FROM tool_agent_turn_run_links AS run_link
                WHERE run_link.thread_id = recipe.thread_id
                  AND run_link.turn_id = recipe.turn_id
            )
              AND (recipe.lease_expires_at_epoch_ms IS NULL
                   OR recipe.lease_expires_at_epoch_ms <= ?)
              AND (
                  (recipe.terminal_message_id IS NULL
                   AND recipe.created_at_epoch_ms <= ?)
                  OR (recipe.terminal_message_id IS NOT NULL
                      AND recipe.terminal_marked_at_epoch_ms <= ?)
              )
            """,
            (
                now,
                now - ORPHAN_RECIPE_GRACE_MS,
                now - TERMINAL_RECIPE_RETENTION_MS,
            ),
        ) as cursor:
            removable = await cursor.fetchall()
        for thread_id, turn_id, reason in removable:
            await self._saver.conn.execute(
                "DELETE FROM tool_agent_turn_recipes WHERE thread_id = ? AND turn_id = ?",
                (thread_id, turn_id),
            )
            logger.info(
                "tool_agent_recipe_reconciled thread_id=%s turn_id=%s reason=%s",
                thread_id,
                turn_id,
                reason,
            )
        return len(removable)

    async def _load_locked(self, key: RecipeKey) -> ToolAgentTurnRecipe | None:
        async with self._saver.conn.execute(
            """
            SELECT recipe_json, created_at_epoch_ms, lease_owner_id, lease_fence,
                   lease_expires_at_epoch_ms, terminal_message_id, terminal_marked_at_epoch_ms
            FROM tool_agent_turn_recipes
            WHERE thread_id = ? AND turn_id = ?
            """,
            key,
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return ToolAgentTurnRecipe.from_storage(
            row[0],
            int(row[1]),
            lease_owner_id=row[2],
            lease_fence=int(row[3]),
            lease_expires_at_epoch_ms=row[4],
            terminal_message_id=row[5],
            terminal_marked_at_epoch_ms=row[6],
        )

    async def _record_origin_link_locked(self, recipe: ToolAgentTurnRecipe) -> None:
        if recipe.origin_run_id is None:
            return
        await self._record_run_link_locked(recipe, recipe.origin_run_id)

    async def _record_run_link_locked(self, recipe: ToolAgentTurnRecipe, run_id: str) -> None:
        created_at = await self._database_now_locked()
        await self._saver.conn.execute(
            """
            INSERT OR IGNORE INTO tool_agent_turn_run_links(
                thread_id, run_id, turn_id, created_at_epoch_ms
            ) VALUES (?, ?, ?, ?)
            """,
            (recipe.thread_id, run_id, recipe.turn_id, created_at),
        )

    async def _database_now_locked(self) -> int:
        async with self._saver.conn.execute(f"SELECT {_DB_EPOCH_MS}") as cursor:
            return int((await cursor.fetchone())[0])

    async def _mutate_lease(
        self,
        lease: LeaseToken,
        status: LeaseStatus,
        *,
        lease_duration_ms: int,
    ) -> LeaseResult:
        if lease_duration_ms <= 0:
            raise ValueError("lease_duration_ms must be positive")
        await self._saver.setup()
        async with self._saver.lock, self._saver.transaction_locked():
            now = await self._database_now_locked()
            async with self._saver.conn.execute(
                """
                UPDATE tool_agent_turn_recipes
                SET lease_expires_at_epoch_ms = ? + ?
                WHERE thread_id = ? AND turn_id = ? AND lease_owner_id = ? AND lease_fence = ?
                  AND lease_expires_at_epoch_ms > ?
                """,
                (
                    now,
                    lease_duration_ms,
                    lease.thread_id,
                    lease.turn_id,
                    lease.owner_id,
                    lease.fence,
                    now,
                ),
            ) as cursor:
                changed = cursor.rowcount == 1
            await self._finish_lease_mutation(changed, lease)
            return LeaseResult(status, lease=lease)

    async def _finish_lease_mutation(self, changed: bool, lease: LeaseToken) -> None:
        if changed:
            return
        async with self._saver.conn.execute(
            "SELECT 1 FROM tool_agent_turn_recipes WHERE thread_id = ? AND turn_id = ?",
            (lease.thread_id, lease.turn_id),
        ) as cursor:
            exists = await cursor.fetchone()
        if exists is None:
            raise MissingRecipeError(f"Missing recipe: {lease.thread_id}/{lease.turn_id}")
        raise StaleLeaseError(f"Stale lease fence: {lease.thread_id}/{lease.turn_id}/{lease.fence}")


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "CreateOrLoadResult",
    "CreateOrLoadStatus",
    "DEFAULT_LEASE_DURATION_MS",
    "RUN_LINK_RETENTION_MS",
    "LeaseResult",
    "LeaseStatus",
    "LeaseToken",
    "MissingRecipeError",
    "RecipeConflictError",
    "ReconciliationResult",
    "StaleLeaseError",
    "ToolAgentTurnRecipe",
    "ToolAgentTurnRecipeStore",
]
