from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint

from src.rag_agent.runtime.agent_server_checkpointer import LocalAsyncSqliteSaver
from src.rag_agent.runtime.tool_agent_recipe_store import (
    RUN_LINK_RETENTION_MS,
    CreateOrLoadStatus,
    LeaseStatus,
    RecipeConflictError,
    StaleLeaseError,
    ToolAgentTurnRecipe,
)


def _recipe(*, thread_id: str = "thread-1", turn_id: str = "turn-1") -> ToolAgentTurnRecipe:
    return ToolAgentTurnRecipe(
        thread_id=thread_id,
        turn_id=turn_id,
        origin_run_id="run-1",
        request_id="request-1",
        session_id="session-1",
        mode="mcp",
        model_key="model-1",
        collection_key=None,
        mcp_server_keys=("calculator",),
        mcp_config_digest="mcp-digest-1",
        enable_tracing=True,
        tool_round_limit=5,
    )


def test_recipe_creation_is_idempotent_and_conflicts_are_not_overwritten(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "recipes.sqlite")
        ) as saver:
            store = saver.recipe_store
            first = await store.create_or_load(_recipe())
            identical = await store.create_or_load(_recipe())
            assert first.status is CreateOrLoadStatus.CREATED
            assert identical.status is CreateOrLoadStatus.EXISTING_IDENTICAL
            assert identical.recipe.created_at_epoch_ms == first.recipe.created_at_epoch_ms

            with pytest.raises(RecipeConflictError):
                await store.create_or_load(_recipe_with_model("different-model"))
            loaded = await store.load(("thread-1", "turn-1"))
            assert loaded == first.recipe

    asyncio.run(probe())


def test_old_recipe_rows_default_additive_reranker_field_to_false() -> None:
    old_payload = """{
        "schema_version": 1,
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "origin_run_id": null,
        "request_id": null,
        "session_id": null,
        "mode": "mixed",
        "model_key": "model-1",
        "collection_key": "collection-1",
        "mcp_server_keys": [],
        "mcp_config_digest": null,
        "enable_tracing": false,
        "tool_round_limit": 4
    }"""

    recipe = ToolAgentTurnRecipe.from_storage(old_payload, 123)

    assert recipe.enable_reranker is False


def test_lease_claim_renew_release_and_terminal_mark_are_fenced(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "leases.sqlite")) as saver:
            store = saver.recipe_store
            await store.create_or_load(_recipe())
            claimed = await store.claim(("thread-1", "turn-1"), str(uuid4()))
            assert claimed.status is LeaseStatus.CLAIMED
            assert claimed.lease is not None
            renewed = await store.renew(claimed.lease)
            assert renewed.status is LeaseStatus.RENEWED
            marked = await store.mark_terminal(claimed.lease, "terminal-1")
            assert marked.status is LeaseStatus.MARKED
            released = await store.release(claimed.lease)
            assert released.status is LeaseStatus.RELEASED
            assert (await store.load(("thread-1", "turn-1"))).terminal_message_id == "terminal-1"

    asyncio.run(probe())


def test_stale_lease_token_fails_closed_after_takeover(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "stale.sqlite")) as saver:
            store = saver.recipe_store
            await store.create_or_load(_recipe())
            old = await store.claim(("thread-1", "turn-1"), "owner-old", lease_duration_ms=1)
            assert old.lease is not None
            await asyncio.sleep(0.02)
            successor = await store.claim(("thread-1", "turn-1"), "owner-new")
            assert successor.lease is not None
            assert successor.lease.fence == old.lease.fence + 1
            with pytest.raises(StaleLeaseError):
                await store.renew(old.lease)
            with pytest.raises(StaleLeaseError):
                await store.release(old.lease)
            with pytest.raises(StaleLeaseError):
                await store.mark_terminal(old.lease, "stale-terminal")

    asyncio.run(probe())


def test_two_saver_connections_have_one_claim_winner(tmp_path) -> None:
    async def probe() -> None:
        db_path = str(tmp_path / "concurrent.sqlite")
        async with (
            LocalAsyncSqliteSaver.from_conn_string(db_path) as first,
            LocalAsyncSqliteSaver.from_conn_string(db_path) as second,
        ):
            await first.recipe_store.create_or_load(_recipe())
            results = await asyncio.gather(
                first.recipe_store.claim(("thread-1", "turn-1"), "owner-a"),
                second.recipe_store.claim(("thread-1", "turn-1"), "owner-b"),
            )
            assert sum(result.status is LeaseStatus.CLAIMED for result in results) == 1

    asyncio.run(probe())


def test_thread_and_origin_run_cleanup_remove_recipe_rows(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "cleanup.sqlite")
        ) as saver:
            store = saver.recipe_store
            await store.create_or_load(_recipe(thread_id="thread-a", turn_id="turn-a"))
            await store.create_or_load(_recipe(thread_id="thread-b", turn_id="turn-b"))
            await store.delete_for_origin_runs(["run-1"])
            assert await store.load(("thread-a", "turn-a")) is None
            assert await store.load(("thread-b", "turn-b")) is None

            await store.create_or_load(_recipe(thread_id="thread-c", turn_id="turn-c"))
            await store.delete_for_thread("thread-c")
            assert await store.load(("thread-c", "turn-c")) is None

    asyncio.run(probe())


def test_saver_thread_deletion_removes_checkpoints_and_recipes_together(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "thread-delete.sqlite")
        ) as saver:
            await saver.recipe_store.create_or_load(_recipe())
            await saver.conn.execute("""
                INSERT INTO checkpoints(
                    thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata
                ) VALUES ('thread-1', '', 'checkpoint-1', 'json', X'7B7D', X'7B7D')
                """)
            await saver.conn.commit()
            await saver.adelete_thread("thread-1")
            assert await saver.recipe_store.load(("thread-1", "turn-1")) is None
            assert await saver.aget_tuple({"configurable": {"thread_id": "thread-1"}}) is None

    asyncio.run(probe())


def test_saver_run_deletion_removes_origin_recipe_and_run_link_in_same_operation(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "run-delete.sqlite")
        ) as saver:
            await saver.recipe_store.create_or_load(_recipe())
            await saver.conn.execute(
                """
                INSERT INTO checkpoints(
                    thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata
                ) VALUES ('thread-1', '', 'checkpoint-1', 'json', X'7B7D', ?)
                """,
                (b'{"run_id":"run-1"}',),
            )
            await saver.conn.commit()
            await saver.adelete_for_runs(["run-1"])
            assert await saver.recipe_store.load(("thread-1", "turn-1")) is None
            async with saver.conn.execute(
                "SELECT COUNT(*) FROM tool_agent_turn_run_links WHERE run_id = 'run-1'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0

    asyncio.run(probe())


def test_failed_recipe_transaction_rolls_back_and_connection_recovers(
    tmp_path, monkeypatch
) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "rollback.sqlite")
        ) as saver:
            store = saver.recipe_store
            original = store._record_origin_link_locked

            async def fail_after_insert(_recipe) -> None:
                raise RuntimeError("injected recipe-link failure")

            monkeypatch.setattr(store, "_record_origin_link_locked", fail_after_insert)
            with pytest.raises(RuntimeError, match="injected"):
                await store.create_or_load(_recipe())
            assert not saver.conn.in_transaction
            monkeypatch.setattr(store, "_record_origin_link_locked", original)

            result = await store.create_or_load(_recipe())
            assert result.status is CreateOrLoadStatus.CREATED
            assert await store.load(("thread-1", "turn-1")) is not None

    asyncio.run(probe())


def test_continuation_link_is_idempotent_and_preserves_resumable_recipe(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "continuation.sqlite")
        ) as saver:
            store = saver.recipe_store
            await store.create_or_load(_recipe())
            await store.record_run_link(("thread-1", "turn-1"), "run-continuation")
            await store.record_run_link(("thread-1", "turn-1"), "run-continuation")
            async with saver.conn.execute(
                "SELECT COUNT(*) FROM tool_agent_turn_run_links WHERE run_id = 'run-continuation'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1

            await saver.adelete_for_runs(["run-1"])
            assert await store.load(("thread-1", "turn-1")) is not None
            await saver.adelete_for_runs(["run-continuation"])
            assert await store.load(("thread-1", "turn-1")) is None

    asyncio.run(probe())


def test_checkpoint_link_keeps_terminal_recipe_until_checkpoint_lifecycle_removes_it(
    tmp_path,
) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "terminal-retention.sqlite")
        ) as saver:
            store = saver.recipe_store
            await store.create_or_load(_recipe())
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {
                "messages": [HumanMessage(id="turn-1", content="question")]
            }
            config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
            await saver.aput(
                config,
                checkpoint,
                {"source": "loop", "step": 1, "parents": {}, "run_id": "run-1"},
                {},
            )

            claimed = await store.claim(("thread-1", "turn-1"), "terminal-owner")
            assert claimed.lease is not None
            await store.mark_terminal(claimed.lease, "final-answer")
            await store.release(claimed.lease)
            assert (await store.reconcile()).removed_recipe_count == 0
            assert await store.load(("thread-1", "turn-1")) is not None

            await saver.adelete_for_runs(["run-1"])
            assert await store.load(("thread-1", "turn-1")) is None

    asyncio.run(probe())


def test_checkpoint_and_recipe_link_roll_back_together_when_linking_fails(
    tmp_path, monkeypatch
) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "atomic.sqlite")) as saver:
            await saver.recipe_store.create_or_load(_recipe())
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {
                "messages": [HumanMessage(id="turn-1", content="question")]
            }

            async def fail_link(**_kwargs) -> None:
                raise RuntimeError("injected checkpoint-link failure")

            monkeypatch.setattr(saver.recipe_store, "_link_checkpoint_locked", fail_link)
            config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
            with pytest.raises(RuntimeError, match="checkpoint-link"):
                await saver.aput(
                    config, checkpoint, {"source": "loop", "step": 1, "parents": {}}, {}
                )
            assert await saver.aget_tuple(config) is None

    asyncio.run(probe())


def test_copy_preserves_reachability_but_resets_recipe_lifecycle_state(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(str(tmp_path / "copy.sqlite")) as saver:
            await saver.recipe_store.create_or_load(_recipe())
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {
                "messages": [HumanMessage(id="turn-1", content="question")]
            }
            await saver.aput(
                {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}},
                checkpoint,
                {"source": "loop", "step": 1, "parents": {}},
                {},
            )
            claimed = await saver.recipe_store.claim(("thread-1", "turn-1"), "owner")
            assert claimed.lease is not None
            await saver.recipe_store.mark_terminal(claimed.lease, "terminal")
            await saver.recipe_store.release(claimed.lease)

            await saver.acopy_thread("thread-1", "thread-copy")

            copied = await saver.recipe_store.load(("thread-copy", "turn-1"))
            assert copied is not None
            assert copied.origin_run_id is None
            assert copied.terminal_message_id is None
            assert copied.lease_owner_id is None

    asyncio.run(probe())


def test_copy_rejects_a_thread_with_an_active_recipe_lease(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "active-copy.sqlite")
        ) as saver:
            await saver.recipe_store.create_or_load(_recipe())
            claimed = await saver.recipe_store.claim(("thread-1", "turn-1"), "owner")
            assert claimed.lease is not None
            with pytest.raises(RuntimeError, match="active tool-agent recipe lease"):
                await saver.acopy_thread("thread-1", "thread-copy")

    asyncio.run(probe())


def test_recipe_run_link_foreign_key_is_enabled_and_cascades(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "foreign-keys.sqlite")
        ) as saver:
            await saver.setup()
            async with saver.conn.execute("PRAGMA foreign_keys") as cursor:
                assert (await cursor.fetchone())[0] == 1
            await saver.recipe_store.create_or_load(_recipe())
            with pytest.raises(sqlite3.IntegrityError):
                await saver.conn.execute("""
                    INSERT INTO tool_agent_turn_run_links(
                        thread_id, run_id, turn_id, created_at_epoch_ms
                    ) VALUES ('missing', 'run-missing', 'turn-missing', 1)
                    """)
            await saver.conn.execute(
                "DELETE FROM tool_agent_turn_recipes WHERE thread_id = 'thread-1' AND turn_id = 'turn-1'"
            )
            await saver.conn.commit()
            async with saver.conn.execute(
                "SELECT COUNT(*) FROM tool_agent_turn_run_links WHERE thread_id = 'thread-1'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0

    asyncio.run(probe())


def test_reconciliation_expires_old_run_links_then_removes_uncheckpointed_orphans(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "reconcile.sqlite")
        ) as saver:
            store = saver.recipe_store
            await store.create_or_load(_recipe())
            async with saver.lock, saver.transaction_locked():
                now = await store._database_now_locked()
                await saver.conn.execute(
                    "UPDATE tool_agent_turn_run_links SET created_at_epoch_ms = ?",
                    (now - RUN_LINK_RETENTION_MS,),
                )
                await saver.conn.execute(
                    "UPDATE tool_agent_turn_recipes SET created_at_epoch_ms = ?",
                    (now - RUN_LINK_RETENTION_MS,),
                )
            assert (await store.reconcile()).removed_recipe_count == 1
            assert await store.load(("thread-1", "turn-1")) is None

    asyncio.run(probe())


def test_reconciliation_preserves_checkpointed_and_live_leased_recipes(tmp_path) -> None:
    async def probe() -> None:
        async with LocalAsyncSqliteSaver.from_conn_string(
            str(tmp_path / "reconcile-live.sqlite")
        ) as saver:
            store = saver.recipe_store
            await store.create_or_load(_recipe())
            claimed = await store.claim(("thread-1", "turn-1"), "owner")
            assert claimed.lease is not None
            async with saver.lock, saver.transaction_locked():
                now = await store._database_now_locked()
                await saver.conn.execute(
                    "UPDATE tool_agent_turn_run_links SET created_at_epoch_ms = ?",
                    (now - RUN_LINK_RETENTION_MS,),
                )
                await saver.conn.execute(
                    "UPDATE tool_agent_turn_recipes SET created_at_epoch_ms = ?",
                    (now - RUN_LINK_RETENTION_MS,),
                )
            assert (await store.reconcile()).removed_recipe_count == 0
            assert await store.load(("thread-1", "turn-1")) is not None

    asyncio.run(probe())


def test_copy_rejects_active_target_lease_across_saver_connections(tmp_path) -> None:
    async def probe() -> None:
        db_path = str(tmp_path / "target-copy.sqlite")
        async with (
            LocalAsyncSqliteSaver.from_conn_string(db_path) as first,
            LocalAsyncSqliteSaver.from_conn_string(db_path) as second,
        ):
            await first.recipe_store.create_or_load(
                _recipe(thread_id="target", turn_id="turn-target")
            )
            claimed = await second.recipe_store.claim(("target", "turn-target"), "target-owner")
            assert claimed.lease is not None
            with pytest.raises(RuntimeError, match="active tool-agent recipe lease"):
                await first.acopy_thread("source", "target")
            assert await second.recipe_store.load(("target", "turn-target")) is not None

    asyncio.run(probe())


def _recipe_with_model(model_key: str) -> ToolAgentTurnRecipe:
    return replace(_recipe(), model_key=model_key)
