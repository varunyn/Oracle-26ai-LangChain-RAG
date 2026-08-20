from __future__ import annotations

import asyncio
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt


class ProbeState(TypedDict, total=False):
    messages: Annotated[list[object], add_messages]


class ProbeContext(TypedDict):
    observations: list[tuple[str, str | None, str, str, int]]


def _latest_human_id(state: ProbeState) -> str | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.id
    return None


def _record(name: str, state: ProbeState, runtime: Runtime[ProbeContext]) -> None:
    info = runtime.execution_info
    runtime.context["observations"].append(
        (
            name,
            _latest_human_id(state),
            info.checkpoint_ns,
            info.task_id,
            info.node_attempt,
        )
    )


def _build_probe_graph(*, interrupt_in_subgraph: bool = False):
    sub_builder = StateGraph(ProbeState, context_schema=ProbeContext)

    def subgraph_node(state: ProbeState, runtime: Runtime[ProbeContext]) -> ProbeState:
        _record("subgraph", state, runtime)
        if interrupt_in_subgraph:
            interrupt("approval")
        return {"messages": [AIMessage(content="subgraph-complete")]}

    sub_builder.add_node("subgraph_node", subgraph_node)
    sub_builder.add_edge(START, "subgraph_node")
    sub_builder.add_edge("subgraph_node", END)
    subgraph = sub_builder.compile()

    builder = StateGraph(ProbeState, context_schema=ProbeContext)

    def setup(state: ProbeState, runtime: Runtime[ProbeContext]) -> ProbeState:
        _record("setup", state, runtime)
        return {}

    def compose(state: ProbeState, runtime: Runtime[ProbeContext]) -> ProbeState:
        _record("compose", state, runtime)
        return {"messages": [AIMessage(content="terminal", id="terminal-message")]}

    builder.add_node("setup", setup)
    builder.add_node("tool_loop", subgraph)
    builder.add_node("compose", compose)
    builder.add_edge(START, "setup")
    builder.add_edge("setup", "tool_loop")
    builder.add_edge("tool_loop", "compose")
    builder.add_edge("compose", END)
    return builder


def test_human_message_id_survives_parent_subgraph_interrupt_resume_replay_and_new_turn(
    tmp_path,
) -> None:
    async def probe() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "identity.sqlite")) as saver:
            await saver.setup()
            graph = _build_probe_graph(interrupt_in_subgraph=True).compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "identity-thread"}}
            context: ProbeContext = {"observations": []}

            paused = await graph.ainvoke(
                {"messages": [HumanMessage(content="first", id="turn-1")]},
                config=config,
                context=context,
            )
            assert "__interrupt__" in paused
            assert {row[1] for row in context["observations"]} == {"turn-1"}

            resumed = await graph.ainvoke(Command(resume="approved"), config=config, context=context)
            assert resumed["messages"][-1].id == "terminal-message"
            assert {row[1] for row in context["observations"]} == {"turn-1"}

            replay = await graph.aget_state(config)
            replay_ids = [message.id for message in replay.values["messages"] if isinstance(message, HumanMessage)]
            assert replay_ids == ["turn-1"]
            assert replay.config["configurable"]["checkpoint_ns"] == ""

            await graph.ainvoke(
                {"messages": [HumanMessage(content="second", id="turn-2")]},
                config=config,
                context=context,
            )
            assert {row[1] for row in context["observations"] if row[0] == "setup"} >= {"turn-1", "turn-2"}

            namespaces = {row[2] for row in context["observations"]}
            assert any(namespace for namespace in namespaces), "subgraph must have a distinct checkpoint namespace"

    asyncio.run(probe())


def test_checkpointer_persists_generated_id_for_idless_message_and_stateless_is_ephemeral(
    tmp_path,
) -> None:
    async def probe() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "idless.sqlite")) as saver:
            await saver.setup()
            graph = _build_probe_graph().compile(checkpointer=saver)
            context: ProbeContext = {"observations": []}
            await graph.ainvoke(
                {"messages": [HumanMessage(content="no explicit id")]},
                config={"configurable": {"thread_id": "idless-thread"}},
                context=context,
            )
            generated_ids = {row[1] for row in context["observations"]}
            assert len(generated_ids) == 1
            assert next(iter(generated_ids))

        stateless_context: ProbeContext = {"observations": []}
        graph = _build_probe_graph().compile()
        await graph.ainvoke(
            {"messages": [HumanMessage(content="stateless")]},
            context=stateless_context,
        )
        stateless_ids = {row[1] for row in stateless_context["observations"]}
        assert len(stateless_ids) == 1
        assert next(iter(stateless_ids))

    asyncio.run(probe())


def test_execution_info_supports_fenced_owner_identity_and_terminal_checkpoint_signal(
    tmp_path,
) -> None:
    async def probe() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "owner.sqlite")) as saver:
            await saver.setup()
            graph = _build_probe_graph().compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "owner-thread"}}
            context: ProbeContext = {"observations": []}

            stream = [
                part
                async for part in graph.astream(
                    {"messages": [HumanMessage(content="first", id="turn-owner")]},
                    config=config,
                    context=context,
                    stream_mode="debug",
                    version="v2",
                )
            ]
            owner_tokens = {
                f"lease-owner|owner-thread|{namespace}|{task_id}|{attempt}"
                for _, _, namespace, task_id, attempt in context["observations"]
            }
            assert all(token not in repr(part) for part in stream for token in owner_tokens)

            owners = {
                (thread_id, namespace, task_id, attempt)
                for _, turn_id, namespace, task_id, attempt in context["observations"]
                for thread_id in ["owner-thread"]
                if turn_id == "turn-owner"
            }
            assert len(owners) == len(context["observations"])
            assert all(namespace and task_id and attempt >= 1 for _, namespace, task_id, attempt in owners)

            checkpoint = await saver.aget_tuple(config)
            assert checkpoint is not None
            messages = checkpoint.checkpoint["channel_values"]["messages"]
            assert any(isinstance(message, AIMessage) and message.id == "terminal-message" for message in messages)
            assert checkpoint.metadata.get("step") is not None
            assert not checkpoint.pending_writes

    asyncio.run(probe())


def test_saver_connection_can_atomically_share_checkpoint_transaction_without_nested_lock(
    tmp_path,
) -> None:
    async def probe() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "transaction.sqlite")) as saver:
            await saver.setup()
            assert isinstance(saver.lock, asyncio.Lock)
            await saver.conn.execute("CREATE TABLE probe_lifecycle (value TEXT PRIMARY KEY)")
            await saver.conn.commit()

            async with saver.lock:
                await saver.conn.execute("BEGIN IMMEDIATE")
                await saver.conn.execute("INSERT INTO probe_lifecycle(value) VALUES ('same-tx')")
                await saver.conn.execute(
                    "INSERT INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) VALUES (?, '', ?, ?, ?, ?)",
                    ("tx-thread", "tx-checkpoint", "json", b"{}", b"{}"),
                )
                await saver.conn.rollback()

            async with saver.conn.execute("SELECT value FROM probe_lifecycle") as cursor:
                assert await cursor.fetchone() is None
            assert await saver.aget_tuple({"configurable": {"thread_id": "tx-thread"}}) is None

    asyncio.run(probe())


def test_two_sqlite_connections_allow_only_one_concurrent_claim(tmp_path) -> None:
    async def claim(saver: AsyncSqliteSaver, owner_id: str) -> tuple[bool, int | None]:
        async with saver.lock:
            await saver.conn.execute("BEGIN IMMEDIATE")
            cursor = await saver.conn.execute(
                """
                UPDATE tool_agent_turn_lease_probe
                SET lease_owner_id = ?, lease_fence = lease_fence + 1,
                    lease_expires_at_epoch_ms = 2000
                WHERE thread_id = 'claim-thread' AND turn_id = 'claim-turn'
                  AND (lease_owner_id IS NULL OR lease_expires_at_epoch_ms <= 1000)
                """,
                (owner_id,),
            )
            claimed = cursor.rowcount == 1
            await cursor.close()
            await saver.conn.commit()
            if not claimed:
                return False, None
            async with saver.conn.execute(
                "SELECT lease_fence FROM tool_agent_turn_lease_probe"
            ) as fence_cursor:
                row = await fence_cursor.fetchone()
            return True, int(row[0])

    async def probe() -> None:
        db_path = str(tmp_path / "concurrent-claim.sqlite")
        async with (
            AsyncSqliteSaver.from_conn_string(db_path) as first,
            AsyncSqliteSaver.from_conn_string(db_path) as second,
        ):
            await first.setup()
            await first.conn.execute(
                """
                CREATE TABLE tool_agent_turn_lease_probe (
                    thread_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    lease_owner_id TEXT,
                    lease_fence INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at_epoch_ms INTEGER,
                    PRIMARY KEY (thread_id, turn_id)
                )
                """
            )
            await first.conn.execute(
                "INSERT INTO tool_agent_turn_lease_probe(thread_id, turn_id) VALUES ('claim-thread', 'claim-turn')"
            )
            await first.conn.commit()
            owner_ids = (str(uuid4()), str(uuid4()))
            results = await asyncio.gather(
                claim(first, owner_ids[0]),
                claim(second, owner_ids[1]),
            )
            assert sum(result[0] for result in results) == 1
            assert owner_ids[0] != owner_ids[1]
            async with first.conn.execute(
                "SELECT lease_owner_id, lease_fence FROM tool_agent_turn_lease_probe"
            ) as cursor:
                owner, fence = await cursor.fetchone()
            assert owner in owner_ids
            assert fence == 1

    asyncio.run(probe())


def test_expired_takeover_fences_every_stale_token_mutation(tmp_path) -> None:
    async def claim(saver: AsyncSqliteSaver, owner_id: str, now: int) -> tuple[str, int]:
        async with saver.lock:
            await saver.conn.execute("BEGIN IMMEDIATE")
            cursor = await saver.conn.execute(
                """
                UPDATE tool_agent_turn_lease_probe
                SET lease_owner_id = ?, lease_fence = lease_fence + 1,
                    lease_expires_at_epoch_ms = ? + 1000
                WHERE thread_id = 'fence-thread' AND turn_id = 'fence-turn'
                  AND (lease_owner_id IS NULL OR lease_expires_at_epoch_ms <= ?)
                """,
                (owner_id, now, now),
            )
            assert cursor.rowcount == 1
            await cursor.close()
            await saver.conn.commit()
            async with saver.conn.execute(
                "SELECT lease_fence FROM tool_agent_turn_lease_probe"
            ) as fence_cursor:
                return owner_id, int((await fence_cursor.fetchone())[0])

    async def stale_mutation(saver: AsyncSqliteSaver, owner_id: str, fence: int) -> int:
        async with saver.lock:
            cursor = await saver.conn.execute(
                """
                UPDATE tool_agent_turn_lease_probe
                SET lease_expires_at_epoch_ms = 9999
                WHERE thread_id = 'fence-thread' AND turn_id = 'fence-turn'
                  AND lease_owner_id = ? AND lease_fence = ?
                """,
                (owner_id, fence),
            )
            await saver.conn.commit()
            count = cursor.rowcount
            await cursor.close()
            return count

    async def probe() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "fencing.sqlite")) as saver:
            await saver.setup()
            await saver.conn.execute(
                """
                CREATE TABLE tool_agent_turn_lease_probe (
                    thread_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    lease_owner_id TEXT,
                    lease_fence INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at_epoch_ms INTEGER,
                    PRIMARY KEY (thread_id, turn_id)
                )
                """
            )
            await saver.conn.execute(
                "INSERT INTO tool_agent_turn_lease_probe(thread_id, turn_id) VALUES ('fence-thread', 'fence-turn')"
            )
            await saver.conn.commit()
            old_token = await claim(saver, str(uuid4()), 1000)
            await saver.conn.execute(
                "UPDATE tool_agent_turn_lease_probe SET lease_expires_at_epoch_ms = 1000"
            )
            await saver.conn.commit()
            new_token = await claim(saver, str(uuid4()), 1000)
            assert new_token[1] == old_token[1] + 1
            assert await stale_mutation(saver, *old_token) == 0
            async with saver.conn.execute(
                "SELECT lease_owner_id, lease_fence, lease_expires_at_epoch_ms FROM tool_agent_turn_lease_probe"
            ) as cursor:
                owner, fence, expiry = await cursor.fetchone()
            assert (owner, fence) == new_token
            assert expiry == 2000

    asyncio.run(probe())
