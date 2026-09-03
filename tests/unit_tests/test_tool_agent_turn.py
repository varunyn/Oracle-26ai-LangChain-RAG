from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from src.rag_agent.graphs import tool_agent_execution, tool_agent_turn
from src.rag_agent.runtime.agent_server_checkpointer import LocalAsyncSqliteSaver
from src.rag_agent.runtime.tool_agent_recipe_store import StaleLeaseError


def test_prepare_tool_agent_turn_builds_one_execution_value_from_durable_recipe(
    monkeypatch,
) -> None:
    class FakeTool:
        name = "calculator"
        description = "Adds values"

        def __call__(self, *_args: object, **_kwargs: object) -> str:
            return "42"

    loaded_tool = FakeTool()

    async def fake_load_adapter_tools(**_kwargs: object) -> list[object]:
        return [loaded_tool]

    monkeypatch.setattr(
        tool_agent_turn,
        "load_adapter_tools",
        fake_load_adapter_tools,
    )
    monkeypatch.setattr(
        tool_agent_turn,
        "_mcp_definitions",
        lambda: {"calculator": {"transport": "stdio", "command": "calculator"}},
    )
    runtime = SimpleNamespace(
        context={
            "model_id": "model-c",
            "session_id": "session-mcp",
            "enable_tracing": True,
            "mcp_server_keys": ["calculator"],
        },
        execution_info=SimpleNamespace(thread_id="thread-123", run_id="run-1"),
    )

    async def run() -> object:
        async with LocalAsyncSqliteSaver.from_conn_string(":memory:") as saver:
            return await tool_agent_turn.prepare_tool_agent_turn(
                state={
                    "messages": [
                        HumanMessage(id="h1", content="Remember this context"),
                        AIMessage(content="I will"),
                        HumanMessage(id="h2", content="19 + 23"),
                    ]
                },
                parent_config={
                    "callbacks": ["outer-callback"],
                    "configurable": {"__pregel_checkpointer": saver},
                },
                runtime=runtime,
                mode="mcp",
            )

    turn = asyncio.run(run())
    assert isinstance(turn, dict)

    assert turn["question"] == "19 + 23"
    assert [message.content for message in turn["chat_history"]] == [
        "Remember this context",
        "I will",
    ]
    assert turn["model_id"] == "model-c"
    assert turn["tools"] == [loaded_tool]
    assert "calculator: Adds values" in turn["system_prompt"]
    assert turn["run_config"]["callbacks"] == ["outer-callback"]
    configurable = turn["run_config"]["configurable"]
    assert configurable["mode"] == "mcp"
    assert configurable["enable_tracing"] is True
    assert configurable["thread_id"] == "thread-123"
    assert configurable["model_id"] == "model-c"
    assert configurable["session_id"] == "session-mcp"
    assert configurable["mcp_server_keys"] == ["calculator"]


def test_mcp_digest_excludes_credentials_but_detects_behavioral_drift(monkeypatch) -> None:
    definitions = {
        "server": {
            "transport": "streamable-http",
            "url": "https://one.example/mcp",
            "auth": {
                "type": "bearer",
                "bearer_token": "token-one",
                "client_secret": "secret-one",
                "client_id": "client-one",
                "token_url": "https://auth-one.example/token",
            },
            "headers": {
                "Authorization": "Bearer token-one",
                "Cookie": "session-one",
                "X-Api-Key": "key-one",
                "X-Tenant-ID": "tenant-one",
                "X-Trace": "one",
            },
            "tool_names": ["calculator"],
            "args": ["--mode", "strict", "--token", "token-one"],
            "env": {"MCP_MODE": "strict", "MCP_TOKEN": "token-one"},
            "cwd": "/srv/mcp",
            "timeout": 30,
            "keep_alive": True,
        }
    }
    monkeypatch.setattr(tool_agent_turn, "_mcp_definitions", lambda: definitions)

    baseline = tool_agent_turn._mcp_digest(("server",), {})
    definitions["server"]["auth"]["bearer_token"] = "token-two"
    definitions["server"]["auth"]["client_secret"] = "secret-two"
    definitions["server"]["headers"] = {
        "Authorization": "Bearer token-two",
        "Cookie": "session-two",
        "X-Api-Key": "key-two",
        "X-Tenant-ID": "tenant-one",
        "X-Trace": "one",
    }
    definitions["server"]["args"] = ["--mode", "strict", "--token", "token-two"]
    definitions["server"]["env"] = {"MCP_MODE": "strict", "MCP_TOKEN": "token-two"}
    assert tool_agent_turn._mcp_digest(("server",), {}) == baseline

    original = copy.deepcopy(definitions["server"])
    for field, value in (
        ("url", "https://two.example/mcp"),
        ("transport", "stdio"),
        ("tool_names", ["calculator", "search"]),
        (
            "headers",
            {
                "Authorization": "Bearer token-two",
                "Cookie": "session-two",
                "X-Api-Key": "key-two",
                "X-Tenant-ID": "tenant-two",
                "X-Trace": "two",
            },
        ),
        ("args", ["--mode", "permissive"]),
        ("env", {"MCP_MODE": "permissive"}),
        ("cwd", "/srv/other-mcp"),
        ("timeout", 60),
        ("keep_alive", False),
    ):
        definitions["server"] = copy.deepcopy(original)
        definitions["server"][field] = value
        assert tool_agent_turn._mcp_digest(("server",), {}) != baseline

    definitions["server"] = copy.deepcopy(original)
    definitions["server"]["auth"]["token_url"] = "https://auth-two.example/token"
    assert tool_agent_turn._mcp_digest(("server",), {}) != baseline

    definitions["server"] = copy.deepcopy(original)
    definitions["server"]["auth"]["client_id"] = "client-two"
    assert tool_agent_turn._mcp_digest(("server",), {}) != baseline


def test_tool_agent_graph_consumes_a_reconstructed_turn(monkeypatch) -> None:
    captured: dict[str, object] = {}
    loaded_tool = StructuredTool.from_function(
        lambda value: value, name="calculator", description="Adds values"
    )

    class FakeModel:
        model_id = "model-c"

        def bind_tools(self, tools: list[object]) -> FakeModel:
            captured["tools"] = tools
            return self

        async def ainvoke(self, messages: list[object], *, config: object) -> AIMessage:
            captured["messages"] = messages
            captured["config"] = config
            return AIMessage(content="42")

    monkeypatch.setattr(tool_agent_execution, "get_llm", lambda model_id=None: FakeModel())

    async def fake_load_adapter_tools(**_kwargs: object) -> list[object]:
        return [loaded_tool]

    monkeypatch.setattr(tool_agent_turn, "load_adapter_tools", fake_load_adapter_tools)

    async def run() -> dict:
        async with LocalAsyncSqliteSaver.from_conn_string(":memory:") as saver:
            runtime = SimpleNamespace(
                context={"model_id": "model-c", "mcp_server_keys": []},
                execution_info=SimpleNamespace(thread_id="thread-123", run_id="run-1"),
            )
            config = {"configurable": {"__pregel_checkpointer": saver}}
            setup_turn = await tool_agent_turn.prepare_tool_agent_turn(
                state={"messages": [HumanMessage(id="h2", content="19 + 23")]},
                parent_config=config,
                runtime=runtime,
                mode="mcp",
            )
            await tool_agent_turn.release_tool_agent_turn(config, setup_turn)
            return await tool_agent_execution.call_llm_node(
                {"messages": [HumanMessage(id="h2", content="19 + 23")]},
                config,
                runtime=runtime,  # type: ignore[arg-type]
            )

    result = asyncio.run(run())

    assert result["messages"][0].content == "42"
    assert captured["tools"] == [loaded_tool]
    assert "calculator" in captured["messages"][0].content
    assert captured["messages"][-1].content == "19 + 23"


def test_lease_heartbeat_cancels_external_work_after_lease_loss(monkeypatch) -> None:
    renewals = 0
    cancelled = asyncio.Event()

    async def fail_on_heartbeat(*_args: object) -> None:
        nonlocal renewals
        renewals += 1
        if renewals == 2:
            raise StaleLeaseError("lease taken over")

    async def slow_operation() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(tool_agent_turn, "renew_tool_agent_turn", fail_on_heartbeat)
    monkeypatch.setattr(tool_agent_turn, "_lease_heartbeat_interval_seconds", lambda: 0.001)

    async def run() -> None:
        with pytest.raises(StaleLeaseError, match="taken over"):
            await tool_agent_turn.run_with_lease_heartbeat(  # type: ignore[arg-type]
                {}, {}, slow_operation
            )

    asyncio.run(run())
    assert renewals == 2
    assert cancelled.is_set()


def test_lease_heartbeat_checks_fence_before_accepting_completed_work(monkeypatch) -> None:
    renewals = 0
    operation_started = False

    async def fail_final_fence(*_args: object) -> None:
        nonlocal renewals
        renewals += 1
        if renewals == 2:
            raise StaleLeaseError("lease taken over at completion")

    async def completed_operation() -> str:
        nonlocal operation_started
        operation_started = True
        return "must not be accepted"

    monkeypatch.setattr(tool_agent_turn, "renew_tool_agent_turn", fail_final_fence)

    async def run() -> None:
        with pytest.raises(StaleLeaseError, match="completion"):
            await tool_agent_turn.run_with_lease_heartbeat(  # type: ignore[arg-type]
                {}, {}, completed_operation
            )

    asyncio.run(run())
    assert operation_started
    assert renewals == 2


def test_lease_heartbeat_does_not_construct_operation_before_initial_fence(monkeypatch) -> None:
    constructed = False

    async def fail_initial_fence(*_args: object) -> None:
        raise StaleLeaseError("lease already taken over")

    def operation_factory():
        nonlocal constructed
        constructed = True

        async def operation() -> None:
            return None

        return operation()

    monkeypatch.setattr(tool_agent_turn, "renew_tool_agent_turn", fail_initial_fence)

    async def run() -> None:
        with pytest.raises(StaleLeaseError, match="already taken"):
            await tool_agent_turn.run_with_lease_heartbeat(  # type: ignore[arg-type]
                {}, {}, operation_factory
            )

    asyncio.run(run())
    assert not constructed


def test_lease_heartbeat_refreshes_runtime_and_cleans_up_on_caller_cancellation(
    monkeypatch,
) -> None:
    heartbeats = 0
    periodic_heartbeat = asyncio.Event()
    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()

    async def renew(*_args: object) -> None:
        return None

    async def slow_operation() -> None:
        operation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()

    runtime = SimpleNamespace()

    def heartbeat() -> None:
        nonlocal heartbeats
        heartbeats += 1
        if heartbeats >= 2:
            periodic_heartbeat.set()

    runtime.heartbeat = heartbeat
    monkeypatch.setattr(tool_agent_turn, "renew_tool_agent_turn", renew)
    monkeypatch.setattr(tool_agent_turn, "_lease_heartbeat_interval_seconds", lambda: 0.001)

    async def run() -> None:
        task = asyncio.create_task(
            tool_agent_turn.run_with_lease_heartbeat(  # type: ignore[arg-type]
                {}, {}, slow_operation, runtime=runtime
            )
        )
        await operation_started.wait()
        await asyncio.wait_for(periodic_heartbeat.wait(), timeout=0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert heartbeats >= 2
    assert operation_cancelled.is_set()


def test_reconstruction_preserves_failure_when_stale_release_cleanup_races(monkeypatch) -> None:
    lease = SimpleNamespace()
    recipe = SimpleNamespace(
        mode="mcp",
        collection_key=None,
        thread_id="thread",
        turn_id="turn",
        model_key="model",
        session_id=None,
        enable_tracing=False,
        mcp_server_keys=(),
        request_id=None,
        tool_round_limit=1,
    )

    class Store:
        async def claim(self, *_args: object) -> SimpleNamespace:
            return SimpleNamespace(lease=lease)

        async def release(self, _lease: object) -> None:
            raise StaleLeaseError("taken over during cleanup")

    async def load_recipe(**_kwargs: object) -> tuple[Store, object]:
        return Store(), recipe

    def fail_run_config(**_kwargs: object) -> object:
        raise RuntimeError("reconstruction failed")

    monkeypatch.setattr(tool_agent_turn, "_load_or_create_recipe", load_recipe)
    monkeypatch.setattr(tool_agent_turn, "build_run_config", fail_run_config)
    runtime = SimpleNamespace(context={}, execution_info=SimpleNamespace(thread_id="thread"))

    async def run() -> None:
        with pytest.raises(RuntimeError, match="reconstruction failed"):
            await tool_agent_turn.reconstruct_tool_agent_turn(
                state={"messages": [HumanMessage(id="turn", content="question")]},
                parent_config={},
                runtime=runtime,
                mode="mcp",
            )

    asyncio.run(run())


def test_mark_terminal_renews_before_marking_terminal(monkeypatch) -> None:
    calls: list[str] = []

    class Store:
        async def renew(self, _lease: object) -> None:
            calls.append("renew")

        async def mark_terminal(self, _lease: object, message_id: str) -> None:
            calls.append(f"mark:{message_id}")

    monkeypatch.setattr(tool_agent_turn, "_recipe_store", lambda _config: Store())

    asyncio.run(
        tool_agent_turn.mark_tool_agent_turn_terminal(
            {}, {"lease": object()}, "terminal-message"  # type: ignore[arg-type]
        )
    )
    assert calls == ["renew", "mark:terminal-message"]
