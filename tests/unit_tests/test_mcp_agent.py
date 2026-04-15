from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.rag_agent.infrastructure import mcp_agent


class _Tool:
    name = "calculator.add"
    description = "Add values"


async def _run_sync_wrapper() -> tuple[str, list[str], list[dict[str, object]]]:
    return mcp_agent.get_mcp_answer("2+2", tools=[_Tool()])


def test_get_mcp_answer_disabled_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_agent,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=False),
    )

    answer, tools_used, invocations = mcp_agent.get_mcp_answer("2+2")

    assert answer == ""
    assert tools_used == []
    assert invocations == []


def test_get_mcp_answer_loads_tools_and_uses_executor(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_agent,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True),
    )

    async def _tools_loader(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [_Tool()]

    async def _executor(**kwargs):
        assert kwargs["question"] == "2+2"
        assert len(kwargs["tools"]) == 1
        return "4", ["calculator.add"], []

    monkeypatch.setattr(mcp_agent, "get_mcp_tools_async", _tools_loader)
    monkeypatch.setattr(mcp_agent, "get_mcp_answer_with_langchain_agent_async", _executor)

    answer, tools_used, invocations = mcp_agent.get_mcp_answer("2+2")

    assert answer == "4"
    assert tools_used == ["calculator.add"]
    assert invocations == []


def test_get_mcp_answer_runs_from_async_context(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_agent,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True),
    )

    async def _executor(**kwargs):
        _ = kwargs
        return "ok", ["calculator.add"], []

    monkeypatch.setattr(mcp_agent, "get_mcp_answer_with_langchain_agent_async", _executor)

    answer, tools_used, invocations = asyncio.run(_run_sync_wrapper())

    assert answer == "ok"
    assert tools_used == ["calculator.add"]
    assert invocations == []
