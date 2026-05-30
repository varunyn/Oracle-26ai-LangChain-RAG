from __future__ import annotations

import asyncio
from typing import cast

from langchain_core.runnables.config import RunnableConfig

from src.rag_agent.runtime import mcp_turn


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} tool"


def test_run_mcp_agent_turn_requires_tool_call_when_question_names_mcp_tool(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_load_adapter_tools(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [_Tool("Calculator_calculate")]

    async def fake_should_use_repeated_workflow(**kwargs: object) -> bool:
        _ = kwargs
        return False

    async def fake_get_mcp_answer_async(*args: object, **kwargs: object):
        _ = args
        captured.update(kwargs)
        return "answer", ["oracle_retrieval", "Calculator_calculate"], []

    monkeypatch.setattr(mcp_turn, "load_adapter_tools", fake_load_adapter_tools)
    monkeypatch.setattr(
        mcp_turn,
        "should_use_repeated_workflow",
        fake_should_use_repeated_workflow,
    )
    monkeypatch.setattr(mcp_turn, "get_mcp_answer_async", fake_get_mcp_answer_async)

    asyncio.run(
        mcp_turn.run_mcp_agent_turn(
            question="Check RAG first, then call Calculator_calculate for 125 * 48.",
            chat_history=[],
            resolved_model_id="fake-model",
            run_config=cast(RunnableConfig, {"configurable": {"mode": "mixed"}}),
            mode="mixed",
            mcp_server_keys=None,
            require_tool_call=False,
            repeated_workflow_enabled=False,
            workflow_checkpoint_path=None,
            tool_progress_callback=None,
            stop_after_tool_names={"oracle_retrieval"},
            extra_tools=[_Tool("oracle_retrieval")],
            require_mcp_tool_call_when_referenced=True,
        )
    )

    assert captured["require_tool_call"] is True
    assert captured["stop_after_tool_names"] == {"oracle_retrieval"}


def test_run_mcp_agent_turn_still_stops_after_retrieval_for_rag_only_question(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_load_adapter_tools(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [_Tool("Calculator_calculate")]

    async def fake_get_mcp_answer_async(*args: object, **kwargs: object):
        _ = args
        captured.update(kwargs)
        return "answer", ["oracle_retrieval"], []

    monkeypatch.setattr(mcp_turn, "load_adapter_tools", fake_load_adapter_tools)
    monkeypatch.setattr(mcp_turn, "get_mcp_answer_async", fake_get_mcp_answer_async)

    asyncio.run(
        mcp_turn.run_mcp_agent_turn(
            question="What are the payment terms for Northway Solutions?",
            chat_history=[],
            resolved_model_id="fake-model",
            run_config=cast(RunnableConfig, {"configurable": {"mode": "mixed"}}),
            mode="mixed",
            mcp_server_keys=None,
            require_tool_call=False,
            repeated_workflow_enabled=False,
            workflow_checkpoint_path=None,
            tool_progress_callback=None,
            stop_after_tool_names={"oracle_retrieval"},
            extra_tools=[_Tool("oracle_retrieval")],
            require_mcp_tool_call_when_referenced=True,
        )
    )

    assert captured["require_tool_call"] is False
    assert captured["stop_after_tool_names"] == {"oracle_retrieval"}
