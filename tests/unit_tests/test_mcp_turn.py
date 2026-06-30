from __future__ import annotations

import asyncio
from types import SimpleNamespace
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

    async def fake_get_mcp_answer_result_async(*args: object, **kwargs: object):
        _ = args
        captured.update(kwargs)
        return SimpleNamespace(
            answer="answer",
            tools_used=["oracle_retrieval", "Calculator_calculate"],
            tool_invocations=[],
            state_messages=[],
        )

    monkeypatch.setattr(mcp_turn, "load_adapter_tools", fake_load_adapter_tools)
    monkeypatch.setattr(
        mcp_turn,
        "should_use_repeated_workflow",
        fake_should_use_repeated_workflow,
    )
    monkeypatch.setattr(
        mcp_turn, "get_mcp_answer_result_async", fake_get_mcp_answer_result_async
    )

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
            extra_tools=[_Tool("oracle_retrieval")],
            require_mcp_tool_call_when_referenced=True,
        )
    )

    assert captured["require_tool_call"] is True
    assert set(captured) == {
        "chat_history",
        "model_id",
        "tools",
        "run_config",
        "require_tool_call",
    }


def test_run_mcp_agent_turn_does_not_require_mcp_for_rag_only_question(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_load_adapter_tools(server_keys=None, run_config=None):
        _ = server_keys, run_config
        return [_Tool("Calculator_calculate")]

    async def fake_get_mcp_answer_result_async(*args: object, **kwargs: object):
        _ = args
        captured.update(kwargs)
        return SimpleNamespace(
            answer="answer",
            tools_used=["oracle_retrieval"],
            tool_invocations=[],
            state_messages=[],
        )

    monkeypatch.setattr(mcp_turn, "load_adapter_tools", fake_load_adapter_tools)
    monkeypatch.setattr(
        mcp_turn, "get_mcp_answer_result_async", fake_get_mcp_answer_result_async
    )

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
            extra_tools=[_Tool("oracle_retrieval")],
            require_mcp_tool_call_when_referenced=True,
        )
    )

    assert captured["require_tool_call"] is False
    assert set(captured) == {
        "chat_history",
        "model_id",
        "tools",
        "run_config",
        "require_tool_call",
    }
