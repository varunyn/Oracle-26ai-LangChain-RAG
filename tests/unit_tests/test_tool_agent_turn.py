from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs import tool_agent_turn
from src.rag_agent.graphs.nodes import mixed


def test_prepare_tool_agent_turn_builds_one_execution_value_from_chat_context(
    monkeypatch,
) -> None:
    loaded_tool = SimpleNamespace(name="calculator", description="Adds values")

    async def fake_load_adapter_tools(**_kwargs: object) -> list[object]:
        return [loaded_tool]

    monkeypatch.setattr(
        tool_agent_turn,
        "load_adapter_tools",
        fake_load_adapter_tools,
    )
    runtime = SimpleNamespace(
        context={
            "model_id": "model-c",
            "session_id": "session-mcp",
            "enable_tracing": True,
            "mcp_server_keys": ["calculator"],
        },
        execution_info=SimpleNamespace(thread_id="thread-123"),
    )

    turn = asyncio.run(
        tool_agent_turn.prepare_tool_agent_turn(
            state={
                "messages": [
                    HumanMessage(content="Remember this context"),
                    AIMessage(content="I will"),
                    HumanMessage(content="19 + 23"),
                ]
            },
            parent_config={"callbacks": ["outer-callback"]},
            runtime=runtime,
            mode="mcp",
        )
    )

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


def test_tool_agent_graph_consumes_the_prepared_turn(monkeypatch) -> None:
    captured: dict[str, object] = {}
    loaded_tool = SimpleNamespace(name="calculator", description="Adds values")

    class FakeModel:
        model_id = "model-c"

        def bind_tools(self, tools: list[object]) -> FakeModel:
            captured["tools"] = tools
            return self

        async def ainvoke(self, messages: list[object], *, config: object) -> AIMessage:
            captured["messages"] = messages
            captured["config"] = config
            return AIMessage(content="42")

    monkeypatch.setattr(mixed, "get_llm", lambda model_id=None: FakeModel())
    turn = {
        "chat_history": [HumanMessage(content="Earlier context")],
        "model_id": "model-c",
        "question": "19 + 23",
        "run_config": {"configurable": {"mode": "mcp"}},
        "system_prompt": "Use calculator.",
        "tools": [loaded_tool],
    }
    runtime = SimpleNamespace(context={"tool_agent_turn": turn})

    result = asyncio.run(
        mixed.call_llm_node(
            {"messages": []},
            {"callbacks": ["outer-callback"]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assert result["messages"][0].content == "42"
    assert captured["tools"] == [loaded_tool]
    assert [message.content for message in captured["messages"]] == [
        "Use calculator.",
        "Earlier context",
        "19 + 23",
    ]
