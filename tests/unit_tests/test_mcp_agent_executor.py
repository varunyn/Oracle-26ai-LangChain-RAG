from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from src.rag_agent.infrastructure import mcp_agent_executor as mod


class _FakeAgent:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, inp: dict[str, object], *, config: object | None = None) -> dict[str, object]:
        self.calls.append({"input": inp, "config": config})
        return self.output


def test_langchain_executor_returns_final_answer_and_tools(monkeypatch) -> None:
    fake_agent = _FakeAgent(
        {
            "messages": [
                AIMessage(
                    content="thinking",
                    tool_calls=[
                        {"name": "calculator.add", "args": {"expression": "2+2"}, "id": "t1"},
                    ],
                ),
                ToolMessage(content="4", tool_call_id="t1", name="calculator.add"),
                AIMessage(content="4"),
            ]
        }
    )

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    result = mod.get_mcp_answer_with_langchain_agent_async(
        question="2+2?",
        chat_history=None,
        model_id=None,
        tools=[SimpleNamespace(name="calculator.add", description="add")],
        run_config=None,
        require_tool_call=False,
    )

    import asyncio

    answer, tools_used, invocations = asyncio.run(result)
    assert answer == "4"
    assert tools_used == ["calculator.add"]
    assert invocations == [
        {"tool_name": "calculator.add", "args": {"expression": "2+2"}, "result": "4"},
    ]
    assert len(fake_agent.calls) == 1


def test_langchain_executor_enforces_require_tool_call(monkeypatch) -> None:
    fake_agent = _FakeAgent({"messages": [AIMessage(content="No tools needed")]})

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    import asyncio

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="2+2?",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="calculator.add", description="add")],
            run_config=None,
            require_tool_call=True,
        )
    )

    assert answer == "MCP tool call required but none was produced after retry. Please try again."
    assert tools_used == []
    assert invocations == []


def test_build_middleware_skips_llm_selector_for_oci_models() -> None:
    class FakeOCIModel:
        __module__ = "langchain_oci.chat_models.oci_generative_ai"

    settings = SimpleNamespace(
        MCP_TOOL_SELECTION_MAX_TOOLS=5,
        MCP_TOOL_SELECTION_ALWAYS_INCLUDE=[],
        MCP_MAX_ROUNDS=2,
    )
    middleware = mod._build_middleware(settings, FakeOCIModel())
    names = [type(m).__name__ for m in middleware]
    assert "LLMToolSelectorMiddleware" not in names


def test_build_system_prompt_uses_mixed_prompt_when_oracle_retrieval_tool_present() -> None:
    prompt = mod._build_system_prompt(
        "How can I create applications?",
        [SimpleNamespace(name="oracle_retrieval", description="retrieve")],
        run_config=None,
    )
    assert "When document context was provided in the user message" in prompt


def test_langchain_executor_normalizes_missing_tool_call_ids(monkeypatch) -> None:
    fake_agent = _FakeAgent(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "oci.run", "args": {"command": "os ns get"}, "id": ""}],
                    additional_kwargs={
                        "tool_calls": [
                            {"id": "", "function": {"name": "oci.run", "arguments": {"command": "os ns get"}}}
                        ]
                    },
                    response_metadata={
                        "tool_calls": [
                            {"id": "", "function": {"name": "oci.run", "arguments": {"command": "os ns get"}}}
                        ]
                    },
                ),
                ToolMessage(content="ok", tool_call_id="abc", name="oci.run"),
                AIMessage(content="done"),
            ]
        }
    )

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="a1b2c3d4e5f67890"))

    import asyncio

    _ = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="namespace?",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="oci.run", description="run")],
            run_config=None,
            require_tool_call=False,
        )
    )

    first = fake_agent.output["messages"][0]
    assert isinstance(first, AIMessage)
    assert first.tool_calls[0]["id"] == "call_0_a1b2c3d4e5f6"
    assert first.additional_kwargs["tool_calls"][0]["id"] == "call_0_a1b2c3d4e5f6"
    assert first.response_metadata["tool_calls"][0]["id"] == "call_0_a1b2c3d4e5f6"


def test_extract_answer_and_tools_supports_mapping_messages() -> None:
    state = {
        "messages": [
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "calculator_calculate",
                        "args": {"expression": "200 * 0.25"},
                        "id": "call_1",
                    }
                ],
            },
            {
                "type": "tool",
                "name": "calculator_calculate",
                "content": [{"type": "text", "text": {"result": 50}}],
            },
            {"type": "ai", "content": "25% of 200 is 50."},
        ]
    }

    answer, tools_used = mod._extract_answer_and_tools(state)
    assert answer == "25% of 200 is 50."
    assert tools_used == ["calculator_calculate"]


def test_extract_answer_and_tools_reads_additional_kwargs_tool_calls_from_ai_message() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "",
                            "function": {
                                "name": "calculator_calculate",
                                "arguments": {"expression": "200*0.25"},
                            },
                            "type": "function",
                        }
                    ]
                },
                response_metadata={},
            ),
            AIMessage(content="25% of 200 is 50."),
        ]
    }

    answer, tools_used = mod._extract_answer_and_tools(state)
    assert answer == "25% of 200 is 50."
    assert tools_used == ["calculator_calculate"]


def test_clean_leaked_tool_syntax_for_calculator_expression() -> None:
    leaked = '<|python_start|>calculator_calculate(expression="12/16")<|python_end|>'
    cleaned = mod._clean_leaked_tool_syntax(leaked, [])
    assert cleaned == "3/4"


def test_extract_tool_invocations_pairs_ai_tool_calls_with_tool_messages() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "oci-mcp-server_run_oci_command",
                        "args": {"command": ["logs", "api"]},
                        "id": "call_1",
                    },
                ],
            ),
            ToolMessage(
                content="log line one\nlog line two",
                tool_call_id="call_1",
                name="oci-mcp-server_run_oci_command",
            ),
            AIMessage(content="Here is what the logs show."),
        ]
    }
    invocations = mod._extract_tool_invocations(state)
    assert invocations == [
        {
            "tool_name": "oci-mcp-server_run_oci_command",
            "args": {"command": ["logs", "api"]},
            "result": "log line one\nlog line two",
        },
    ]
