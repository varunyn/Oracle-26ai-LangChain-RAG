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


class _FakeSequencedAgent:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []
        self._idx = 0

    async def ainvoke(self, inp: dict[str, object], *, config: object | None = None) -> dict[str, object]:
        self.calls.append({"input": inp, "config": config})
        idx = min(self._idx, len(self.outputs) - 1)
        self._idx += 1
        return self.outputs[idx]


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


def test_langchain_executor_closes_async_llm_client(monkeypatch) -> None:
    fake_agent = _FakeAgent({"messages": [AIMessage(content="done")]})

    class FakeLLM:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    fake_llm = FakeLLM()
    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: fake_llm)
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    import asyncio

    asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="finish",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="finish", description="finish")],
            run_config=None,
            require_tool_call=False,
        )
    )

    assert fake_llm.close_calls == 1


def test_langchain_executor_keeps_substantive_answer_when_no_tool_call(monkeypatch) -> None:
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

    assert answer == "No tools needed"
    assert tools_used == []
    assert invocations == []


def test_langchain_executor_enforces_require_tool_call_when_answer_empty(monkeypatch) -> None:
    fake_agent = _FakeAgent({"messages": [AIMessage(content="")]})

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


def test_build_middleware_uses_retry_tool_selector_and_tool_limit_controls() -> None:
    settings = SimpleNamespace(MCP_MAX_ROUNDS=2)
    middleware = mod._build_middleware(
        settings,
        [SimpleNamespace(name="calculator.add", description="add")],
    )
    names = [type(m).__name__ for m in middleware]
    assert names == [
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "LLMToolSelectorMiddleware",
        "ToolCallLimitMiddleware",
    ]


def test_build_middleware_keeps_oracle_retrieval_available_for_selector() -> None:
    settings = SimpleNamespace(MCP_MAX_ROUNDS=0)
    middleware = mod._build_middleware(
        settings,
        [
            SimpleNamespace(name="oracle_retrieval", description="retrieve"),
            SimpleNamespace(name="calculator.add", description="add"),
        ],
    )

    selector = middleware[2]
    assert type(selector).__name__ == "LLMToolSelectorMiddleware"
    assert selector.always_include == ["oracle_retrieval"]


def test_build_system_prompt_uses_mixed_prompt_when_oracle_retrieval_tool_present() -> None:
    prompt = mod._build_system_prompt(
        "How can I create applications?",
        [SimpleNamespace(name="oracle_retrieval", description="retrieve")],
        run_config=None,
    )
    assert "When document context was provided in the user message" in prompt


def test_build_system_prompt_prioritizes_explicit_workflows_generically() -> None:
    prompt = mod._build_system_prompt(
        "Process every invoice document and email the final summary.",
        [SimpleNamespace(name="oracle_retrieval", description="retrieve")],
        run_config={"configurable": {"mode": "mixed"}},
    )
    assert "Explicit multi-step workflows override the preference for fewer tool calls" in prompt
    assert "work unit" in prompt
    assert "user's requested completion criteria" in prompt
    assert "item/document/record" not in prompt


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


def test_executor_retries_when_literal_tool_call_text_is_emitted(monkeypatch) -> None:
    fake_agent = _FakeSequencedAgent(
        [
            {
                "messages": [
                    AIMessage(content='Now I will verify via oracle_retrieval(query="Summit payment terms").')
                ]
            },
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "oracle_retrieval",
                                "args": {"query": "Summit payment terms"},
                                "id": "call_1",
                            }
                        ],
                    ),
                    ToolMessage(content="Found net 30", tool_call_id="call_1", name="oracle_retrieval"),
                    AIMessage(content="Verified payment terms and completed processing."),
                ]
            },
        ]
    )

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=4))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    import asyncio

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="process invoices and verify payment terms",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="oracle_retrieval", description="retrieve")],
            run_config=None,
            require_tool_call=False,
        )
    )

    assert answer == "Verified payment terms and completed processing."
    assert tools_used == ["oracle_retrieval"]
    assert invocations == [
        {
            "tool_name": "oracle_retrieval",
            "args": {"query": "Summit payment terms"},
            "result": "Found net 30",
        }
    ]
    assert len(fake_agent.calls) == 2
