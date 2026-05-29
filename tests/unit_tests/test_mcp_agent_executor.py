from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.rag_agent.infrastructure import mcp_agent_executor as mod


class _FakeAgent:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, inp: dict[str, object], *, config: object | None = None) -> dict[str, object]:
        self.calls.append({"input": inp, "config": config})
        return self.output


class _FakeStreamingAgent:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []

    async def astream_events(
        self,
        inp: dict[str, object],
        *,
        config: object | None = None,
        version: str,
    ):
        self.calls.append({"input": inp, "config": config, "version": version})
        for event in self.events:
            yield event

    async def ainvoke(
        self,
        inp: dict[str, object],
        *,
        config: object | None = None,
    ) -> dict[str, object]:
        raise AssertionError("streaming path should not call ainvoke")


class _FakeToolCall:
    def __init__(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, object],
        output: object,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.input = tool_input
        self.output = output
        self.error = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration


class _FakeToolCalls:
    def __init__(self, calls: list[_FakeToolCall]) -> None:
        self.calls = calls
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> _FakeToolCall:
        if self._idx >= len(self.calls):
            raise StopAsyncIteration
        call = self.calls[self._idx]
        self._idx += 1
        return call


class _FakeOutputDeltas:
    def __aiter__(self):
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration


class _FakeProjectedToolCall:
    def __init__(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, object],
        output: object,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.input = tool_input
        self.output = output
        self.output_deltas = _FakeOutputDeltas()
        self.error = None


class _FakeProjectionStream:
    def __init__(self, output: dict[str, object], tool_calls: list[_FakeToolCall]) -> None:
        self._output = output
        self.tool_calls = _FakeToolCalls(tool_calls)

    async def output(self) -> dict[str, object]:
        return self._output


class _FakeProjectionPropertyStream:
    def __init__(self, output: dict[str, object], tool_calls: list[_FakeProjectedToolCall]) -> None:
        self._output = output
        self.tool_calls = _FakeToolCalls(tool_calls)  # type: ignore[arg-type]
        self.output = self._resolve_output()

    async def _resolve_output(self) -> dict[str, object]:
        return self._output


class _FakeProjectionStreamingAgent:
    def __init__(self, stream: _FakeProjectionStream | _FakeProjectionPropertyStream) -> None:
        self.stream = stream
        self.calls: list[dict[str, Any]] = []

    async def astream_events(
        self,
        inp: dict[str, object],
        *,
        config: object | None = None,
        version: str,
    ) -> _FakeProjectionStream | _FakeProjectionPropertyStream:
        self.calls.append({"input": inp, "config": config, "version": version})
        return self.stream

    async def ainvoke(
        self,
        inp: dict[str, object],
        *,
        config: object | None = None,
    ) -> dict[str, object]:
        raise AssertionError("streaming path should not call ainvoke")


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


def test_langchain_executor_does_not_install_callback_progress_fallback(monkeypatch) -> None:
    fake_agent = _FakeAgent({"messages": [AIMessage(content="done")]})
    progress_events: list[dict[str, object]] = []

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="finish",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="finish", description="finish")],
            run_config=None,
            require_tool_call=False,
            tool_progress_callback=progress_events.append,
        )
    )

    assert answer == "done"
    assert tools_used == []
    assert invocations == []
    assert fake_agent.calls[0]["config"] == {}
    assert progress_events == []


def test_langchain_executor_streams_tool_progress_events(monkeypatch) -> None:
    tool_call_id = str(uuid.uuid4())
    fake_agent = _FakeStreamingAgent(
        [
            {
                "method": "tools",
                "params": {
                    "data": {
                        "event": "tool-started",
                        "tool_call_id": tool_call_id,
                        "tool_name": "Calculator_solve_equation",
                        "input": {"equation": "x^2 - 5x + 6 = 0"},
                    }
                },
            },
            {
                "method": "tools",
                "params": {
                    "data": {
                        "event": "tool-finished",
                        "tool_call_id": tool_call_id,
                        "tool_name": "Calculator_solve_equation",
                        "output": {"solutions": "[2, 3]"},
                    }
                },
            },
            {
                "method": "values",
                "params": {
                    "data": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "Calculator_solve_equation",
                                        "args": {"equation": "x^2 - 5x + 6 = 0"},
                                        "id": tool_call_id,
                                    }
                                ],
                            ),
                            ToolMessage(
                                content='{"solutions":"[2, 3]"}',
                                tool_call_id=tool_call_id,
                                name="Calculator_solve_equation",
                            ),
                            AIMessage(content="x = 2 and x = 3"),
                        ]
                    }
                },
            },
        ]
    )
    progress_events: list[dict[str, object]] = []

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    import asyncio

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="solve",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="Calculator_solve_equation", description="solve")],
            run_config=None,
            require_tool_call=False,
            tool_progress_callback=progress_events.append,
        )
    )

    assert fake_agent.calls[0]["version"] == "v3"
    assert answer == "x = 2 and x = 3"
    assert tools_used == ["Calculator_solve_equation"]
    assert invocations == [
        {
            "tool_name": "Calculator_solve_equation",
            "args": {"equation": "x^2 - 5x + 6 = 0"},
            "result": '{"solutions":"[2, 3]"}',
        }
    ]
    assert progress_events == [
        {
            "phase": "start",
            "tool_name": "Calculator_solve_equation",
            "args": {"equation": "x^2 - 5x + 6 = 0"},
            "tool_run_id": tool_call_id,
        },
        {
            "phase": "end",
            "tool_name": "Calculator_solve_equation",
            "result": '{"solutions": "[2, 3]"}',
            "tool_run_id": tool_call_id,
        },
    ]


def test_sanitize_agent_payload_removes_oci_unsupported_tool_call_content() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "Calling a tool."},
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "Calculator_solve_equation",
                "input": {"equation": "x^2 - 5x + 6 = 0"},
            },
        ],
        tool_calls=[
            {
                "name": "Calculator_solve_equation",
                "args": {"equation": "x^2 - 5x + 6 = 0"},
                "id": "call-1",
            }
        ],
    )

    payload = mod._sanitize_agent_payload_for_oci({"messages": [message]})

    sanitized = payload["messages"][0]
    assert isinstance(sanitized, AIMessage)
    assert sanitized.content == [{"type": "text", "text": "Calling a tool."}]
    assert sanitized.tool_calls == message.tool_calls


def test_sanitize_agent_payload_keeps_placeholder_for_tool_call_only_message() -> None:
    message = AIMessage(
        content=[
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "Calculator_solve_equation",
                "input": {"equation": "x^2 - 5x + 6 = 0"},
            }
        ],
        tool_calls=[
            {
                "name": "Calculator_solve_equation",
                "args": {"equation": "x^2 - 5x + 6 = 0"},
                "id": "call-1",
            }
        ],
    )

    payload = mod._sanitize_agent_payload_for_oci({"messages": [message]})

    sanitized = payload["messages"][0]
    assert isinstance(sanitized, AIMessage)
    assert sanitized.content == [{"type": "text", "text": "."}]
    assert sanitized.tool_calls == message.tool_calls


def test_oci_tool_call_content_middleware_sanitizes_internal_model_requests() -> None:
    tool_message = AIMessage(
        content=[
            {"type": "tool_call", "id": "call_1", "name": "Calculator_solve_equation"},
        ],
        tool_calls=[
            {
                "name": "Calculator_solve_equation",
                "args": {"equation": "x^2 - 5x + 6 = 0"},
                "id": "call_1",
            }
        ],
    )
    request = ModelRequest(model=object(), messages=[tool_message])
    seen_messages: list[object] = []

    def handler(sanitized_request: ModelRequest) -> ModelResponse:
        seen_messages.extend(sanitized_request.messages)
        return ModelResponse(result=[AIMessage(content="done")])

    response = mod.OCIToolCallContentMiddleware().wrap_model_call(request, handler)

    assert isinstance(response, ModelResponse)
    sanitized = seen_messages[0]
    assert isinstance(sanitized, AIMessage)
    assert sanitized.content == [{"type": "text", "text": "."}]
    assert sanitized.tool_calls == tool_message.tool_calls


def test_model_timeout_propagates_after_successful_tool_result() -> None:
    request = ModelRequest(
        model=object(),
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Calculator_linear_regression",
                        "args": {"data": [[1, 2], [2, 3.5]]},
                        "id": "call_1",
                    }
                ],
            ),
            ToolMessage(
                content='{"slope":1.54,"intercept":0.44}',
                tool_call_id="call_1",
                name="Calculator_linear_regression",
            ),
        ],
    )

    async def handler(_request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="too late")])

    try:
        asyncio.run(
            asyncio.wait_for(handler(request), timeout=0.01)
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected TimeoutError")


def test_serialize_tool_output_prefers_tool_message_structured_content() -> None:
    output = ToolMessage(
        content=[
            {
                "type": "text",
                "text": '{"slope":1.54,"intercept":0.4400000000000004}',
            }
        ],
        tool_call_id="call_1",
        name="Calculator_linear_regression",
        artifact={"structured_content": {"slope": 1.54, "intercept": 0.4400000000000004}},
    )

    assert mod._serialize_tool_output(output) == (
        '{"slope": 1.54, "intercept": 0.4400000000000004}'
    )


def test_langchain_executor_uses_tool_call_projection_for_live_progress(monkeypatch) -> None:
    tool_call_id = str(uuid.uuid4())
    fake_agent = _FakeProjectionStreamingAgent(
        _FakeProjectionStream(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "Calculator_solve_equation",
                                "args": {"equation": "x^2 - 5x + 6 = 0"},
                                "id": tool_call_id,
                            }
                        ],
                    ),
                    ToolMessage(
                        content='{"solutions":"[2, 3]"}',
                        tool_call_id=tool_call_id,
                        name="Calculator_solve_equation",
                    ),
                    AIMessage(content="x = 2 and x = 3"),
                ]
            },
            [
                _FakeToolCall(
                    tool_call_id=tool_call_id,
                    tool_name="Calculator_solve_equation",
                    tool_input={"equation": "x^2 - 5x + 6 = 0"},
                    output={"solutions": "[2, 3]"},
                )
            ],
        )
    )
    progress_events: list[dict[str, object]] = []

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    import asyncio

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="solve",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="Calculator_solve_equation", description="solve")],
            run_config=None,
            require_tool_call=False,
            tool_progress_callback=progress_events.append,
        )
    )

    assert fake_agent.calls[0]["version"] == "v3"
    assert answer == "x = 2 and x = 3"
    assert tools_used == ["Calculator_solve_equation"]
    assert invocations == [
        {
            "tool_name": "Calculator_solve_equation",
            "args": {"equation": "x^2 - 5x + 6 = 0"},
            "result": '{"solutions":"[2, 3]"}',
        }
    ]
    assert progress_events == [
        {
            "phase": "start",
            "tool_name": "Calculator_solve_equation",
            "args": {"equation": "x^2 - 5x + 6 = 0"},
            "tool_run_id": tool_call_id,
        },
        {
            "phase": "end",
            "tool_name": "Calculator_solve_equation",
            "result": '{"solutions": "[2, 3]"}',
            "tool_run_id": tool_call_id,
        },
    ]


def test_langchain_executor_uses_async_output_property_projection(monkeypatch) -> None:
    tool_call_id = str(uuid.uuid4())
    fake_agent = _FakeProjectionStreamingAgent(
        _FakeProjectionPropertyStream(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "Calculator_linear_regression",
                                "args": {"data": [[1, 2], [2, 4]]},
                                "id": tool_call_id,
                            }
                        ],
                    ),
                    ToolMessage(
                        content='{"slope":2.0,"intercept":0.0}',
                        tool_call_id=tool_call_id,
                        name="Calculator_linear_regression",
                    ),
                    AIMessage(content="y = 2x"),
                ]
            },
            [
                _FakeProjectedToolCall(
                    tool_call_id=tool_call_id,
                    tool_name="Calculator_linear_regression",
                    tool_input={"data": [[1, 2], [2, 4]]},
                    output={"slope": 2.0, "intercept": 0.0},
                )
            ],
        )
    )
    progress_events: list[dict[str, object]] = []

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="fit line",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="Calculator_linear_regression", description="fit")],
            run_config=None,
            require_tool_call=False,
            tool_progress_callback=progress_events.append,
        )
    )

    assert fake_agent.calls[0]["version"] == "v3"
    assert answer == "y = 2x"
    assert tools_used == ["Calculator_linear_regression"]
    assert invocations == [
        {
            "tool_name": "Calculator_linear_regression",
            "args": {"data": [[1, 2], [2, 4]]},
            "result": '{"slope":2.0,"intercept":0.0}',
        }
    ]
    assert progress_events == [
        {
            "phase": "start",
            "tool_name": "Calculator_linear_regression",
            "args": {"data": [[1, 2], [2, 4]]},
            "tool_run_id": tool_call_id,
        },
        {
            "phase": "end",
            "tool_name": "Calculator_linear_regression",
            "result": '{"slope": 2.0, "intercept": 0.0}',
            "tool_run_id": tool_call_id,
        },
    ]


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


def test_langchain_executor_rejects_direct_answer_when_tool_call_required(monkeypatch) -> None:
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
    assert len(fake_agent.calls) == 2


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


def test_build_middleware_defaults_to_single_agent_retry_and_limit_controls() -> None:
    settings = SimpleNamespace(MCP_MAX_ROUNDS=2)
    middleware = mod._build_middleware(
        settings,
        [SimpleNamespace(name="calculator.add", description="add")],
    )
    names = [type(m).__name__ for m in middleware]
    assert names == [
        "OCIToolCallContentMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "ToolCallLimitMiddleware",
    ]
    assert middleware[-1].run_limit == 2


def test_build_middleware_can_enable_selector_for_large_catalogs() -> None:
    settings = SimpleNamespace(MCP_MAX_ROUNDS=0)
    middleware = mod._build_middleware(
        settings,
        [
            SimpleNamespace(name="oracle_retrieval", description="retrieve"),
            SimpleNamespace(name="calculator.add", description="add"),
        ],
        use_tool_selector=True,
        use_tool_retry=False,
    )

    selector = middleware[2]
    assert type(selector).__name__ == "LLMToolSelectorMiddleware"
    assert selector.always_include == ["oracle_retrieval"]


def test_build_middleware_can_disable_tool_retry() -> None:
    settings = SimpleNamespace(MCP_MAX_ROUNDS=0)
    middleware = mod._build_middleware(
        settings,
        [SimpleNamespace(name="calculator.add", description="add")],
        use_tool_selector=False,
        use_tool_retry=False,
    )

    names = [type(m).__name__ for m in middleware]
    assert names == [
        "OCIToolCallContentMiddleware",
        "ModelRetryMiddleware",
    ]


def test_langchain_executor_retries_tool_error_in_same_agent_harness(
    monkeypatch,
) -> None:
    fake_agent = _FakeSequencedAgent(
        [
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "Calculator_calculate",
                                "args": {"expression": "x**2 - 5*x + 6"},
                                "id": "bad-tool",
                            }
                        ],
                    ),
                    ToolMessage(
                        content='{"error":"name x is not defined"}',
                        tool_call_id="bad-tool",
                        name="Calculator_calculate",
                    ),
                ]
            },
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "Calculator_solve_equation",
                                "args": {"equation": "x**2 - 5*x + 6 = 0"},
                                "id": "good-tool",
                            }
                        ],
                    ),
                    ToolMessage(
                        content='{"solutions":["2","3"]}',
                        tool_call_id="good-tool",
                        name="Calculator_solve_equation",
                    ),
                    AIMessage(content="x = 2 and x = 3"),
                ]
            },
        ]
    )
    created_agents: list[dict[str, Any]] = []

    def fake_create_agent(**kwargs: Any) -> _FakeAgent:
        created_agents.append(kwargs)
        return fake_agent

    monkeypatch.setattr(
        "api.settings.get_settings",
        lambda: SimpleNamespace(MCP_MAX_ROUNDS=2, MCP_USE_LLM_TOOL_SELECTOR=False),
    )
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", fake_create_agent)

    import asyncio

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="Solve the following equation: x^2 - 5x + 6 = 0 using tools",
            chat_history=None,
            model_id=None,
            tools=[
                SimpleNamespace(name="Calculator_calculate", description="evaluate expression"),
                SimpleNamespace(name="Calculator_solve_equation", description="solve equation"),
            ],
            run_config=None,
            require_tool_call=True,
        )
    )

    assert answer == "x = 2 and x = 3"
    assert tools_used == ["Calculator_solve_equation"]
    assert invocations == [
        {
            "tool_name": "Calculator_solve_equation",
            "args": {"equation": "x**2 - 5*x + 6 = 0"},
            "result": '{"solutions":["2","3"]}',
        }
    ]
    assert len(created_agents) == 1
    middleware_names = [type(m).__name__ for m in created_agents[0]["middleware"]]
    assert "LLMToolSelectorMiddleware" not in middleware_names
    assert "ToolRetryMiddleware" in middleware_names
    assert created_agents[0]["middleware"][-1].run_limit == 2
    assert "transformers" not in created_agents[0]
    retry_messages = fake_agent.calls[1]["input"]["messages"]
    assert len(retry_messages) == 2
    assert isinstance(retry_messages[0], HumanMessage)
    assert isinstance(retry_messages[1], HumanMessage)
    assert "Calculator_calculate" in retry_messages[1].content
    assert "bad-tool" not in retry_messages[1].content
    assert not any(isinstance(message, (AIMessage, ToolMessage)) for message in retry_messages)


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


def test_executor_does_not_retry_literal_tool_text_without_tool_requirement(
    monkeypatch,
) -> None:
    literal_answer = '<|python_start|>Calculator_calculate(expression="12/16")<|python_end|>'
    fake_agent = _FakeSequencedAgent([{"messages": [AIMessage(content=literal_answer)]}])

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=4))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="Calculate 12/16 using tools",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="Calculator_calculate", description="calculate")],
            run_config=None,
            require_tool_call=False,
        )
    )

    assert answer == literal_answer
    assert tools_used == []
    assert invocations == []
    assert len(fake_agent.calls) == 1


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
