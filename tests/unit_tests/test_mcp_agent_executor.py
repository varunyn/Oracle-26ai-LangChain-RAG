from __future__ import annotations

import asyncio
import inspect
import uuid
from types import SimpleNamespace
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, ToolMessage

from src.rag_agent.infrastructure import mcp_agent_executor as mod


class _FakeAgent:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self, inp: dict[str, object], *, config: object | None = None
    ) -> dict[str, object]:
        self.calls.append({"input": inp, "config": config})
        return self.output


class _FakeRaisingAgent:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self, inp: dict[str, object], *, config: object | None = None
    ) -> dict[str, object]:
        self.calls.append({"input": inp, "config": config})
        raise self.exc


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


class _FakeTextProjection:
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._idx >= len(self.deltas):
            raise StopAsyncIteration
        delta = self.deltas[self._idx]
        self._idx += 1
        return delta


class _FakeMessageProjection:
    def __init__(self, deltas: list[str]) -> None:
        self.text = _FakeTextProjection(deltas)


class _FakeMessages:
    def __init__(self, messages: list[_FakeMessageProjection]) -> None:
        self.messages = messages
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> _FakeMessageProjection:
        if self._idx >= len(self.messages):
            raise StopAsyncIteration
        message = self.messages[self._idx]
        self._idx += 1
        return message


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
    def __init__(
        self,
        output: dict[str, object],
        tool_calls: list[_FakeToolCall],
        messages: list[_FakeMessageProjection] | None = None,
        raw_events: list[dict[str, object]] | None = None,
    ) -> None:
        self._output = output
        self.tool_calls = _FakeToolCalls(tool_calls)
        self.messages = _FakeMessages(messages or [])
        self.raw_events = raw_events or []
        self._raw_idx = 0

    async def output(self) -> dict[str, object]:
        return self._output

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict[str, object]:
        if self._raw_idx >= len(self.raw_events):
            raise StopAsyncIteration
        event = self.raw_events[self._raw_idx]
        self._raw_idx += 1
        return event


class _FakeProjectionPropertyStream:
    def __init__(
        self,
        output: dict[str, object],
        tool_calls: list[_FakeProjectedToolCall],
        messages: list[_FakeMessageProjection] | None = None,
        raw_events: list[dict[str, object]] | None = None,
    ) -> None:
        self._output = output
        self.tool_calls = _FakeToolCalls(tool_calls)  # type: ignore[arg-type]
        self.messages = _FakeMessages(messages or [])
        self.raw_events = raw_events or []
        self._raw_idx = 0
        self.output = self._resolve_output()

    async def _resolve_output(self) -> dict[str, object]:
        return self._output

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict[str, object]:
        if self._raw_idx >= len(self.raw_events):
            raise StopAsyncIteration
        event = self.raw_events[self._raw_idx]
        self._raw_idx += 1
        return event


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

    async def ainvoke(
        self, inp: dict[str, object], *, config: object | None = None
    ) -> dict[str, object]:
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


def test_langchain_executor_requires_typed_event_stream_projections(monkeypatch) -> None:
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

    try:
        asyncio.run(
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
    except RuntimeError as exc:
        assert "typed stream projections" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for stream without typed projections")

    assert progress_events == []


def test_langchain_executor_streams_message_projection_deltas(monkeypatch) -> None:
    tool_call_id = str(uuid.uuid4())
    fake_agent = _FakeProjectionStreamingAgent(
        _FakeProjectionStream(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "oracle_retrieval",
                                "args": {"query": "Northway Solutions payment terms"},
                                "id": tool_call_id,
                            }
                        ],
                    ),
                    ToolMessage(
                        content="Payment terms are net 30.",
                        tool_call_id=tool_call_id,
                        name="oracle_retrieval",
                    ),
                    AIMessage(content="Payment terms are net 30."),
                ]
            },
            [
                _FakeToolCall(
                    tool_call_id=tool_call_id,
                    tool_name="oracle_retrieval",
                    tool_input={"query": "Northway Solutions payment terms"},
                    output="Payment terms are net 30.",
                )
            ],
            raw_events=[
                {
                    "method": "messages",
                    "params": {
                        "data": (
                            {
                                "event": "content-block-delta",
                                "delta": {"type": "block-delta", "text": "ignore"},
                            },
                            {"langgraph_node": "model"},
                        )
                    },
                },
                {
                    "method": "messages",
                    "params": {
                        "data": (
                            {
                                "event": "content-block-delta",
                                "delta": {"type": "text-delta", "text": "Payment "},
                            },
                            {"langgraph_node": "model"},
                        )
                    },
                },
                {
                    "method": "messages",
                    "params": {
                        "data": (
                            {
                                "event": "content-block-delta",
                                "delta": {"type": "text-delta", "text": "terms "},
                            },
                            {"langgraph_node": "model"},
                        )
                    },
                },
                {
                    "method": "messages",
                    "params": {
                        "data": (
                            {
                                "event": "content-block-delta",
                                "delta": {"type": "text-delta", "text": "are "},
                            },
                            {"langgraph_node": "model"},
                        )
                    },
                },
                {
                    "method": "messages",
                    "params": {
                        "data": (
                            {
                                "event": "content-block-delta",
                                "delta": {"type": "text-delta", "text": "net 30."},
                            },
                            {"langgraph_node": "model"},
                        )
                    },
                },
            ],
        )
    )
    progress_events: list[dict[str, object]] = []
    answer_deltas: list[str] = []
    fake_llm = SimpleNamespace(is_stream=False)

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: fake_llm)
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="payment terms",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="oracle_retrieval", description="retrieve")],
            run_config=None,
            require_tool_call=False,
            tool_progress_callback=progress_events.append,
            answer_delta_callback=answer_deltas.append,
        )
    )

    assert answer == "Payment terms are net 30."
    assert tools_used == ["oracle_retrieval"]
    assert invocations == [
        {
            "tool_name": "oracle_retrieval",
            "args": {"query": "Northway Solutions payment terms"},
            "result": "Payment terms are net 30.",
        }
    ]
    assert [event["phase"] for event in progress_events] == ["start", "end"]
    assert answer_deltas == ["Payment ", "terms ", "are ", "net 30."]
    assert fake_llm.is_stream is True


def test_langchain_executor_can_stop_after_requested_tool_result(monkeypatch) -> None:
    tool_call_id = str(uuid.uuid4())
    final_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "oracle_retrieval",
                        "args": {"query": "Northway Solutions payment terms"},
                        "id": tool_call_id,
                    }
                ],
            ),
            ToolMessage(
                content="Payment terms are net 30.",
                tool_call_id=tool_call_id,
                name="oracle_retrieval",
            ),
            AIMessage(content="This final model answer should not be consumed."),
        ]
    }
    post_tool_state = {"messages": final_state["messages"][:2]}
    stream = _FakeProjectionStream(
        final_state,
        [],
        raw_events=[
            {"method": "values", "params": {"data": {"messages": final_state["messages"][:1]}}},
            {"method": "values", "params": {"data": post_tool_state}},
            {"method": "values", "params": {"data": final_state}},
        ],
    )
    fake_agent = _FakeProjectionStreamingAgent(stream)
    progress_events: list[dict[str, object]] = []

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="payment terms",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="oracle_retrieval", description="retrieve")],
            run_config=None,
            require_tool_call=False,
            tool_progress_callback=progress_events.append,
            stop_after_tool_names={"oracle_retrieval"},
        )
    )

    assert answer == ""
    assert tools_used == ["oracle_retrieval"]
    assert invocations == [
        {
            "tool_name": "oracle_retrieval",
            "args": {"query": "Northway Solutions payment terms"},
            "result": "Payment terms are net 30.",
        }
    ]
    assert stream._raw_idx == 2
    assert [event["phase"] for event in progress_events] == ["start", "end"]


def test_langchain_executor_does_not_stop_when_other_tool_was_requested(
    monkeypatch,
) -> None:
    retrieval_id = str(uuid.uuid4())
    calculator_id = str(uuid.uuid4())
    final_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "oracle_retrieval",
                        "args": {"query": "Northway Solutions payment terms"},
                        "id": retrieval_id,
                    },
                    {
                        "name": "Calculator_calculate",
                        "args": {"expression": "125 * 48"},
                        "id": calculator_id,
                    },
                ],
            ),
            ToolMessage(
                content="Payment terms are net 30.",
                tool_call_id=retrieval_id,
                name="oracle_retrieval",
            ),
            ToolMessage(
                content="6000",
                tool_call_id=calculator_id,
                name="Calculator_calculate",
            ),
            AIMessage(content="Payment terms are net 30 and 125 * 48 is 6000."),
        ]
    }
    post_retrieval_state = {"messages": final_state["messages"][:2]}
    stream = _FakeProjectionStream(
        final_state,
        [],
        raw_events=[
            {"method": "values", "params": {"data": {"messages": final_state["messages"][:1]}}},
            {"method": "values", "params": {"data": post_retrieval_state}},
            {"method": "values", "params": {"data": final_state}},
        ],
    )
    fake_agent = _FakeProjectionStreamingAgent(stream)

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="payment terms and calculate 125 * 48",
            chat_history=None,
            model_id=None,
            tools=[
                SimpleNamespace(name="oracle_retrieval", description="retrieve"),
                SimpleNamespace(name="Calculator_calculate", description="calculate"),
            ],
            run_config=None,
            require_tool_call=False,
            stop_after_tool_names={"oracle_retrieval"},
        )
    )

    assert answer == "Payment terms are net 30 and 125 * 48 is 6000."
    assert tools_used == ["oracle_retrieval", "Calculator_calculate"]
    assert [item["tool_name"] for item in invocations] == [
        "oracle_retrieval",
        "Calculator_calculate",
    ]
    assert stream._raw_idx == 3


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
        asyncio.run(asyncio.wait_for(handler(request), timeout=0.01))
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


def test_build_middleware_always_enables_tool_selector_and_limit_controls() -> None:
    settings = SimpleNamespace(MCP_MAX_ROUNDS=2)
    middleware = mod._build_middleware(
        settings,
        [
            SimpleNamespace(name="calculator.add", description="add"),
        ],
    )
    names = [type(m).__name__ for m in middleware]
    assert names == [
        "OCIToolCallContentMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "LLMToolSelectorMiddleware",
        "ToolCallLimitMiddleware",
    ]
    selector = middleware[3]
    assert selector.always_include == []
    assert selector.max_tools is None
    assert "select all tools that may be needed" in selector.system_prompt.lower()
    assert "oracle_retrieval" in selector.system_prompt
    assert "select tools for every independent evidence or action need" in selector.system_prompt
    assert middleware[-1].run_limit == 2
    assert middleware[-1].exit_behavior == "error"


def test_build_middleware_keeps_full_toolbox_when_retrieval_tool_is_present() -> None:
    settings = SimpleNamespace(MCP_MAX_ROUNDS=4)
    middleware = mod._build_middleware(
        settings,
        [
            SimpleNamespace(name="oracle_retrieval", description="retrieve"),
            SimpleNamespace(name="Calculator_calculate", description="calculate"),
            SimpleNamespace(name="Calculator_integrate", description="integrate"),
        ],
    )

    names = [type(m).__name__ for m in middleware]
    assert "LLMToolSelectorMiddleware" not in names
    assert names == [
        "OCIToolCallContentMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "ToolCallLimitMiddleware",
    ]
    assert middleware[-1].run_limit == 4


def test_build_middleware_has_no_tool_retry_opt_out() -> None:
    signature = inspect.signature(mod._build_middleware)

    assert "use_tool_retry" not in signature.parameters


def test_langchain_executor_does_not_rerun_agent_after_tool_error(
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
        lambda: SimpleNamespace(MCP_MAX_ROUNDS=2),
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

    assert answer == ""
    assert tools_used == ["Calculator_calculate"]
    assert invocations == [
        {
            "tool_name": "Calculator_calculate",
            "args": {"expression": "x**2 - 5*x + 6"},
            "result": '{"error":"name x is not defined"}',
        }
    ]
    assert len(created_agents) == 1
    middleware_names = [type(m).__name__ for m in created_agents[0]["middleware"]]
    assert "LLMToolSelectorMiddleware" in middleware_names
    assert "ToolRetryMiddleware" in middleware_names
    assert created_agents[0]["middleware"][-1].run_limit == 2
    assert created_agents[0]["middleware"][-1].exit_behavior == "error"
    assert "transformers" not in created_agents[0]
    assert len(fake_agent.calls) == 1


def test_langchain_executor_returns_tool_limit_error(monkeypatch) -> None:
    fake_agent = _FakeRaisingAgent(ToolCallLimitExceededError(0, 3, None, 2))

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    import asyncio

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="Do too many tool calls",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="Calculator_calculate", description="calculate")],
            run_config=None,
            require_tool_call=True,
        )
    )

    assert answer == "Tool call limit reached: run limit exceeded (3/2 calls)."
    assert tools_used == []
    assert invocations == []
    assert len(fake_agent.calls) == 1


def test_build_system_prompt_uses_mixed_prompt_when_oracle_retrieval_tool_present() -> None:
    prompt = mod._build_system_prompt(
        "How can I create applications?",
        [SimpleNamespace(name="oracle_retrieval", description="retrieve")],
        run_config=None,
    )
    assert "When document context was provided in the user message" in prompt
    assert "Treat retrieval as evidence for collection facts only" in prompt
    assert "Before saying information is unavailable from the selected collection" in prompt
    assert "Prefer the most specific listed tool for each requested action" in prompt


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


def test_langchain_executor_does_not_mutate_missing_tool_call_ids(monkeypatch) -> None:
    fake_agent = _FakeAgent(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "oci.run", "args": {"command": "os ns get"}, "id": ""}],
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "",
                                "function": {
                                    "name": "oci.run",
                                    "arguments": {"command": "os ns get"},
                                },
                            }
                        ]
                    },
                    response_metadata={
                        "tool_calls": [
                            {
                                "id": "",
                                "function": {
                                    "name": "oci.run",
                                    "arguments": {"command": "os ns get"},
                                },
                            }
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
    assert first.tool_calls[0]["id"] == ""
    assert first.additional_kwargs["tool_calls"][0]["id"] == ""
    assert first.response_metadata["tool_calls"][0]["id"] == ""


def test_extract_answer_and_tools_ignores_mapping_messages() -> None:
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
    assert answer == ""
    assert tools_used == []


def test_extract_answer_and_tools_ignores_provider_side_channel_tool_calls() -> None:
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
    assert tools_used == []


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


def test_extract_tool_invocations_preserves_tool_message_error_status() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "oic_CREATE_INVOICE",
                        "args": {"InvoiceData": {}},
                        "id": "call_1",
                    },
                ],
            ),
            ToolMessage(
                content="validation failed",
                tool_call_id="call_1",
                name="oic_CREATE_INVOICE",
                status="error",
            ),
            AIMessage(content="."),
        ]
    }

    invocations = mod._extract_tool_invocations(state)
    assert invocations == [
        {
            "tool_name": "oic_CREATE_INVOICE",
            "args": {"InvoiceData": {}},
            "result": "validation failed",
            "error": "validation failed",
        },
    ]


def test_extract_tool_invocations_ignores_mapping_messages() -> None:
    state = {
        "messages": [
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "oci-mcp-server_run_oci_command",
                            "arguments": '{"command": ["logs", "api"]}',
                        },
                    },
                ],
            },
            {
                "type": "tool",
                "name": "oci-mcp-server_run_oci_command",
                "tool_call_id": "call_1",
                "content": "log line one\nlog line two",
            },
        ]
    }

    invocations = mod._extract_tool_invocations(state)
    assert invocations == []
