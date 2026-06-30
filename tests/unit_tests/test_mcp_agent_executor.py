from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolCallTransformer

from src.rag_agent.graphs.state import ChatGraphState
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


class _ConfigurableLLM:
    def __init__(self) -> None:
        self.configs: list[object] = []
        self.is_stream = False

    def with_config(self, config: object) -> _ConfigurableLLM:
        self.configs.append(config)
        return self


class _BindableFakeMessagesModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: object,
        *,
        tool_choice: object | None = None,
        **kwargs: object,
    ) -> _BindableFakeMessagesModel:
        _ = tools, tool_choice, kwargs
        return self


@tool
def _native_lookup(query: str) -> str:
    """Lookup a deterministic test value."""

    return f"found {query}"


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


def test_langchain_executor_suppresses_internal_llm_message_streams(monkeypatch) -> None:
    fake_agent = _FakeAgent({"messages": [AIMessage(content="done")]})
    fake_llm = _ConfigurableLLM()
    models: list[object] = []

    def fake_create_agent(**kwargs: Any) -> _FakeAgent:
        models.append(kwargs["model"])
        return fake_agent

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: fake_llm)
    monkeypatch.setattr(mod, "create_agent", fake_create_agent)

    answer, tools_used, invocations = asyncio.run(
        mod.get_mcp_answer_with_langchain_agent_async(
            question="finish",
            chat_history=None,
            model_id=None,
            tools=[SimpleNamespace(name="finish", description="finish")],
            run_config=None,
            require_tool_call=False,
        )
    )

    assert answer == "done"
    assert tools_used == []
    assert invocations == []
    assert models == [fake_llm]
    assert fake_llm.configs == [{"tags": ["nostream"]}]
    assert fake_llm.is_stream is False
    assert fake_agent.calls[0]["config"] == {"tags": ["nostream"]}


def test_graph_native_mcp_execution_emits_tool_call_projection(monkeypatch) -> None:
    fake_model = _BindableFakeMessagesModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "_native_lookup",
                        "args": {"query": "invoice"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: fake_model)

    async def mcp_node(state: ChatGraphState, config: RunnableConfig) -> ChatGraphState:
        result = await mod.get_mcp_answer_execution_with_langchain_agent_async(
            question="use lookup",
            chat_history=state["messages"],
            model_id=None,
            tools=[_native_lookup],
            run_config=config,
            require_tool_call=False,
        )
        return {"messages": result.state_messages}

    async def run() -> list[str]:
        graph = (
            StateGraph(ChatGraphState)
            .add_node("mcp", mcp_node)
            .add_edge(START, "mcp")
            .add_edge("mcp", END)
            .compile(transformers=[ToolCallTransformer])
        )
        raw_stream = graph.astream_events(
            {"messages": [{"role": "user", "content": "use lookup"}]},
            version="v3",
        )
        stream = await raw_stream if inspect.isawaitable(raw_stream) else raw_stream
        tool_names: list[str] = []
        async for tool_call in stream.tool_calls:
            tool_names.append(str(tool_call.tool_name))
        return tool_names

    assert asyncio.run(run()) == ["_native_lookup"]


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


def test_langchain_executor_state_messages_exclude_input_human_message(monkeypatch) -> None:
    input_message = HumanMessage(content="use lookup")
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "x"},
                "id": "call-1",
            }
        ],
    )
    tool_message = ToolMessage(
        content="found x",
        tool_call_id="call-1",
        name="lookup",
    )
    final_message = AIMessage(content="done")
    fake_agent = _FakeAgent(
        {"messages": [input_message, tool_call_message, tool_message, final_message]}
    )

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: object())
    monkeypatch.setattr(mod, "create_agent", lambda **kwargs: fake_agent)

    result = asyncio.run(
        mod.get_mcp_answer_execution_with_langchain_agent_async(
            question="use lookup",
            chat_history=[],
            model_id=None,
            tools=[SimpleNamespace(name="lookup", description="lookup")],
            run_config=None,
            require_tool_call=False,
        )
    )

    assert result.answer == "done"
    assert [type(message) for message in result.state_messages] == [
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert all(not isinstance(message, HumanMessage) for message in result.state_messages)


def test_graph_native_tool_loop_state_messages_exclude_input_human_message(monkeypatch) -> None:
    captured_state_messages: list[object] = []
    lookup_tool = StructuredTool.from_function(
        func=lambda query: f"found {query}",
        name="native_lookup",
        description="Lookup a deterministic test value.",
    )
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "native_lookup",
                    "args": {"query": "x"},
                    "id": "call-1",
                }
            ],
        ),
        AIMessage(content="done"),
    ]
    llm = _BindableFakeMessagesModel(responses=responses)

    monkeypatch.setattr("api.settings.get_settings", lambda: SimpleNamespace(MCP_MAX_ROUNDS=2))
    monkeypatch.setattr(mod, "get_llm", lambda model_id=None: llm)

    async def mcp_node(state: ChatGraphState, config: RunnableConfig) -> ChatGraphState:
        result = await mod.get_mcp_answer_execution_with_langchain_agent_async(
            question="use lookup",
            chat_history=state["messages"],
            model_id=None,
            tools=[lookup_tool],
            run_config=config,
            require_tool_call=False,
        )
        captured_state_messages[:] = result.state_messages
        return {"messages": result.state_messages}

    async def run() -> ChatGraphState:
        graph = (
            StateGraph(ChatGraphState)
            .add_node("mcp", mcp_node)
            .add_edge(START, "mcp")
            .add_edge("mcp", END)
            .compile(transformers=[ToolCallTransformer])
        )
        return await graph.ainvoke(
            {"messages": [{"role": "user", "content": "use lookup"}]},
        )

    state = asyncio.run(run())
    messages = state["messages"]
    assert [type(message) for message in captured_state_messages] == [
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert all(not isinstance(message, HumanMessage) for message in captured_state_messages)
    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert messages[0].content == "use lookup"
    assert all(not isinstance(message, HumanMessage) for message in messages[1:])


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


def test_build_system_prompt_requires_material_tool_use_when_requested() -> None:
    prompt = mod._build_system_prompt(
        "Perform a linear regression on these points using tools.",
        [SimpleNamespace(name="Calculator_basic_arithmetic", description="evaluate arithmetic")],
        run_config=None,
    )
    assert "materially contribute to the final answer" in prompt
    assert "user's actual inputs" in prompt
    assert "{{TOOL_SUMMARY}}" not in prompt
    assert "Calculator_basic_arithmetic: evaluate arithmetic" in prompt
    assert "If the available tools cannot materially help" in prompt


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
