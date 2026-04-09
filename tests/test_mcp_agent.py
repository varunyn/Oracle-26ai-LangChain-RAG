import asyncio
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

_mcp_skip_reason = ""
try:
    from src.rag_agent.infrastructure.mcp_agent import get_mcp_answer
except ImportError as e:
    get_mcp_answer = None  # type: ignore[assignment]
    _mcp_skip_reason = str(e)

if get_mcp_answer is None:
    pytest.skip(f"mcp_agent not available: {_mcp_skip_reason}", allow_module_level=True)

get_mcp_answer_fn = cast(Callable[..., tuple[str, list[str]]], get_mcp_answer)


def test_get_mcp_answer_disabled():
    """Test get_mcp_answer when MCP is disabled."""
    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=False),
    ):
        answer, tools_used = get_mcp_answer_fn("test question")
        assert answer == ""
        assert tools_used == []


def test_get_mcp_answer_no_tools_fallback():
    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch(
            "src.rag_agent.infrastructure.mcp_agent.get_mcp_tools_async",
            new=AsyncMock(return_value=[]),
        ):
            answer, tools_used = get_mcp_answer_fn("test question")
        assert answer == "MCP tools are currently unavailable. Please try again."
        assert tools_used == []


def test_get_mcp_answer_with_direct_tools_default_loader():
    class StubLLM:
        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses: Iterator[AIMessage] = iter(responses)

        def bind_tools(self, tools: list[object], **kwargs: object) -> "StubLLM":
            _ = tools
            _ = kwargs
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return next(self._responses)

    class StubTool:
        name: str = "calculator.calculate"
        description: str = "Calculate arithmetic expressions"

        def invoke(self, tool_call: dict[str, object]) -> dict[str, object]:
            _ = tool_call
            return {"result": "ok", "logs": ["log"]}

    mock_tool = StubTool()

    mock_llm = StubLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator.calculate",
                        "args": {"expression": "2+2"},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(content="Test answer"),
        ]
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch(
            "src.rag_agent.infrastructure.mcp_agent.get_mcp_tools_async",
            new=AsyncMock(return_value=[mock_tool]),
        ):
            with patch("src.rag_agent.infrastructure.mcp_agent.get_llm", return_value=mock_llm):
                answer, tools_used = get_mcp_answer_fn("test question", tools=None)
                assert answer == "Test answer"
                assert tools_used == ["calculator.calculate"]


def test_get_mcp_answer_require_tool_call_retry_then_success():
    bind_calls: list[dict[str, object]] = []

    class StubLLM:
        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses: Iterator[AIMessage] = iter(responses)

        def bind_tools(self, tools: list[object], **kwargs: object) -> "StubLLM":
            bind_calls.append({"tools": tools, "kwargs": kwargs})
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return next(self._responses)

    class StubTool:
        name: str = "calculator.calculate"
        description: str = "Calculate arithmetic expressions"

        def invoke(self, tool_call: dict[str, object]) -> dict[str, object]:
            _ = tool_call
            return {"result": "ok", "logs": []}

    mock_tool = StubTool()
    mock_llm = StubLLM(
        [
            AIMessage(content="No tool call yet"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator.calculate",
                        "args": {"expression": "2+2"},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(content="Final answer"),
        ]
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch(
            "src.rag_agent.infrastructure.mcp_agent.get_mcp_tools_async",
            new=AsyncMock(return_value=[mock_tool]),
        ):
            with patch("src.rag_agent.infrastructure.mcp_agent.get_llm", return_value=mock_llm):
                answer, tools_used = get_mcp_answer_fn(
                    "test question",
                    require_tool_call=True,
                )
                assert answer == "Final answer"
                assert tools_used == ["calculator.calculate"]
                assert bind_calls[0]["tools"] == [mock_tool]
                assert bind_calls[0]["kwargs"] == {"tool_choice": "required"}
                assert bind_calls[1]["kwargs"] == {}


def test_get_mcp_answer_require_tool_call_retry_fallback():
    class StubLLM:
        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses: Iterator[AIMessage] = iter(responses)

        def bind_tools(self, tools: list[object], **kwargs: object) -> "StubLLM":
            _ = tools
            _ = kwargs
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return next(self._responses)

    mock_llm = StubLLM(
        [
            AIMessage(content="No tools used"),
            AIMessage(content="Still no tools"),
        ]
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch("src.rag_agent.infrastructure.mcp_agent.get_llm", return_value=mock_llm):
            answer, tools_used = get_mcp_answer_fn(
                "test question",
                require_tool_call=True,
                tools=[SimpleNamespace(name="calculator.calculate", description="desc")],
            )
            assert (
                answer
                == "MCP tool call required but none was produced after retry. Please try again."
            )
            assert tools_used == []


def test_get_mcp_answer_require_tool_call_rejects_textual_tool_call_after_retry():
    class StubLLM:
        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses: Iterator[AIMessage] = iter(responses)

        def bind_tools(self, tools: list[object], **kwargs: object) -> "StubLLM":
            _ = tools
            _ = kwargs
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return next(self._responses)

    mock_llm = StubLLM(
        [
            AIMessage(content='calculator.integrate(expression="x**2 * exp(x)")'),
            AIMessage(content='calculator.integrate(expression="x**2 * exp(x)")'),
        ]
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch("src.rag_agent.infrastructure.mcp_agent.get_llm", return_value=mock_llm):
            answer, tools_used = get_mcp_answer_fn(
                "integrate this",
                require_tool_call=True,
                tools=[SimpleNamespace(name="calculator.integrate", description="Integrate expressions")],
            )

    assert answer == "MCP tool call required but none was produced after retry. Please try again."
    assert tools_used == []


def test_get_mcp_answer_async_uses_direct_loader_fallback_when_no_tools():
    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch(
            "src.rag_agent.infrastructure.mcp_agent.get_mcp_tools_async",
            new=AsyncMock(return_value=[]),
        ):
            from src.rag_agent.infrastructure.mcp_agent import get_mcp_answer_async

            answer, tools_used = asyncio.run(get_mcp_answer_async("test question"))
            assert answer == "MCP tools are currently unavailable. Please try again."
            assert tools_used == []


def test_get_mcp_answer_unwraps_kwargs_wrapped_tool_args():
    class StubLLM:
        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses: Iterator[AIMessage] = iter(responses)

        def bind_tools(self, tools: list[object]) -> "StubLLM":
            _ = tools
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return next(self._responses)

    class StubTool:
        name: str = "calculator.solve_equation"
        description: str = "Solve equations"

        def __init__(self) -> None:
            self.seen_tool_call: dict[str, object] | None = None

        def invoke(self, tool_call: dict[str, object]) -> dict[str, object]:
            self.seen_tool_call = tool_call
            return {"result": "roots"}

    mock_tool = StubTool()
    mock_llm = StubLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator.solve_equation",
                        "args": {"kwargs": {"equation": "x**2 - 5*x + 6 = 0"}},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(content="Solved"),
        ]
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch("src.rag_agent.infrastructure.mcp_agent.get_llm", return_value=mock_llm):
            answer, tools_used = get_mcp_answer_fn(
                "solve it",
                tools=[mock_tool],
            )

    assert answer == "Solved"
    assert tools_used == ["calculator.solve_equation"]
    assert mock_tool.seen_tool_call is not None
    assert mock_tool.seen_tool_call["args"] == {"equation": "x**2 - 5*x + 6 = 0"}


def test_get_mcp_answer_preserves_existing_value_unwrap_behavior():
    class StubLLM:
        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses: Iterator[AIMessage] = iter(responses)

        def bind_tools(self, tools: list[object]) -> "StubLLM":
            _ = tools
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return next(self._responses)

    class StubTool:
        name: str = "calculator.solve_equation"
        description: str = "Solve equations"

        def __init__(self) -> None:
            self.seen_tool_call: dict[str, object] | None = None

        def invoke(self, tool_call: dict[str, object]) -> dict[str, object]:
            self.seen_tool_call = tool_call
            return {"result": "roots"}

    mock_tool = StubTool()
    mock_llm = StubLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator.solve_equation",
                        "args": {"equation": {"value": "x**2 - 5*x + 6 = 0"}},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(content="Solved"),
        ]
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch("src.rag_agent.infrastructure.mcp_agent.get_llm", return_value=mock_llm):
            answer, tools_used = get_mcp_answer_fn(
                "solve it",
                tools=[mock_tool],
            )

    assert answer == "Solved"
    assert tools_used == ["calculator.solve_equation"]
    assert mock_tool.seen_tool_call is not None
    assert mock_tool.seen_tool_call["args"] == {"equation": "x**2 - 5*x + 6 = 0"}


def test_get_mcp_answer_stops_repeated_identical_tool_calls_after_first_execution():
    class StubLLM:
        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses: Iterator[AIMessage] = iter(responses)

        def bind_tools(self, tools: list[object], **kwargs: object) -> "StubLLM":
            _ = tools
            _ = kwargs
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = messages
            _ = config
            return next(self._responses)

    class StubTool:
        name: str = "calculator.solve_equation"
        description: str = "Solve equations"

        def __init__(self) -> None:
            self.invocations = 0

        def invoke(self, tool_call: dict[str, object]) -> dict[str, object]:
            self.invocations += 1
            return {"solutions": ["2", "3"]}

    mock_tool = StubTool()
    repeated_call = {
        "name": "calculator.solve_equation",
        "args": {"kwargs": {"equation": "x**2 - 5*x + 6 = 0"}},
        "id": "t1",
    }
    mock_llm = StubLLM(
        [
            AIMessage(content="", tool_calls=[repeated_call]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator.solve_equation",
                        "args": {"equation": {"value": "x**2 - 5*x + 6 = 0"}},
                        "id": "t2",
                    }
                ],
            ),
        ]
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch("src.rag_agent.infrastructure.mcp_agent.get_llm", return_value=mock_llm):
            answer, tools_used = get_mcp_answer_fn(
                "solve it",
                tools=[mock_tool],
            )

    assert answer == '{"solutions": ["2", "3"]}'
    assert tools_used == ["calculator.solve_equation"]
    assert mock_tool.invocations == 1


def test_get_mcp_answer_rebinds_without_required_tool_choice_after_first_success():
    bind_calls: list[dict[str, object]] = []
    captured_second_messages: list[object] = []

    class StubLLM:
        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses: Iterator[AIMessage] = iter(responses)
            self.seen_messages: list[list[object]] = []

        def bind_tools(self, tools: list[object], **kwargs: object) -> "StubLLM":
            bind_calls.append({"tools": tools, "kwargs": kwargs})
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> AIMessage:
            _ = config
            self.seen_messages.append(messages)
            if len(self.seen_messages) == 2:
                captured_second_messages.extend(messages)
            return next(self._responses)

    class StubTool:
        name: str = "calculator.solve_equation"
        description: str = "Solve equations"

        def invoke(self, tool_call: dict[str, object]) -> dict[str, object]:
            _ = tool_call
            return {"solutions": ["2", "3"]}

    mock_tool = StubTool()
    mock_llm = StubLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator.solve_equation",
                        "args": {"equation": "x**2 - 5*x + 6 = 0"},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(content="The solutions are x = 2 and x = 3."),
        ]
    )

    with patch(
        "src.rag_agent.infrastructure.mcp_agent.get_mcp_settings",
        return_value=SimpleNamespace(enable_mcp_tools=True),
    ):
        with patch("src.rag_agent.infrastructure.mcp_agent.get_llm", return_value=mock_llm):
            answer, tools_used = get_mcp_answer_fn(
                "solve it",
                require_tool_call=True,
                tools=[mock_tool],
            )

    assert answer == "The solutions are x = 2 and x = 3."
    assert tools_used == ["calculator.solve_equation"]
    assert bind_calls[0]["kwargs"] == {"tool_choice": "required"}
    assert bind_calls[1]["kwargs"] == {}
    assert [type(message).__name__ for message in captured_second_messages[-2:]] == [
        "AIMessage",
        "ToolMessage",
    ]
