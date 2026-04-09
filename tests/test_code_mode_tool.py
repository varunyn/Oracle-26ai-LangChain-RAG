import asyncio
from unittest.mock import AsyncMock

import pytest

from src.rag_agent.infrastructure import code_mode_tool


class FakeClient:
    call_tool_chain: AsyncMock

    def __init__(self, call_tool_chain: AsyncMock) -> None:
        self.call_tool_chain = call_tool_chain


def test_call_tool_chain_tool_invoke_calls_client(monkeypatch: pytest.MonkeyPatch) -> None:
    call_tool_chain = AsyncMock(return_value={"result": 1, "logs": ["ok"]})
    mock_client = FakeClient(call_tool_chain)
    monkeypatch.setattr(code_mode_tool, "get_code_mode_client", lambda: mock_client)

    result = code_mode_tool.call_tool_chain_tool._run(  # pyright: ignore[reportPrivateUsage]
        code="return 1"
    )

    assert result == {"result": 1, "logs": ["ok"]}
    call_tool_chain.assert_awaited_once_with("return 1", timeout=30)


def test_call_tool_chain_tool_ainvoke_respects_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_tool_chain = AsyncMock(return_value={"result": "ok", "logs": []})
    mock_client = FakeClient(call_tool_chain)
    monkeypatch.setattr(code_mode_tool, "get_code_mode_client", lambda: mock_client)

    result = asyncio.run(
        code_mode_tool.call_tool_chain_tool._arun(  # pyright: ignore[reportPrivateUsage]
            code="x",
            timeout=5,
        )
    )

    assert result == {"result": "ok", "logs": []}
    call_tool_chain.assert_awaited_once_with("x", timeout=5)
