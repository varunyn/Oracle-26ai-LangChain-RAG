import asyncio

import pytest
from langchain_core.tools import BaseTool
from pydantic import Field

from src.rag_agent.infrastructure import direct_mcp_tools


def test_direct_mcp_tools_import_smoke() -> None:
    assert hasattr(direct_mcp_tools, "get_mcp_tools_async")
    assert hasattr(direct_mcp_tools, "load_adapter_tools")
    assert not hasattr(direct_mcp_tools, "get_mcp_tools")
    assert not hasattr(direct_mcp_tools, "get_mcp_tool_metadata")
    assert not hasattr(direct_mcp_tools, "get_mcp_tool_metadata_async")


def test_get_mcp_tools_async_propagates_discovery_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raise_on_load_adapter_tools(**_: object) -> list[BaseTool]:
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", raise_on_load_adapter_tools)

    with pytest.raises(RuntimeError, match="discovery failed"):
        asyncio.run(direct_mcp_tools.get_mcp_tools_async())


class _FakeAdapterTool(BaseTool):
    name: str = Field()
    description: str = Field()
    args_schema: dict[str, object] | None = None

    def _run(self, *args: object, **kwargs: object) -> object:
        _ = args
        _ = kwargs
        raise NotImplementedError


def test_get_mcp_tools_async_returns_adapter_loaded_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_tool = _FakeAdapterTool(name="math.solve", description="Solve equations")

    async def fake_load_adapter_tools(**_: object) -> list[BaseTool]:
        return [expected_tool]

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", fake_load_adapter_tools)

    tools = asyncio.run(direct_mcp_tools.get_mcp_tools_async())

    assert tools == [expected_tool]
