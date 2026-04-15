import asyncio

import pytest
from langchain_core.tools import BaseTool
from pydantic import Field

from src.rag_agent.infrastructure import direct_mcp_tools


def test_direct_mcp_tools_import_smoke() -> None:
    assert hasattr(direct_mcp_tools, "get_mcp_tools")
    assert hasattr(direct_mcp_tools, "get_mcp_tools_async")
    assert hasattr(direct_mcp_tools, "get_mcp_tool_metadata")
    assert hasattr(direct_mcp_tools, "get_mcp_tool_metadata_async")


def test_get_mcp_tools_async_graceful_failure_on_discovery_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raise_on_load_adapter_tools(**_: object) -> list[BaseTool]:
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", raise_on_load_adapter_tools)

    tools = asyncio.run(direct_mcp_tools.get_mcp_tools_async())
    metadata = asyncio.run(direct_mcp_tools.get_mcp_tool_metadata_async())

    assert tools == []
    assert metadata == []


def test_get_mcp_tool_metadata_async_serialization_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped_tool = _FakeAdapterTool(
        name="default.search_docs",
        description="Search docs",
    )

    async def fake_load_adapter_tools(**_: object) -> list[BaseTool]:
        return [wrapped_tool]

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", fake_load_adapter_tools)

    metadata = asyncio.run(direct_mcp_tools.get_mcp_tool_metadata_async())

    assert metadata == [
        {
            "canonical_name": "default.search_docs",
            "tool_name": "search_docs",
            "server_key": "default",
            "description": "Search docs",
            "input_schema": {},
        }
    ]


def test_get_mcp_tools_sync_wraps_async_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_tool = _FakeAdapterTool(name="default.lookup", description="Lookup")

    async def fake_get_mcp_tools_async(**_: object) -> list[BaseTool]:
        return [expected_tool]

    monkeypatch.setattr(direct_mcp_tools, "get_mcp_tools_async", fake_get_mcp_tools_async)

    tools = direct_mcp_tools.get_mcp_tools()

    assert tools == [expected_tool]


class _FakeAdapterTool(BaseTool):
    name: str = Field()
    description: str = Field()
    args_schema: dict[str, object] | None = None

    def _run(self, *args: object, **kwargs: object) -> object:
        _ = args
        _ = kwargs
        raise NotImplementedError


FAKE_ARGS_SCHEMA: dict[str, object] = {"type": "object", "properties": {"query": {"type": "string"}}}


def test_get_mcp_tools_async_returns_adapter_loaded_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_tool = _FakeAdapterTool(name="math.solve", description="Solve equations")

    async def fake_load_adapter_tools(**_: object) -> list[BaseTool]:
        return [expected_tool]

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", fake_load_adapter_tools)

    tools = asyncio.run(direct_mcp_tools.get_mcp_tools_async())

    assert tools == [expected_tool]


def test_get_mcp_tool_metadata_async_normalizes_adapter_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_load_adapter_tools(**_: object) -> list[BaseTool]:
        return [
            _FakeAdapterTool(
                name="math.search_docs",
                description="Search docs",
                args_schema=FAKE_ARGS_SCHEMA,
            )
        ]

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", fake_load_adapter_tools)

    metadata = asyncio.run(direct_mcp_tools.get_mcp_tool_metadata_async())

    assert metadata == [
        {
            "canonical_name": "math.search_docs",
            "tool_name": "search_docs",
            "server_key": "math",
            "description": "Search docs",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]
