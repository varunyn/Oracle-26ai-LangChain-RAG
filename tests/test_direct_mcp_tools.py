import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.tools import BaseTool

from src.rag_agent.infrastructure import direct_mcp_tools


def test_direct_mcp_tools_import_smoke() -> None:
    assert hasattr(direct_mcp_tools, "get_mcp_tools")
    assert hasattr(direct_mcp_tools, "get_mcp_tools_async")
    assert hasattr(direct_mcp_tools, "get_mcp_tool_metadata")
    assert hasattr(direct_mcp_tools, "get_mcp_tool_metadata_async")


def test_get_mcp_tools_async_graceful_failure_on_discovery_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        direct_mcp_tools,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True, mcp_server_keys=None),
    )
    monkeypatch.setattr(
        direct_mcp_tools,
        "get_mcp_servers_config",
        lambda: {"default": {"url": "http://mcp.example"}},
    )

    async def raise_on_load(server_key: str, server_config: object) -> list[object]:
        _ = server_key
        _ = server_config
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(direct_mcp_tools, "_load_tools_for_server_async", raise_on_load)

    tools = asyncio.run(direct_mcp_tools.get_mcp_tools_async())
    metadata = asyncio.run(direct_mcp_tools.get_mcp_tool_metadata_async())

    assert tools == []
    assert metadata == []


def test_get_mcp_tool_metadata_async_serialization_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped_tool = direct_mcp_tools.DirectMCPTool(
        name="default.search_docs",
        description="Search docs",
        server_key="default",
        mcp_tool_name="search_docs",
        server_config={"url": "http://mcp.example"},
        mcp_input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    loaded = [
        direct_mcp_tools._LoadedTool(  # pyright: ignore[reportPrivateUsage]
            tool=wrapped_tool,
            metadata=direct_mcp_tools.MCPToolMetadata(
                canonical_name="default.search_docs",
                tool_name="search_docs",
                server_key="default",
                description="Search docs",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        )
    ]

    async def fake_load_catalog(
        **_: object,
    ) -> list[direct_mcp_tools._LoadedTool]:  # pyright: ignore[reportPrivateUsage]
        return loaded

    monkeypatch.setattr(direct_mcp_tools, "_load_catalog_async", fake_load_catalog)

    metadata = asyncio.run(direct_mcp_tools.get_mcp_tool_metadata_async())

    assert metadata == [
        {
            "canonical_name": "default.search_docs",
            "tool_name": "search_docs",
            "server_key": "default",
            "description": "Search docs",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]


def test_get_mcp_tools_sync_wraps_async_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_tool = direct_mcp_tools.DirectMCPTool(
        name="default.lookup",
        description="Lookup",
        server_key="default",
        mcp_tool_name="lookup",
        server_config={"url": "http://mcp.example"},
    )

    async def fake_get_mcp_tools_async(**_: object) -> list[BaseTool]:
        return [expected_tool]

    monkeypatch.setattr(direct_mcp_tools, "get_mcp_tools_async", fake_get_mcp_tools_async)

    tools = direct_mcp_tools.get_mcp_tools()

    assert tools == [expected_tool]
