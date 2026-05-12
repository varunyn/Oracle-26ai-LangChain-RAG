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


def test_get_mcp_tools_async_sanitizes_openapi_extensions_from_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "invoice": {
                "type": "object",
                "x-visible": True,
                "x-in": "body",
                "properties": {
                    "status": {
                        "type": "string",
                        "const": "approved",
                        "x-visible": False,
                    },
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "amount": {"type": "number", "x-visible": True}
                            },
                        },
                    },
                },
            }
        },
    }

    async def fake_load_adapter_tools(**_: object) -> list[BaseTool]:
        return [
            _FakeAdapterTool(
                name="oic.process_invoice",
                description="Process invoice",
                args_schema=raw_schema,
            )
        ]

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", fake_load_adapter_tools)

    tools = asyncio.run(direct_mcp_tools.get_mcp_tools_async())

    sanitized_schema = tools[0].args_schema
    assert sanitized_schema == {
        "type": "object",
        "properties": {
            "invoice": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["approved"],
                    },
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "amount": {"type": "number"},
                            },
                        },
                    },
                },
            }
        },
    }
    assert raw_schema["properties"]["invoice"]["x-visible"] is True  # type: ignore[index]


def test_get_mcp_tools_async_drops_boolean_const_for_string_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "includeDetails": {
                "type": "string",
                "const": True,
                "x-visible": True,
            }
        },
    }

    async def fake_load_adapter_tools(**_: object) -> list[BaseTool]:
        return [
            _FakeAdapterTool(
                name="oic.list_tools",
                description="List tools",
                args_schema=raw_schema,
            )
        ]

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", fake_load_adapter_tools)

    tools = asyncio.run(direct_mcp_tools.get_mcp_tools_async())
    sanitized_schema = tools[0].args_schema

    assert sanitized_schema == {
        "type": "object",
        "properties": {
            "includeDetails": {
                "type": "string",
            }
        },
    }


def test_get_mcp_tools_async_drops_non_string_const_for_non_string_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "const": True,
                "x-in": "query",
            }
        },
    }

    async def fake_load_adapter_tools(**_: object) -> list[BaseTool]:
        return [
            _FakeAdapterTool(
                name="oic.list_tools",
                description="List tools",
                args_schema=raw_schema,
            )
        ]

    monkeypatch.setattr(direct_mcp_tools, "load_adapter_tools", fake_load_adapter_tools)

    tools = asyncio.run(direct_mcp_tools.get_mcp_tools_async())
    sanitized_schema = tools[0].args_schema

    assert sanitized_schema == {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
            }
        },
    }
