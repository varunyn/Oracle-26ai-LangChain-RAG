import asyncio
import importlib


def test_select_mcp_tools_for_question_async_returns_direct_descriptors(monkeypatch):
    selector_mod = importlib.import_module("rag_agent.infrastructure.tool_selection")

    async def fake_get_mcp_tool_metadata_async(*, server_keys=None, run_config=None):
        return [
            {
                "canonical_name": "beta.lookup",
                "tool_name": "lookup",
                "server_key": "beta",
                "description": "Lookup records",
                "input_schema": {"type": "object"},
            },
            {
                "canonical_name": "alpha.describe",
                "tool_name": "describe",
                "server_key": "alpha",
                "description": "Describe resource",
                "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
            {"tool_name": "missing-canonical"},
        ]

    monkeypatch.setattr(
        selector_mod,
        "get_mcp_tool_metadata_async",
        fake_get_mcp_tool_metadata_async,
    )

    result = asyncio.run(
        selector_mod.select_mcp_tools_for_question_async(
            "Describe instance",
            limit=1,
            server_keys=["alpha", "beta"],
        )
    )

    assert result["selection_failed"] is False
    assert result["failure"] is None
    assert result["total_tools"] == 2
    assert result["question"] == "Describe instance"
    assert result["limit"] == 1
    assert result["selected_tools"] == [
        {
            "canonical_name": "alpha.describe",
            "tool_name": "describe",
            "server_key": "alpha",
            "description": "Describe resource",
            "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
        }
    ]


def test_select_mcp_tools_for_question_async_handles_metadata_loader_failure(monkeypatch):
    selector_mod = importlib.import_module("rag_agent.infrastructure.tool_selection")

    async def fake_get_mcp_tool_metadata_async(*, server_keys=None, run_config=None):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(
        selector_mod,
        "get_mcp_tool_metadata_async",
        fake_get_mcp_tool_metadata_async,
    )

    result = asyncio.run(selector_mod.select_mcp_tools_for_question_async("What can you do?"))

    assert result["selection_failed"] is True
    assert result["selected_tools"] == []
    assert result["total_tools"] == 0
    assert result["failure"] == {
        "stage": "metadata_load",
        "error_type": "RuntimeError",
        "message": "metadata unavailable",
    }


def test_select_mcp_tools_for_question_sync_matches_async(monkeypatch):
    selector_mod = importlib.import_module("rag_agent.infrastructure.tool_selection")

    metadata = [
        {
            "canonical_name": "gamma.search",
            "tool_name": "search",
            "server_key": "gamma",
            "description": "Search dataset",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]

    async def fake_get_mcp_tool_metadata_async(*, server_keys=None, run_config=None):
        return metadata

    def fake_get_mcp_tool_metadata(*, server_keys=None, run_config=None):
        return metadata

    monkeypatch.setattr(
        selector_mod,
        "get_mcp_tool_metadata_async",
        fake_get_mcp_tool_metadata_async,
    )
    monkeypatch.setattr(
        selector_mod,
        "get_mcp_tool_metadata",
        fake_get_mcp_tool_metadata,
    )

    sync_result = selector_mod.select_mcp_tools_for_question("Find records", limit=5)
    async_result = asyncio.run(
        selector_mod.select_mcp_tools_for_question_async("Find records", limit=5)
    )

    assert sync_result == async_result
