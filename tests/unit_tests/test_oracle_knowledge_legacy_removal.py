import asyncio
from pathlib import Path

from fastmcp import Client

from tests.fixtures.oracle_knowledge_stdio_server import create_fixture_server


def test_legacy_standalone_files_and_settings_are_removed():
    root = Path(__file__).parents[2]
    for relative in (
        "mcp_servers/mcp_semantic_search.py",
        "mcp_servers/mcp_rag_server.py",
        "tests/run_mcp_semantic_search.py",
        "tests/run_mcp_list_collection.py",
        "tests/run_mcp_rag.py",
        "tests/run_mcp_minimal.py",
    ):
        assert not (root / relative).exists()

    from api.settings import Settings

    fields = Settings.model_fields
    assert "MCP_SEARCH_MODE" not in fields
    assert "TRANSPORT" not in fields
    assert "HOST" not in fields
    assert "PORT" not in fields


def test_oracle_knowledge_tools_are_exact_after_legacy_removal():
    async def check():
        async with Client(create_fixture_server()) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools] == [
                "search_knowledge",
                "list_knowledge_bases",
                "list_documents",
            ]

    asyncio.run(check())
