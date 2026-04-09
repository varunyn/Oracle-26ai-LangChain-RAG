"""
Manual script: call get_collections on the MCP server.

Usage:
  # Start the MCP server first, then:
  uv run python tests/run_mcp_list_collection.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio

from fastmcp import Client

from api.settings import get_settings
from src.rag_agent.utils.utils import print_mcp_available_tools

ENDPOINT = f"http://localhost:{get_settings().PORT}/mcp/"


async def main():
    """List tools and call get_collections on the MCP server."""
    client = Client(ENDPOINT)
    async with client:
        tools = await client.list_tools()
        print_mcp_available_tools(tools)
        print("\nCalling get_collections tool...\n")
        raw = await client.call_tool("get_collections", {})
        results = [raw] if not isinstance(raw, (list, tuple)) else raw
        print("List Collections Results:")
        for result in results:
            text = getattr(result, "data", None) or getattr(result, "text", str(result))
            print("Collection name:", text)
        print()


asyncio.run(main())
