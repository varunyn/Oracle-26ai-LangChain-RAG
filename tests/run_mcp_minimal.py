"""
Manual script: call say_the_truth on the minimal MCP server.

Usage:
  # Start the minimal MCP server in another terminal (uses config.PORT, config.TRANSPORT):
  uv run python mcp_servers/minimal_mcp_server.py
  # Then run this script:
  uv run python tests/run_mcp_minimal.py
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

ENDPOINT = f"http://localhost:{get_settings().PORT}/mcp"


async def main():
    """List tools and call say_the_truth on the minimal MCP server."""
    client = Client(ENDPOINT)
    async with client:
        tools = await client.list_tools()
        print_mcp_available_tools(tools)
        print("\nCalling say_the_truth tool...\n")
        raw = await client.call_tool("say_the_truth", {"user": "Tester"})
        # FastMCP may return a single CallToolResult or a list
        results = [raw] if not isinstance(raw, (list, tuple)) else raw
        for result in results:
            text = getattr(result, "data", None) or getattr(result, "text", str(result))
            print("Result:", text)
        print()


if __name__ == "__main__":
    asyncio.run(main())
