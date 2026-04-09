"""
Manual script: call rag_ask on the RAG MCP server.

Usage:
  # Start the RAG MCP server in another terminal (uses config.PORT, config.TRANSPORT):
  uv run python mcp_servers/mcp_rag_server.py
  # Then run this script:
  uv run python tests/run_mcp_rag.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio
import json

from fastmcp import Client

from api.settings import get_settings
from src.rag_agent.utils.utils import print_mcp_available_tools

ENDPOINT = f"http://localhost:{get_settings().PORT}/mcp"


async def main():
    """List tools and call rag_ask on the RAG MCP server."""
    client = Client(ENDPOINT)
    async with client:
        tools = await client.list_tools()
        print_mcp_available_tools(tools)
        print("\nCalling rag_ask tool...\n")
        raw = await client.call_tool(
            "rag_ask",
            {"question": "How to setup OCI CLI in linux?"},
        )
        results = [raw] if not isinstance(raw, (list, tuple)) else raw
        for result in results:
            data = getattr(result, "data", None) or getattr(result, "text", None)
            if data is None:
                print("Result:", result)
                continue
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    print("Answer:", data)
                    continue
            if not isinstance(data, dict):
                print("Result:", data)
                continue
            print("Answer:", data.get("answer", ""))
            print("Citations:", json.dumps(data.get("citations", []), indent=2))
            if data.get("error"):
                print("Error:", data["error"])
        print()


if __name__ == "__main__":
    asyncio.run(main())
