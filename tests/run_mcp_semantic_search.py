"""
Manual script: call semantic_search on the MCP server.

Usage:
  # Start the MCP server in another terminal:
  uv run python mcp_servers/mcp_semantic_search.py
  # Then run this script:
  uv run python tests/run_mcp_semantic_search.py
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

ENDPOINT = f"http://localhost:{get_settings().PORT}/mcp"


async def main():
    """Call semantic_search for each collection in COLLECTION_LIST."""
    client = Client(ENDPOINT)
    collections = (
        get_settings().COLLECTION_LIST if get_settings().COLLECTION_LIST else ["RAG_KNOWLEDGE_BASE"]
    )
    query = "How to setup OCI CLI in linux?"

    async with client:
        print("\nCalling semantic_search for each collection in COLLECTION_LIST...\n")
        for collection_name in collections:
            print(f"--- Collection: {collection_name} | Query: {query}\n")
            raw = await client.call_tool(
                "semantic_search",
                {"query": query, "top_k": 5, "collection_name": collection_name},
            )
            results = [raw] if not isinstance(raw, (list, tuple)) else raw
            text = getattr(results[0], "data", None) or getattr(results[0], "text", str(results[0]))
            if isinstance(text, dict):
                payload = text
            else:
                payload = json.loads(text) if isinstance(text, str) else {}
            if payload.get("error"):
                print(f"Error: {payload['error']}\n")
                continue
            relevant_docs = payload.get("relevant_docs", [])
            for i, doc in enumerate(relevant_docs, 1):
                print(f"  [{i}] {doc.get('page_content', '')[:200]}...")
                print(f"      Metadata: {doc.get('metadata', {})}")
            print()


asyncio.run(main())
