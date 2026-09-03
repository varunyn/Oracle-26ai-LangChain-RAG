"""Manual deterministic MCP compatibility probe for outer-client integration work.

The JSON report is written to a file (or stderr), never to STDIO transport
stdout. Provider implementations come from the deterministic Ticket 05 fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx2
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport, StreamableHttpTransport

# Allow direct `python tests/run_...py` execution without requiring an installed package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.fixtures.oracle_knowledge_stdio_server import create_fixture_server


async def _probe(transport: str) -> dict[str, object]:
    server = create_fixture_server()
    app = server.http_app(transport="streamable-http") if transport == "streamable-http" else None
    if transport == "stdio":
        script = str(Path(__file__).parent / "fixtures" / "oracle_knowledge_stdio_server.py")
        env = {"PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"}
        client = Client(
            PythonStdioTransport(
                script_path=script,
                python_cmd=sys.executable,
                cwd=str(Path.cwd()),
                env=env,
                keep_alive=False,
            )
        )
        async with client:
            return await _collect(client)

    def factory(**kwargs):
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test", **kwargs
        )

    async with app.router.lifespan_context(app):
        async with Client(
            StreamableHttpTransport("http://test/mcp", httpx_client_factory=factory)
        ) as client:
            return await _collect(client)


async def _collect(client: Client) -> dict[str, object]:
    tools = await client.list_tools()
    success = await client.call_tool(
        "search_knowledge", {"query": "find", "knowledge_base": "docs"}
    )
    no_hits = await client.call_tool(
        "search_knowledge", {"query": "find", "knowledge_base": "empty"}
    )
    failure = await client.call_tool(
        "search_knowledge", {"query": "find", "knowledge_base": "fail"}
    )
    return {
        "protocol": client.protocol_version,
        "tools": [tool.name for tool in tools],
        "success": success.structured_content["outcome"],
        "no_hits": no_hits.structured_content["outcome"],
        "backend_failure": failure.structured_content["outcome"],
        "failure_sanitized": "safe oracle failure" not in str(failure.structured_content),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe Oracle Knowledge MCP compatibility deterministically."
    )
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--report", type=Path, help="Write JSON report here; defaults to stderr.")
    args = parser.parse_args()
    report = {"transport": args.transport, "result": asyncio.run(_probe(args.transport))}
    rendered = json.dumps(report, sort_keys=True)
    if args.report:
        args.report.write_text(rendered + "\n")
    else:
        print(rendered, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
