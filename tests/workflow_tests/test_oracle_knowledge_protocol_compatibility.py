import asyncio
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import httpx2
from fastmcp import Client, FastMCP
from fastmcp.client.transports import PythonStdioTransport, StreamableHttpTransport
from langchain.mcp import MCPAdapter
from mcp_types.version import LATEST_MODERN_VERSION

from mcp_servers.oracle_knowledge import create_oracle_knowledge_server
from src.rag_agent.infrastructure import mcp_adapter_runtime
from tests.fixtures.oracle_knowledge_stdio_server import create_fixture_server

MATRIX = json.loads(
    (
        Path(__file__).parents[1] / "fixtures" / "oracle_knowledge_compatibility_matrix.json"
    ).read_text()
)
EXPECTED_PROOF_KEYS = {"negotiation", "tools_list", "success", "no_hits", "sanitized_failure"}


def test_matrix_schema_and_supported_rows_are_auditable():
    assert {row["status"] for row in MATRIX} <= {"supported", "unsupported", "unverified"}
    assert {row["transport"] for row in MATRIX} <= {"stdio", "streamable-http"}
    assert importlib.metadata.version("fastmcp") == "4.0.2"
    assert importlib.metadata.version("mcp") == "2.1.1"
    assert LATEST_MODERN_VERSION == "2026-07-28"
    for row in MATRIX:
        assert set(row["proof"]) == EXPECTED_PROOF_KEYS
        assert row["evidence"]["date"]
        if row["status"] == "supported":
            assert row["fastmcp_version"] == "4.0.2"
            assert row["mcp_sdk_version"] == "2.1.1"
            assert row["negotiated_protocol"] == LATEST_MODERN_VERSION
            assert all(row["proof"].values())
        else:
            assert not any(row["proof"].values())


def test_matrix_has_separate_client_and_transport_rows():
    supported = {
        (row["outer_client"], row["transport"]) for row in MATRIX if row["status"] == "supported"
    }
    assert supported == {
        ("FastMCP Client", "stdio"),
        ("FastMCP Client", "streamable-http"),
    }


def test_production_server_is_stateless_and_has_no_optional_client_dependencies():
    source = inspect.getsource(create_oracle_knowledge_server)
    for forbidden in ("session_id", "sampling", "roots", "elicitation"):
        assert forbidden not in source


async def _run_stdio_contract():
    script = str(Path(__file__).parents[1] / "fixtures" / "oracle_knowledge_stdio_server.py")
    env = {"PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"}
    async with Client(
        PythonStdioTransport(
            script_path=script,
            python_cmd=".venv/bin/python",
            cwd=str(Path.cwd()),
            env=env,
            keep_alive=False,
        )
    ) as client:
        assert client.protocol_version == LATEST_MODERN_VERSION
        assert len(await client.list_tools()) == 3
        success = await client.call_tool(
            "search_knowledge", {"query": "find", "knowledge_base": "docs"}
        )
        no_hits = await client.call_tool(
            "search_knowledge", {"query": "find", "knowledge_base": "empty"}
        )
        failure = await client.call_tool(
            "search_knowledge", {"query": "find", "knowledge_base": "fail"}
        )
        assert success.structured_content["outcome"] == "success"
        assert no_hits.structured_content["outcome"] == "no_hits"
        assert failure.structured_content["outcome"] == "backend_error"
        assert "safe oracle failure" not in str(failure.structured_content)


async def _run_http_contract():
    server = create_fixture_server()
    app = server.http_app(transport="streamable-http")

    def factory(**kwargs):
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test", **kwargs
        )

    async with app.router.lifespan_context(app):
        async with Client(
            StreamableHttpTransport("http://test/mcp", httpx_client_factory=factory)
        ) as client:
            assert client.protocol_version == LATEST_MODERN_VERSION
            assert len(await client.list_tools()) == 3
            success, no_hits = await asyncio.gather(
                client.call_tool("search_knowledge", {"query": "find", "knowledge_base": "docs"}),
                client.call_tool("search_knowledge", {"query": "find", "knowledge_base": "empty"}),
            )
            assert success.structured_content["outcome"] == "success"
            assert no_hits.structured_content["outcome"] == "no_hits"


def test_real_profiles_negotiate_and_support_fresh_or_concurrent_calls():
    asyncio.run(_run_stdio_contract())
    asyncio.run(_run_http_contract())


def test_first_party_langchain_adapter_loads_and_calls_namespaced_stdio_tool(monkeypatch):
    script = str(Path(__file__).parents[1] / "fixtures" / "oracle_knowledge_stdio_server.py")
    env = {"PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"}
    monkeypatch.setattr(
        mcp_adapter_runtime,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True),
    )

    async def run():
        await mcp_adapter_runtime.clear_adapter_runtime_cache()
        tools = await mcp_adapter_runtime.load_adapter_tools(
            run_config={
                "configurable": {
                    "mcp_servers_config_override": {
                        "oracle": {
                            "transport": "stdio",
                            "command": ".venv/bin/python",
                            "args": [script],
                            "cwd": str(Path.cwd()),
                            "env": env,
                            "keep_alive": False,
                        },
                        "secondary": {
                            "transport": "stdio",
                            "command": ".venv/bin/python",
                            "args": [script],
                            "cwd": str(Path.cwd()),
                            "env": env,
                            "keep_alive": False,
                        },
                    }
                }
            }
        )
        assert [tool.name for tool in tools] == [
            "oracle_search_knowledge",
            "oracle_list_knowledge_bases",
            "oracle_list_documents",
            "secondary_search_knowledge",
            "secondary_list_knowledge_bases",
            "secondary_list_documents",
        ]
        search = tools[3]
        content, artifact = await search.coroutine(query="find", knowledge_base="docs")
        assert artifact["structured_content"]["outcome"] == "success"
        assert json.loads(content[0]["text"])["evidence"][0]["source"] == "fixture"
        await mcp_adapter_runtime.clear_adapter_runtime_cache()

    asyncio.run(run())


def test_real_first_party_adapter_normalizes_success_warning_tool_message():
    server = FastMCP("warning-result")

    @server.tool
    def run_cli() -> dict[str, object]:
        """Return a successful CLI result with a warning in the error field."""
        return {"returncode": 0, "error": "warning text"}

    async def run():
        tools = await MCPAdapter(server).list_tools()
        wrapped = mcp_adapter_runtime._normalize_adapter_tool(tools[0])
        message = await wrapped.ainvoke(
            {
                "type": "tool_call",
                "name": "run_cli",
                "id": "call-warning",
                "args": {},
            }
        )
        assert message.status == "success"
        assert message.artifact["structured_content"] == {
            "returncode": 0,
            "warnings": ["warning text"],
        }
        assert json.loads(message.content[0]["text"]) == {
            "returncode": 0,
            "warnings": ["warning text"],
        }

    asyncio.run(run())
