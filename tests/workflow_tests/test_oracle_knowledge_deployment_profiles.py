import asyncio
import sys
from pathlib import Path

import httpx
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport, StreamableHttpTransport
from fastmcp.exceptions import ToolError

from tests.fixtures.oracle_knowledge_stdio_server import create_fixture_server


class _Probe:
    def __init__(self, ready: bool, reason: str = ""):
        self.ready = ready
        self.reason = reason
        self.calls = 0

    async def check_async(self):
        self.calls += 1
        return self.ready, self.reason


async def _contract(client: Client):
    async with client:
        initialization = client.initialize_result
        protocol = getattr(initialization, "protocolVersion", None)
        assert protocol
        instructions = getattr(initialization, "instructions", "") or ""
        assert instructions
        for marker in (
            "search_knowledge",
            "docs",
            "empty",
            "fail",
            "query length <= 100000",
            "results <= 50",
            "candidates <= 100",
            "metadata filters <= 8",
            "caller writes the final answer",
            "owns citation presentation",
        ):
            assert marker in instructions
        tools = await client.list_tools()
        names = [tool.name for tool in tools]
        assert names == [
            "search_knowledge",
            "list_knowledge_bases",
            "list_documents",
        ]
        schemas = {}
        for tool in tools:
            schemas[tool.name] = {
                "input": tool.inputSchema,
                "output": tool.outputSchema,
            }
        success = await client.call_tool(
            "search_knowledge", {"query": "find", "knowledge_base": "docs"}
        )
        assert success.structured_content["outcome"] == "success"
        empty = await client.call_tool(
            "search_knowledge", {"query": "find", "knowledge_base": "empty"}
        )
        assert empty.structured_content["outcome"] == "no_hits"
        forbidden = await client.call_tool(
            "search_knowledge", {"query": "find", "knowledge_base": "RAW"}
        )
        assert forbidden.structured_content["outcome"] == "forbidden"
        invalid = await client.call_tool("search_knowledge", {"query": "   "})
        assert invalid.structured_content["outcome"] == "invalid_request"

        async def boundary_outcome(args):
            try:
                result = await client.call_tool("search_knowledge", args)
            except ToolError:
                return "invalid_request"
            return result.structured_content["outcome"]

        for args in (
            {"query": "find", "limit": 51},
            {"query": "find", "candidate_limit": 101},
            {
                "query": "find",
                "metadata_filters": {str(index): "value" for index in range(9)},
            },
            {"query": "find", "metadata_filters": {"unsupported": "value"}},
            {"query": "find", "knowledge_base": "RAW_TABLE"},
        ):
            assert await boundary_outcome(args) in {
                "invalid_request",
                "forbidden",
            }
        assert await boundary_outcome({"query": "x" * 100001}) == "invalid_request"
        embedding_failure = await client.call_tool(
            "search_knowledge", {"query": "embedding-fail", "knowledge_base": "docs"}
        )
        assert embedding_failure.structured_content["outcome"] == "backend_error"
        assert "safe embedding failure" not in str(embedding_failure.structured_content)
        retrieval_failure = await client.call_tool(
            "search_knowledge", {"query": "find", "knowledge_base": "fail"}
        )
        assert retrieval_failure.structured_content["outcome"] == "backend_error"
        assert "safe oracle failure" not in str(retrieval_failure.structured_content)
        rerank_failure = await client.call_tool(
            "search_knowledge", {"query": "rerank-fail", "knowledge_base": "docs"}
        )
        assert rerank_failure.structured_content["outcome"] == "success"
        assert rerank_failure.structured_content["reranking_status"] == "failed"
        assert rerank_failure.structured_content["evidence"]
        knowledge_bases = await client.call_tool("list_knowledge_bases", {})
        documents = await client.call_tool("list_documents", {"knowledge_base": "docs"})
        return {
            "names": names,
            "schemas": schemas,
            "protocol": protocol,
            "instructions": instructions,
            "success": success.structured_content,
            "empty": empty.structured_content,
            "forbidden": forbidden.structured_content,
            "invalid": invalid.structured_content,
            "embedding_failure": embedding_failure.structured_content,
            "retrieval_failure": retrieval_failure.structured_content,
            "rerank_failure": rerank_failure.structured_content,
            "knowledge_bases": knowledge_bases.structured_content,
            "documents": documents.structured_content,
        }


def test_stdio_profile_uses_real_child_process():
    script = str(Path(__file__).parents[1] / "fixtures" / "oracle_knowledge_stdio_server.py")
    env = {"PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"}
    snapshot = asyncio.run(
        _contract(
            Client(
                PythonStdioTransport(
                    script_path=script, python_cmd=sys.executable, cwd=str(Path.cwd()), env=env
                )
            )
        )
    )
    assert snapshot["success"]["evidence"]
    assert snapshot["knowledge_bases"]["knowledge_bases"]
    assert snapshot["documents"]["documents"]


def test_http_profile_matches_real_stdio_contract():
    script = str(Path(__file__).parents[1] / "fixtures" / "oracle_knowledge_stdio_server.py")
    env = {"PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"}
    probe = _Probe(True)
    server = create_fixture_server(readiness_probe=probe)
    app = server.http_app(transport="streamable-http")

    def factory(**kwargs):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", **kwargs
        )

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http_client:
                live = await http_client.get("/health/live")
                assert live.status_code == 200
                assert probe.calls == 0
                ready = await http_client.get("/health/ready")
                assert ready.status_code == 200
                assert ready.json() == {"status": "ready", "error": None}
            http_snapshot = await _contract(
                Client(StreamableHttpTransport("http://test/mcp", httpx_client_factory=factory))
            )
        stdio_snapshot = await _contract(
            Client(
                PythonStdioTransport(
                    script_path=script,
                    python_cmd=sys.executable,
                    cwd=str(Path.cwd()),
                    env=env,
                )
            )
        )
        failing_probe = _Probe(False, "password=deployment-secret DSN=oracle://secret")
        failing_server = create_fixture_server(readiness_probe=failing_probe)
        failing_app = failing_server.http_app(transport="streamable-http")
        async with failing_app.router.lifespan_context(failing_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=failing_app), base_url="http://test"
            ) as http_client:
                failed = await http_client.get("/health/ready")
                assert failed.status_code == 503
                assert failed.json() == {"status": "not_ready", "error": "service unavailable"}
                assert "deployment-secret" not in failed.text
        return http_snapshot, stdio_snapshot

    http_snapshot, stdio_snapshot = asyncio.run(run())
    assert http_snapshot == stdio_snapshot
