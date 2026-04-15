import asyncio
from typing import cast

import httpx
from httpx import ASGITransport

from api.main import app


def test_chat_nonstream_direct_mode_uses_oci_direct_agent(monkeypatch):
    from api.dependencies import get_graph_service

    class StubAgentService:
        async def run_chat(
            self,
            *,
            messages: list[object],
            model_id: str | None,
            thread_id: str | None,
            session_id: str | None,
            collection_name: str | None,
            enable_reranker: bool | None,
            enable_tracing: bool | None,
            mode: str | None,
            mcp_server_keys: list[str] | None,
            stream: bool,
        ) -> dict[str, object]:
            _ = (
                model_id,
                thread_id,
                session_id,
                collection_name,
                enable_reranker,
                enable_tracing,
                mcp_server_keys,
                stream,
            )
            assert mode == "direct"
            assert cast(dict[str, object], messages[-1]).get("content") == "How can I create visual application?"
            return {
                "final_answer": "You can create a visual application from the Oracle APEX App Builder.",
                "error": None,
                "standalone_question": None,
                "citations": [],
                "reranker_docs": [],
                "context_usage": None,
                "mcp_used": False,
                "mcp_tools_used": [],
            }

    app.dependency_overrides[get_graph_service] = lambda: StubAgentService()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = {
            "assistant_id": "mcp_agent_executor",
            "input": {
                "messages": [{"type": "human", "content": "How can I create visual application?"}],
                "mode": "direct",
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/api/langgraph/threads/thread-oci-direct/runs",
                headers=headers,
                json=payload,
            )
            assert resp.status_code == 200
            body = cast(dict[str, object], resp.json())
            output = cast(dict[str, object], body.get("output") or {})
            assert (
                output.get("content")
                == "You can create a visual application from the Oracle APEX App Builder."
            )

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_chat_nonstream_rag_mode_uses_oci_rag_runtime(monkeypatch):
    from api.dependencies import get_graph_service

    class StubAgentService:
        async def run_chat(
            self,
            *,
            messages: list[object],
            model_id: str | None,
            thread_id: str | None,
            session_id: str | None,
            collection_name: str | None,
            enable_reranker: bool | None,
            enable_tracing: bool | None,
            mode: str | None,
            mcp_server_keys: list[str] | None,
            stream: bool,
        ) -> dict[str, object]:
            _ = (
                model_id,
                thread_id,
                session_id,
                collection_name,
                enable_reranker,
                enable_tracing,
                mcp_server_keys,
                stream,
            )
            assert mode == "rag"
            assert cast(dict[str, object], messages[-1]).get("content") == "What is Oracle 23AI?"
            return {
                "final_answer": "Oracle 23ai introduces AI Vector Search. [1]",
                "error": None,
                "standalone_question": "What is Oracle 23AI?",
                "citations": [{"source": "Doc1", "page": "1"}],
                "reranker_docs": [
                    {
                        "page_content": "Oracle Database 23ai introduces AI Vector Search.",
                        "metadata": {"source": "Doc1", "page": "1"},
                    }
                ],
                "context_usage": None,
                "mcp_used": False,
                "mcp_tools_used": [],
            }

    app.dependency_overrides[get_graph_service] = lambda: StubAgentService()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = {
            "assistant_id": "mcp_agent_executor",
            "input": {
                "messages": [{"type": "human", "content": "What is Oracle 23AI?"}],
                "mode": "rag",
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/api/langgraph/threads/thread-oci-rag/runs",
                headers=headers,
                json=payload,
            )
            assert resp.status_code == 200
            body = cast(dict[str, object], resp.json())
            output = cast(dict[str, object], body.get("output") or {})
            assert output.get("content") == "Oracle 23ai introduces AI Vector Search. [1]"
            assert output.get("citations") == [{"source": "Doc1", "page": "1"}]

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
