import asyncio
from typing import cast

import httpx
from httpx import ASGITransport

from api.main import app
from src.rag_agent.runtime.streaming import v3_raw_event


class _StreamingStubMixin:
    async def get_state(self, _run_config: dict[str, object]) -> object:
        return type("StateSnapshot", (), {"values": {}})()

    async def astream_events(
        self,
        input_payload: dict[str, object],
        *,
        config: dict[str, object],
        version: str,
    ):
        assert version == "v3"
        configurable = cast(dict[str, object], config.get("configurable") or {})
        result = await self.run_chat(
            messages=cast(list[object], input_payload.get("messages") or []),
            model_id=cast(str | None, configurable.get("model_id")),
            thread_id=cast(str | None, configurable.get("thread_id")),
            session_id=cast(str | None, configurable.get("session_id")),
            collection_name=cast(str | None, configurable.get("collection_name")),
            enable_reranker=cast(bool | None, configurable.get("enable_reranker")),
            enable_tracing=cast(bool | None, configurable.get("enable_tracing")),
            mode=cast(str | None, configurable.get("mode")),
            mcp_server_keys=cast(list[str] | None, configurable.get("mcp_server_keys")),
            stream=True,
        )
        yield v3_raw_event(
            method="messages",
            data=(
                {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": result.get("final_answer") or ""},
                },
                {"langgraph_node": "test"},
            ),
        )
        yield v3_raw_event(
            method="custom",
            data={
                "type": "references",
                "data": {
                    "citations": result.get("citations") or [],
                    "reranker_docs": result.get("reranker_docs") or [],
                },
            },
        )


def test_chat_stream_direct_mode_uses_oci_direct_agent(monkeypatch):
    from api.deps.request import get_graph_service

    class StubAgentService(_StreamingStubMixin):
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
                "/api/langgraph/threads/thread-oci-direct/runs/stream",
                headers=headers,
                json=payload,
            )
            assert resp.status_code == 200
            body = b"".join([chunk async for chunk in resp.aiter_bytes()])
            assert b"You can create a visual application from the Oracle APEX App Builder." in body

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_chat_stream_rag_mode_uses_oci_rag_runtime(monkeypatch):
    from api.deps.request import get_graph_service

    class StubAgentService(_StreamingStubMixin):
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
                "/api/langgraph/threads/thread-oci-rag/runs/stream",
                headers=headers,
                json=payload,
            )
            assert resp.status_code == 200
            body = b"".join([chunk async for chunk in resp.aiter_bytes()])
            assert b"Oracle 23ai introduces AI Vector Search. [1]" in body
            assert b"Doc1" in body

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
