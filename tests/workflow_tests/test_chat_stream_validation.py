import asyncio
from collections.abc import AsyncIterator
from typing import cast

import httpx
from httpx import ASGITransport

from api.main import app
from src.rag_agent.runtime.streaming import v3_raw_event


def _run_payload(messages: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    input_payload: dict[str, object] = {"messages": messages}
    input_payload.update(kwargs)
    return {"assistant_id": "mcp_agent_executor", "input": input_payload}


def test_langgraph_stream_validation_errors_return_4xx_json():
    async def run():
        headers = {"Content-Type": "application/json"}
        invalid_payloads: list[dict[str, object]] = [
            {},
            {"assistant_id": "mcp_agent_executor"},
            {"input": {"messages": "not-a-list"}},
            {"input": {"messages": []}},
            {"input": {"messages": [{"type": "ai", "content": "hi"}]}},
        ]
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            for payload in invalid_payloads:
                resp = await client.post(
                    "/api/langgraph/threads/thread-validation/runs/stream",
                    headers=headers,
                    json=payload,
                )
                assert resp.status_code == 422
                ctype: str = resp.headers.get("content-type", "") or ""
                assert "application/json" in ctype
                body = cast(dict[str, object], resp.json())
                assert isinstance(body.get("detail"), str)
                assert isinstance(body.get("errors"), list)

    asyncio.run(run())


def test_langgraph_thread_create_returns_requested_id():
    async def run():
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/langgraph/threads", json={"thread_id": "thread-abc"})
            assert resp.status_code == 200
            body = cast(dict[str, object], resp.json())
            assert body.get("thread_id") == "thread-abc"

    asyncio.run(run())


def test_langgraph_non_stream_run_endpoint_is_not_registered():
    async def run():
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/api/langgraph/threads/thread-removed/runs",
                json=_run_payload([{"type": "human", "content": "Hello"}]),
            )
            assert resp.status_code == 404

    asyncio.run(run())


def test_langgraph_stream_accepts_top_level_messages_and_context(monkeypatch):
    from api.deps.request import get_graph_service

    class StubStreamService:
        def __init__(self) -> None:
            self.input_payload: dict[str, object] | None = None
            self.config: dict[str, object] | None = None

        async def get_state(self, _run_config: dict[str, object]) -> object:
            return type("StateSnapshot", (), {"values": {}})()

        async def astream_events(
            self,
            input_payload: dict[str, object],
            *,
            config: dict[str, object],
            version: str,
        ) -> AsyncIterator[dict[str, object]]:
            self.input_payload = input_payload
            self.config = config
            assert version == "v3"
            yield v3_raw_event(
                method="messages",
                data=(
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "Langgraph response"},
                    },
                    {"langgraph_node": "test"},
                ),
            )

    stub = StubStreamService()
    app.dependency_overrides[get_graph_service] = lambda: stub

    async def run():
        payload = {
            "messages": [{"type": "human", "content": "hello"}],
            "context": {
                "mode": "mixed",
                "session_id": "sess-1",
                "collection_name": "default",
                "enable_reranker": True,
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST",
                "/api/langgraph/threads/thread-1/runs/stream",
                json=payload,
            ) as response:
                assert response.status_code == 200
                async for _ in response.aiter_bytes():
                    pass

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()

    assert stub.config is not None
    configurable = cast(dict[str, object], stub.config.get("configurable") or {})
    assert configurable.get("mode") == "mixed"
    assert configurable.get("session_id") == "sess-1"
    assert configurable.get("collection_name") == "default"
    assert configurable.get("enable_reranker") is True
    assert stub.input_payload is not None
    messages = cast(list[dict[str, object]], stub.input_payload.get("messages") or [])
    assert messages == [{"role": "user", "content": "hello"}]
