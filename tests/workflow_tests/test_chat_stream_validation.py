import asyncio
from collections.abc import AsyncIterator
from typing import cast

import httpx
from httpx import ASGITransport

from api.main import app
from src.rag_agent.runtime.streaming import v3_raw_event
from tests.workflow_tests.langgraph_protocol_helpers import command_envelope, run_v1_command_stream


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
            for index, payload in enumerate(invalid_payloads):
                resp = await client.post(
                    "/api/langgraph/threads/thread-validation/commands",
                    headers=headers,
                    json=command_envelope(payload, command_id=index),
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
            response, chunks = await run_v1_command_stream(
                client,
                thread_id="thread-1",
                payload=payload,
            )
            assert response.status_code == 200
            assert chunks

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


def test_langgraph_commands_endpoint_accepts_run_start_command_envelope(monkeypatch):
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
                        "delta": {"type": "text-delta", "text": "Command response"},
                    },
                    {"langgraph_node": "test"},
                ),
            )

    stub = StubStreamService()
    app.dependency_overrides[get_graph_service] = lambda: stub

    async def run():
        payload = {
            "id": 1,
            "method": "run.start",
            "params": {
                "input": {
                    "messages": [
                        {
                            "type": "human",
                            "content": "hello from command envelope",
                            "id": "msg-1",
                            "additional_kwargs": {},
                            "response_metadata": {},
                        }
                    ],
                    "model": "cohere.command-a-03-2025",
                    "session_id": "sess-command-envelope",
                    "collection_name": "default",
                    "enable_reranker": False,
                    "enable_tracing": True,
                    "mode": "rag",
                    "context": {"mode": "rag"},
                    "metadata": {"mode": "rag"},
                    "configurable": {"mode": "rag"},
                },
                "config": {"configurable": {"thread_id": "thread-command-envelope"}},
                "assistant_id": "mcp_agent_executor",
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/langgraph/threads/thread-command-envelope/commands",
                json=payload,
            )
            assert response.status_code == 200
            assert response.json()["type"] == "success"

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()

    assert stub.input_payload is not None
    messages = cast(list[dict[str, object]], stub.input_payload.get("messages") or [])
    assert messages == [
        {"id": "msg-1", "role": "user", "content": "hello from command envelope"}
    ]
    assert stub.config is not None
    configurable = cast(dict[str, object], stub.config.get("configurable") or {})
    assert configurable.get("mode") == "rag"


def test_langgraph_commands_endpoint_rejects_missing_command_method():
    async def run():
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/langgraph/threads/thread-missing-method/commands",
                json={
                    "id": 1,
                    "params": {
                        "input": {
                            "messages": [{"type": "human", "content": "hello"}],
                        },
                        "assistant_id": "mcp_agent_executor",
                    },
                },
            )
            assert response.status_code == 422
            body = cast(dict[str, object], response.json())
            assert isinstance(body.get("detail"), str)
            assert isinstance(body.get("errors"), list)

    asyncio.run(run())


def test_langgraph_v1_commands_return_json_and_stream_protocol_events(monkeypatch):
    from api.deps.request import get_graph_service

    class StubStreamService:
        async def get_state(self, _run_config: dict[str, object]) -> object:
            return type("StateSnapshot", (), {"values": {}})()

        async def astream_events(
            self,
            input_payload: dict[str, object],
            *,
            config: dict[str, object],
            version: str,
        ) -> AsyncIterator[dict[str, object]]:
            _ = input_payload
            _ = config
            assert version == "v3"
            yield v3_raw_event(
                method="messages",
                data=(
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "Protocol response"},
                    },
                    {"langgraph_node": "test"},
                ),
            )

    app.dependency_overrides[get_graph_service] = lambda: StubStreamService()

    async def run():
        thread_id = "thread-command-protocol"
        payload = {
            "id": 42,
            "method": "run.start",
            "params": {
                "input": {
                    "messages": [{"type": "human", "content": "hello protocol"}],
                    "mode": "direct",
                },
                "assistant_id": "mcp_agent_executor",
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            timeout=10.0,
        ) as client:
            command_response = await client.post(
                f"/api/langgraph/threads/{thread_id}/commands",
                json=payload,
            )
            assert command_response.status_code == 200
            assert "application/json" in command_response.headers["content-type"]
            command_body = cast(dict[str, object], command_response.json())
            assert command_body.get("type") == "success"
            assert command_body.get("id") == 42

            async with client.stream(
                "POST",
                f"/api/langgraph/threads/{thread_id}/stream/events",
                json={"channels": ["values", "lifecycle"], "depth": 1},
            ) as stream_response:
                stream_body = await stream_response.aread()
                assert stream_response.status_code == 200
                assert b"event: event" in stream_body
                assert b'"method":"values"' in stream_body
                assert b"Protocol response" in stream_body
                assert b'"method":"lifecycle"' in stream_body
                assert b'"event":"completed"' in stream_body

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_langgraph_completed_run_replay_coalesces_partial_values(monkeypatch):
    from api.deps.request import get_graph_service

    class StubStreamService:
        async def get_state(self, _run_config: dict[str, object]) -> object:
            return type("StateSnapshot", (), {"values": {}})()

        async def astream_events(
            self,
            input_payload: dict[str, object],
            *,
            config: dict[str, object],
            version: str,
        ) -> AsyncIterator[dict[str, object]]:
            _ = input_payload
            _ = config
            assert version == "v3"
            for text in ("Partial", " final"):
                yield v3_raw_event(
                    method="messages",
                    data=(
                        {
                            "event": "content-block-delta",
                            "delta": {"type": "text-delta", "text": text},
                        },
                        {"langgraph_node": "test"},
                    ),
                )

    app.dependency_overrides[get_graph_service] = lambda: StubStreamService()

    async def run():
        thread_id = "thread-command-replay-coalesced"
        payload = {
            "id": 43,
            "method": "run.start",
            "params": {
                "input": {
                    "messages": [{"type": "human", "content": "hello replay"}],
                    "mode": "direct",
                },
                "assistant_id": "mcp_agent_executor",
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            timeout=10.0,
        ) as client:
            command_response = await client.post(
                f"/api/langgraph/threads/{thread_id}/commands",
                json=payload,
            )
            assert command_response.status_code == 200

            async with client.stream(
                "POST",
                f"/api/langgraph/threads/{thread_id}/stream/events",
                json={"channels": ["values", "lifecycle"], "depth": 1},
            ) as stream_response:
                stream_body = await stream_response.aread()
                assert stream_response.status_code == 200
                assert stream_body.count(b'"method":"values"') == 1
                assert b"Partial final" in stream_body
                assert b'"event":"completed"' in stream_body

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
