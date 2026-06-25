import asyncio
from collections.abc import AsyncIterator

import httpx
from httpx import ASGITransport

from api.main import app
from src.rag_agent.runtime.streaming import v3_raw_event
from tests.workflow_tests.langgraph_protocol_helpers import run_v1_command_stream


def test_request_id_propagates_into_stream_runtime():
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
            _ = input_payload, config, version
            from src.rag_agent.utils.logging_config import get_request_id

            yield v3_raw_event(
                method="messages",
                data=(
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": get_request_id() or "-"},
                    },
                    {"langgraph_node": "test"},
                ),
            )

    app.dependency_overrides[get_graph_service] = lambda: StubStreamService()

    async def run() -> bytes:
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": "req-test-12345",
        }
        payload = {
            "assistant_id": "mcp_agent_executor",
            "input": {
                "messages": [
                    {
                        "type": "human",
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response, chunks = await run_v1_command_stream(
                client,
                thread_id="thread-request-id",
                headers=headers,
                payload=payload,
            )
            assert response.status_code == 200
        return b"".join(chunks)

    try:
        body = asyncio.run(run())
    finally:
        app.dependency_overrides.clear()

    assert b"req-test-12345" in body
