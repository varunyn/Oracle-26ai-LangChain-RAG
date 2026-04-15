import asyncio
import json
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import cast

import httpx
from httpx import ASGITransport

from api.main import app
from tests.unit_tests.test_streaming_utils import parse_sse_stream

THREAD_ID = "test-thread-stream"


def _stream_payload(messages: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    input_payload: dict[str, object] = {"messages": messages}
    input_payload.update(kwargs)
    return {"assistant_id": "mcp_agent_executor", "input": input_payload}


def _parse_values_events(chunks: list[bytes]) -> list[dict[str, object]]:
    payloads = list(parse_sse_stream(iter(chunks)))
    return [cast(dict[str, object], json.loads(payload)) for payload in payloads]


def _last_assistant(events: list[dict[str, object]]) -> dict[str, object]:
    assert events, "Expected at least one values event"
    messages = cast(list[dict[str, object]], events[-1].get("messages") or [])
    assert messages, "Expected at least one message in final values payload"
    assistant = messages[-1]
    assert assistant.get("type") == "ai"
    return assistant


class StubGraph:
    async def stream_chat(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
        _ = kwargs
        yield {"type": "text", "delta": "Hello from stub values stream. This is deterministic."}
        yield {
            "type": "references",
            "data": {
                "standalone_question": "Hello?",
                "citations": [{"source": "Doc1", "page": 1}],
                "reranker_docs": [{"page_content": "Example text", "metadata": {"id": "d1"}}],
                "context_usage": {
                    "tokens": 123,
                    "prompt_tokens": 12,
                    "completion_tokens": 111,
                },
            },
        }


class StubGraphWithInterpreterLeakAttempt:
    async def stream_chat(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
        _ = kwargs
        yield {"type": "text", "delta": "- Navigate to the Visual Applications page. [1]\n- Click New. [1]"}
        yield {
            "type": "references",
            "data": {
                "standalone_question": None,
                "citations": [{"source": "Doc1", "page": 1}],
                "reranker_docs": [{"page_content": "Example text", "metadata": {"id": "d1"}}],
                "context_usage": {"tokens": 22},
            },
        }


class StubRuntimeStreamService:
    async def stream_chat(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
        _ = kwargs
        yield {"type": "text", "delta": "Hello from new runtime stream."}
        yield {
            "type": "references",
            "data": {
                "standalone_question": "Hello?",
                "citations": [{"source": "Doc1", "page": 1}],
                "reranker_docs": [{"page_content": "Example text", "metadata": {"id": "d1"}}],
                "context_usage": {"tokens": 99},
            },
        }


class StubRuntimeStreamServiceWithDecimal:
    async def stream_chat(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
        _ = kwargs
        yield {"type": "text", "delta": "Hello from decimal runtime stream."}
        yield {
            "type": "references",
            "data": {
                "standalone_question": "Hello?",
                "citations": [{"source": "Doc1", "page": 1, "score": Decimal("0.75")}],
                "reranker_docs": [{"page_content": "Example text", "metadata": {"score": Decimal("0.5")}}],
                "context_usage": {"tokens": Decimal("99")},
            },
        }


def test_values_stream_happy_path() -> None:
    from api.dependencies import get_graph_service

    app.dependency_overrides[get_graph_service] = lambda: StubGraph()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = _stream_payload([{"type": "human", "content": "Hello"}])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST",
                f"/api/langgraph/threads/{THREAD_ID}/runs/stream",
                headers=headers,
                json=payload,
            ) as response:
                assert response.status_code == 200
                assert response.headers.get("content-type", "").startswith("text/event-stream")
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        events = _parse_values_events(chunks)
        assistant = _last_assistant(events)
        refs = cast(dict[str, object], assistant.get("response_metadata") or {})
        assert refs.get("standalone_question") == "Hello?"
        assert cast(list[dict[str, object]], refs.get("citations") or [])[0].get("source") == "Doc1"
        assert cast(dict[str, object], refs.get("context_usage") or {}).get("tokens") == 123
        assert "mcp_used" not in refs

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_prefers_service_level_stream_chat() -> None:
    from api.dependencies import get_graph_service

    app.dependency_overrides[get_graph_service] = lambda: StubRuntimeStreamService()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = _stream_payload([{"type": "human", "content": "Hello"}])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST",
                f"/api/langgraph/threads/{THREAD_ID}/runs/stream",
                headers=headers,
                json=payload,
            ) as response:
                assert response.status_code == 200
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        events = _parse_values_events(chunks)
        assistant = _last_assistant(events)
        assert assistant.get("content") == "Hello from new runtime stream."
        refs = cast(dict[str, object], assistant.get("response_metadata") or {})
        assert refs.get("standalone_question") == "Hello?"

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_sanitizes_decimal_in_references() -> None:
    from api.dependencies import get_graph_service

    app.dependency_overrides[get_graph_service] = lambda: StubRuntimeStreamServiceWithDecimal()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = _stream_payload([{"type": "human", "content": "Hello"}])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST",
                f"/api/langgraph/threads/{THREAD_ID}/runs/stream",
                headers=headers,
                json=payload,
            ) as response:
                assert response.status_code == 200
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        events = _parse_values_events(chunks)
        assistant = _last_assistant(events)
        refs = cast(dict[str, object], assistant.get("response_metadata") or {})
        assert cast(dict[str, object], refs.get("context_usage") or {}).get("tokens") == 99.0
        citation = cast(list[dict[str, object]], refs.get("citations") or [])[0]
        assert citation.get("score") == 0.75

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_error_on_empty_message() -> None:
    from api.dependencies import get_graph_service

    class EmptyErrorStreamService:
        async def stream_chat(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
            _ = kwargs
            yield {"type": "references", "data": {"error": "Empty or missing user message"}}

    app.dependency_overrides[get_graph_service] = lambda: EmptyErrorStreamService()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = _stream_payload([{"type": "human", "content": ""}])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST",
                f"/api/langgraph/threads/{THREAD_ID}/runs/stream",
                headers=headers,
                json=payload,
            ) as response:
                assert response.status_code == 200
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        events = _parse_values_events(chunks)
        assistant = _last_assistant(events)
        refs = cast(dict[str, object], assistant.get("response_metadata") or {})
        assert isinstance(refs.get("error"), str)
        assert refs.get("error")

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_emits_generic_error_on_exception() -> None:
    from api.dependencies import get_graph_service

    class RaisingGraph:
        async def stream_chat(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
            _ = kwargs
            raise Exception("SECRET_DO_NOT_LEAK")
            yield {"type": "text", "delta": ""}  # pragma: no cover

    app.dependency_overrides[get_graph_service] = lambda: RaisingGraph()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = _stream_payload([{"type": "human", "content": "Hello"}])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST",
                f"/api/langgraph/threads/{THREAD_ID}/runs/stream",
                headers=headers,
                json=payload,
            ) as response:
                assert response.status_code == 200
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        payloads = list(parse_sse_stream(iter(chunks)))
        concatenated = "\n".join(payloads)
        assert "SECRET_DO_NOT_LEAK" not in concatenated
        events = [cast(dict[str, object], json.loads(payload)) for payload in payloads]
        assistant = _last_assistant(events)
        refs = cast(dict[str, object], assistant.get("response_metadata") or {})
        assert refs.get("error") == "Internal server error"

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_does_not_leak_followup_interpreter_json() -> None:
    from api.dependencies import get_graph_service

    app.dependency_overrides[get_graph_service] = lambda: StubGraphWithInterpreterLeakAttempt()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = {
            "assistant_id": "mcp_agent_executor",
            "input": {
                "messages": [{"type": "human", "content": "give me that answer in bullet points"}],
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST",
                f"/api/langgraph/threads/{THREAD_ID}/runs/stream",
                headers=headers,
                json=payload,
            ) as response:
                assert response.status_code == 200
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        payloads = list(parse_sse_stream(iter(chunks)))
        concatenated = "\n".join(payloads)
        assert '"intent":"reformat"' not in concatenated
        assert "Provide the previous answer in bullet points" not in concatenated
        events = [cast(dict[str, object], json.loads(payload)) for payload in payloads]
        assistant = _last_assistant(events)
        assert isinstance(assistant.get("content"), str) and assistant.get("content")

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
