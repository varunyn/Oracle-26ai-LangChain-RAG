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
    return [payload for event, payload in _parse_sse_events(chunks) if event == "values"]


def _parse_sse_events(chunks: list[bytes]) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    current_event = "message"
    for raw_line in b"".join(chunks).decode("utf-8").splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("event: "):
            current_event = line[7:]
            continue
        if line.startswith("data: "):
            events.append((current_event, cast(dict[str, object], json.loads(line[6:]))))
            current_event = "message"
    return events


def _last_assistant(events: list[dict[str, object]]) -> dict[str, object]:
    assert events, "Expected at least one values event"
    messages = cast(list[dict[str, object]], events[-1].get("messages") or [])
    assert messages, "Expected at least one message in final values payload"
    assistant = messages[-1]
    assert assistant.get("type") == "ai"
    return assistant


def _v3_raw_event(method: str, data: object) -> dict[str, object]:
    return {"type": "event", "method": method, "params": {"data": data}}


async def _yield_v3_events(
    runtime_events: list[dict[str, object]],
) -> AsyncIterator[dict[str, object]]:
    for event in runtime_events:
        event_type = event.get("type")
        if event_type == "text":
            yield _v3_raw_event(
                "messages",
                (
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": event.get("delta")},
                    },
                    {"langgraph_node": "test"},
                ),
            )
        elif event_type == "tool_event":
            yield _v3_raw_event("tool_calls", event.get("data") or {})
        elif event_type == "references":
            yield _v3_raw_event("custom", {"type": "references", "data": event.get("data") or {}})


class StubGraph:
    async def astream_events(
        self,
        input_payload: dict[str, object],
        *,
        config: dict[str, object],
        version: str,
    ) -> AsyncIterator[dict[str, object]]:
        _ = input_payload, config, version
        events: list[dict[str, object]] = [
            {"type": "text", "delta": "Hello from stub values stream. This is deterministic."},
            {
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
            },
        ]
        async for event in _yield_v3_events(events):
            yield event


class StubGraphWithInterpreterLeakAttempt:
    async def astream_events(
        self,
        input_payload: dict[str, object],
        *,
        config: dict[str, object],
        version: str,
    ) -> AsyncIterator[dict[str, object]]:
        _ = input_payload, config, version
        events: list[dict[str, object]] = [
            {
                "type": "text",
                "delta": "- Navigate to the Visual Applications page. [1]\n- Click New. [1]",
            },
            {
                "type": "references",
                "data": {
                    "standalone_question": None,
                    "citations": [{"source": "Doc1", "page": 1}],
                    "reranker_docs": [{"page_content": "Example text", "metadata": {"id": "d1"}}],
                    "context_usage": {"tokens": 22},
                },
            },
        ]
        async for event in _yield_v3_events(events):
            yield event


class StubRuntimeEventService:
    async def astream_events(
        self,
        input_payload: dict[str, object],
        *,
        config: dict[str, object],
        version: str,
    ) -> AsyncIterator[dict[str, object]]:
        _ = input_payload, config, version
        async for event in _yield_v3_events(self.events()):
            yield event

    def events(self) -> list[dict[str, object]]:
        return [
            {"type": "text", "delta": "Hello from new runtime stream."},
            {
                "type": "references",
                "data": {
                    "standalone_question": "Hello?",
                    "citations": [{"source": "Doc1", "page": 1}],
                    "reranker_docs": [{"page_content": "Example text", "metadata": {"id": "d1"}}],
                    "context_usage": {"tokens": 99},
                },
            },
        ]


class StubRuntimeEventServiceWithDecimal(StubRuntimeEventService):
    def events(self) -> list[dict[str, object]]:
        return [
            {"type": "text", "delta": "Hello from decimal runtime stream."},
            {
                "type": "references",
                "data": {
                    "standalone_question": "Hello?",
                    "citations": [{"source": "Doc1", "page": 1, "score": Decimal("0.75")}],
                    "reranker_docs": [
                        {"page_content": "Example text", "metadata": {"score": Decimal("0.5")}}
                    ],
                    "context_usage": {"tokens": Decimal("99")},
                },
            },
        ]


class StubRuntimeEventServiceWithToolProgress(StubRuntimeEventService):
    def events(self) -> list[dict[str, object]]:
        return [
            {
                "type": "tool_event",
                "data": {
                    "phase": "start",
                    "tool_name": "oic_LIST_DOCUMENTS",
                    "tool_run_id": "tool-call-1",
                    "args": {"folderName": "invoices"},
                },
            },
            {"type": "text", "delta": "Processing complete."},
            {
                "type": "references",
                "data": {
                    "standalone_question": "Review invoices",
                    "citations": [],
                    "reranker_docs": [],
                    "mcp_used": True,
                    "mcp_tools_used": ["oic_LIST_DOCUMENTS"],
                },
            },
        ]


class StubRuntimeV3EventService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def astream_events(
        self,
        input_payload: dict[str, object],
        *,
        config: dict[str, object],
        version: str,
    ) -> AsyncIterator[dict[str, object]]:
        self.calls.append({"input": input_payload, "config": config, "version": version})
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "data": (
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "Hello from v3"},
                    },
                    {"langgraph_node": "model"},
                )
            },
        }
        yield {
            "type": "event",
            "method": "messages",
            "params": {
                "data": (
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": " events."},
                    },
                    {"langgraph_node": "model"},
                )
            },
        }


def test_values_stream_happy_path() -> None:
    from api.deps.request import get_graph_service

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


def test_values_stream_uses_app_v3_event_stream() -> None:
    from api.deps.request import get_graph_service

    app.dependency_overrides[get_graph_service] = lambda: StubRuntimeEventService()

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


def test_values_stream_uses_v3_event_stream_when_available() -> None:
    from api.deps.request import get_graph_service

    stub = StubRuntimeV3EventService()
    app.dependency_overrides[get_graph_service] = lambda: stub

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

        assert stub.calls
        assert stub.calls[0]["version"] == "v3"
        config = cast(dict[str, object], stub.calls[0]["config"])
        assert cast(dict[str, object], config["configurable"])["thread_id"] == THREAD_ID
        events = _parse_values_events(chunks)
        assistant = _last_assistant(events)
        assert assistant.get("content") == "Hello from v3 events."

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_reads_v3_events_without_runtime_stream_bridge() -> None:
    from api.deps.request import get_graph_service

    stub = StubRuntimeV3EventService()
    app.dependency_overrides[get_graph_service] = lambda: stub

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
        assert assistant.get("content") == "Hello from v3 events."

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_logs_conversation_out(monkeypatch) -> None:
    from api.deps.request import get_graph_service

    captured: list[dict[str, object]] = []

    def fake_log_conversation_out(**kwargs: object) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(
        "api.routes.langgraph_server.log_conversation_out",
        fake_log_conversation_out,
        raising=False,
    )
    app.dependency_overrides[get_graph_service] = lambda: StubRuntimeEventService()

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
                async for _ in response.aiter_bytes():
                    pass

        assert captured == [
            {
                "final_answer": "Hello from new runtime stream.",
                "error": None,
                "mcp_used": None,
                "mcp_tools_used": None,
                "standalone_question": "Hello?",
            }
        ]

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_sanitizes_decimal_in_references() -> None:
    from api.deps.request import get_graph_service

    app.dependency_overrides[get_graph_service] = lambda: StubRuntimeEventServiceWithDecimal()

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


def test_values_stream_includes_tool_progress_events() -> None:
    from api.deps.request import get_graph_service

    app.dependency_overrides[get_graph_service] = lambda: StubRuntimeEventServiceWithToolProgress()

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
        progress = cast(list[dict[str, object]], refs.get("mcp_progress_events") or [])
        assert progress
        assert progress[0].get("phase") == "start"
        assert progress[0].get("tool_name") == "oic_LIST_DOCUMENTS"

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_stream_emits_sdk_tool_events_before_final_answer() -> None:
    from api.deps.request import get_graph_service

    app.dependency_overrides[get_graph_service] = lambda: StubRuntimeEventServiceWithToolProgress()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = _stream_payload(
            [{"type": "human", "content": "Hello"}], streamMode=["values", "tools"]
        )
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

        events = _parse_sse_events(chunks)
        tool_event_index = next(
            index for index, (event_name, _) in enumerate(events) if event_name == "tools"
        )
        final_answer_index = next(
            index
            for index, (event_name, payload) in enumerate(events)
            if event_name == "values"
            and "Processing complete."
            in str(cast(list[dict[str, object]], payload.get("messages") or [])[-1].get("content"))
        )
        event_payload = events[tool_event_index][1]
        assert tool_event_index < final_answer_index
        assert event_payload == {
            "event": "on_tool_start",
            "name": "oic_LIST_DOCUMENTS",
            "toolCallId": "tool-call-1",
            "input": {"folderName": "invoices"},
        }

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_values_stream_error_on_empty_message() -> None:
    from api.deps.request import get_graph_service

    class EmptyErrorStreamService:
        async def astream_events(
            self,
            input_payload: dict[str, object],
            *,
            config: dict[str, object],
            version: str,
        ) -> AsyncIterator[dict[str, object]]:
            _ = input_payload, config, version
            async for event in _yield_v3_events(
                [{"type": "references", "data": {"error": "Empty or missing user message"}}]
            ):
                yield event

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
    from api.deps.request import get_graph_service

    class RaisingGraph:
        async def astream_events(
            self,
            input_payload: dict[str, object],
            *,
            config: dict[str, object],
            version: str,
        ) -> AsyncIterator[dict[str, object]]:
            _ = input_payload, config, version
            raise Exception("SECRET_DO_NOT_LEAK")
            yield _v3_raw_event("messages", {})  # pragma: no cover

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
    from api.deps.request import get_graph_service

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
