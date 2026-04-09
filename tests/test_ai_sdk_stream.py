import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import cast

import httpx
from httpx import ASGITransport
from langchain_core.messages import AIMessage
from pytest import MonkeyPatch

from api.main import app
from tests.test_streaming_utils import parse_sse_stream


class StubGraph:
    async def astream(
        self,
        _state: object,
        _run_config: dict[str, object],
        *,
        _stream_mode: list[str] | None = None,
    ) -> AsyncIterator[object]:  # noqa: D401 - simple stub
        # Emit a single DraftAnswer update simulating streamed text
        yield (
            "updates",
            {
                "DraftAnswer": {
                    "final_answer": "Hello from stub AI SDK stream. This is deterministic."
                }
            },
        )

    async def get_state(
        self, _config: dict[str, object]
    ) -> SimpleNamespace:  # noqa: D401 - simple stub
        # Return a snapshot-like object with .values containing metadata fields
        return SimpleNamespace(
            values={
                "standalone_question": "Hello?",
                "final_answer": "Hello from stub AI SDK stream. This is deterministic.",
                "citations": [{"source": "Doc1", "page": 1}],
                "reranker_docs": [{"page_content": "Example text", "metadata": {"id": "d1"}}],
                "context_usage": {
                    "tokens": 123,
                    "prompt_tokens": 12,
                    "completion_tokens": 111,
                },
                "mcp_used": False,
                "mcp_tools_used": [],
                "error": None,
            }
        )


class StubGraphWithInterpreterLeakAttempt:
    async def astream(
        self,
        _state: object,
        _run_config: dict[str, object],
        *,
        _stream_mode: list[str] | None = None,
    ) -> AsyncIterator[object]:
        yield (
            "messages",
            (
                AIMessage(
                    content='{"intent":"reformat","standalone_question":null,"response_instruction":"Provide the previous answer in bullet points","reasoning":"User is asking for the same information again but in a different format."}'
                ),
                {"langgraph_node": "FollowUpInterpreter"},
            ),
        )
        yield (
            "updates",
            {
                "DraftAnswer": {
                    "final_answer": "- Navigate to the Visual Applications page. [1]\n- Click New. [1]"
                }
            },
        )

    async def get_state(self, _config: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            values={
                "standalone_question": None,
                "final_answer": "- Navigate to the Visual Applications page. [1]\n- Click New. [1]",
                "citations": [{"source": "Doc1", "page": 1}],
                "reranker_docs": [{"page_content": "Example text", "metadata": {"id": "d1"}}],
                "context_usage": {"tokens": 22},
                "mcp_used": False,
                "mcp_tools_used": [],
                "error": None,
            }
        )


def test_ai_sdk_stream_happy_path_parts_and_headers(monkeypatch: MonkeyPatch):
    import api.routes.chat as chat
    from api.dependencies import get_graph_service

    hook_calls: list[tuple[str, dict[str, object]]] = []

    async def _record_register_tools_for_run_async(
        user_request: str, run_config: dict[str, object]
    ) -> None:
        hook_calls.append((user_request, run_config))

    monkeypatch.setattr(
        chat, "register_tools_for_run_async", _record_register_tools_for_run_async, raising=False
    )

    # Override DI to use stub graph service
    app.dependency_overrides[get_graph_service] = lambda: StubGraph()

    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {"messages": [{"role": "user", "content": "Hello"}], "stream": True}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST", "/api/chat", headers=headers, json=payload
            ) as response:
                # Header should indicate AI SDK UI message stream
                assert response.status_code == 200
                assert response.headers.get("x-vercel-ai-ui-message-stream") == "v1"

                # Collect raw chunks and parse SSE frames (including [DONE])
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        payloads = list(parse_sse_stream(iter(chunks)))

        # Must terminate with [DONE]
        assert payloads[-1] == "[DONE]"

        # Parse JSON parts (exclude [DONE])
        events: list[dict[str, object]] = [json.loads(p) for p in payloads if p != "[DONE]"]
        types = [e.get("type") for e in events]

        # Ordering and presence assertions
        assert types[0] == "start"
        assert types[1] == "text-start"
        # At least one text-delta before text-end
        assert "text-delta" in types
        assert "text-end" in types
        first_delta_idx = types.index("text-delta")
        end_idx = types.index("text-end")
        assert first_delta_idx < end_idx
        # data-references present, legacy metadata not present
        assert "data-references" in types
        assert "metadata" not in types
        # finish present and should be last JSON event before [DONE]
        assert types[-1] == "finish"

        # Ensure a non-empty delta was emitted
        assert any(e.get("type") == "text-delta" and e.get("delta") for e in events)

        data_refs = next(e for e in events if e.get("type") == "data-references")
        data = cast(dict[str, object], data_refs.get("data") or {})
        assert data.get("standalone_question") == "Hello?"
        assert cast(list[dict[str, object]], data.get("citations") or [])[0].get("source") == "Doc1"
        assert cast(dict[str, object], data.get("context_usage") or {}).get("tokens") == 123
        assert "mcp_used" not in data

        assert len(hook_calls) == 1
        hook_user_request, _hook_run_config = hook_calls[0]
        assert hook_user_request == "Hello"

    try:
        asyncio.run(run())
    finally:
        # Clean up overrides
        app.dependency_overrides.clear()


def test_ai_sdk_stream_error_on_empty_message(monkeypatch: MonkeyPatch):
    import api.routes.chat as chat

    # Ensure no external effects if code path changes in future
    async def _noop_register_tools_for_run_async(
        _user_request: str, _run_config: dict[str, object]
    ) -> None:
        return None

    monkeypatch.setattr(
        chat, "register_tools_for_run_async", _noop_register_tools_for_run_async, raising=False
    )

    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {"messages": [{"role": "user", "content": ""}], "stream": True}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST", "/api/chat", headers=headers, json=payload
            ) as response:
                assert response.status_code == 200
                assert response.headers.get("x-vercel-ai-ui-message-stream") == "v1"

                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        payloads = list(parse_sse_stream(iter(chunks)))

        # Should emit an error part then [DONE]
        assert payloads[-1] == "[DONE]"
        first: dict[str, object] = cast(dict[str, object], json.loads(payloads[0]))
        assert first.get("type") == "error"
        err_text = first.get("errorText")
        assert isinstance(err_text, str) and err_text != ""

    asyncio.run(run())


def test_ai_sdk_stream_emits_generic_error_and_done_on_exception(monkeypatch: MonkeyPatch):
    """If the streaming path raises, client gets generic error and [DONE], no secret leak."""
    import api.routes.chat as chat
    from api.dependencies import get_graph_service

    # Stub that raises during streaming
    class RaisingGraph:
        def astream(
            self,
            _state: object,
            _run_config: dict[str, object],
            *,
            _stream_mode: list[str] | None = None,
        ) -> AsyncIterator[object]:
            class _Iter:
                def __aiter__(self) -> "_Iter":
                    return self

                async def __anext__(self) -> object:
                    raise Exception("SECRET_DO_NOT_LEAK")

            return _Iter()

        async def get_state(self, _run_config: dict[str, object]) -> SimpleNamespace:
            # Should not be called in error path
            return SimpleNamespace(values={})

    # No-op tool registration
    async def _noop_register_tools_for_run_async(
        _user_request: str, _run_config: dict[str, object]
    ) -> None:
        return None

    monkeypatch.setattr(
        chat, "register_tools_for_run_async", _noop_register_tools_for_run_async, raising=False
    )

    # Override DI to use RaisingGraph
    app.dependency_overrides[get_graph_service] = lambda: RaisingGraph()

    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {"messages": [{"role": "user", "content": "Hello"}], "stream": True}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST", "/api/chat", headers=headers, json=payload
            ) as response:
                assert response.status_code == 200
                assert response.headers.get("x-vercel-ai-ui-message-stream") == "v1"

                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        # Ensure no raw exception text appears in any payload
        payloads = list(parse_sse_stream(iter(chunks)))
        assert payloads[-1] == "[DONE]"
        concatenated = "\n".join(payloads)
        assert "SECRET_DO_NOT_LEAK" not in concatenated

        # Find the error event and check it is generic
        events: list[dict[str, object]] = [json.loads(p) for p in payloads if p != "[DONE]"]
        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, "Expected an error event"
        err = error_events[0]
        assert isinstance(err.get("errorText"), str)
        assert err.get("errorText") == "Internal server error"

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_ai_sdk_stream_does_not_leak_followup_interpreter_json(monkeypatch: MonkeyPatch):
    import api.routes.chat as chat
    from api.dependencies import get_graph_service

    async def _noop_register_tools_for_run_async(
        _user_request: str, _run_config: dict[str, object]
    ) -> None:
        return None

    monkeypatch.setattr(
        chat, "register_tools_for_run_async", _noop_register_tools_for_run_async, raising=False
    )
    app.dependency_overrides[get_graph_service] = lambda: StubGraphWithInterpreterLeakAttempt()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": "give me that answer in bullet points"}],
            "stream": True,
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with client.stream(
                "POST", "/api/chat", headers=headers, json=payload
            ) as response:
                assert response.status_code == 200
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)

        payloads = list(parse_sse_stream(iter(chunks)))
        concatenated = "\n".join(payloads)
        assert '"intent":"reformat"' not in concatenated
        assert "Provide the previous answer in bullet points" not in concatenated
        assert '"type": "text-start"' in concatenated
        assert '"type": "text-end"' in concatenated
        assert payloads[-1] == "[DONE]"

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
