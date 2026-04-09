import asyncio
import json
from types import SimpleNamespace
from typing import cast

import httpx
from httpx import ASGITransport
from pytest import MonkeyPatch

from api.main import app
from tests.test_streaming_utils import parse_sse_stream


def test_smoke_mcp_stream_includes_mcp_used(monkeypatch: MonkeyPatch) -> None:
    import api.routes.chat as chat
    from api.deps.request import get_graph_service

    class StubGraph:
        async def astream(
            self,
            _state: object,
            _run_config: dict[str, object],
            *,
            _stream_mode: list[str] | None = None,
        ):
            yield (
                "updates",
                {"DraftAnswer": {"final_answer": "Stub MCP answer."}},
            )

        async def get_state(self, _run_config: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(
                values={
                    "standalone_question": "List available tools",
                    "final_answer": "Stub MCP answer.",
                    "citations": [],
                    "reranker_docs": [],
                    "context_usage": {"tokens": 42},
                    "mcp_used": True,
                    "mcp_tools_used": ["call_tool_chain"],
                    "error": None,
                }
            )

    hook_calls: list[tuple[str, dict[str, object]]] = []

    async def _record_register_tools_for_run_async(
        user_request: str, run_config: dict[str, object]
    ) -> None:
        hook_calls.append((user_request, run_config))

    monkeypatch.setattr(
        chat, "register_tools_for_run_async", _record_register_tools_for_run_async, raising=False
    )
    app.dependency_overrides[get_graph_service] = lambda: StubGraph()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": "List available tools"}],
            "stream": True,
            "mode": "mixed",
        }
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
        assert payloads[-1] == "[DONE]"
        events = [cast(dict[str, object], json.loads(p)) for p in payloads if p != "[DONE]"]
        data_event = next(e for e in events if e.get("type") == "data-references")
        data = cast(dict[str, object], data_event.get("data") or {})
        assert data.get("mcp_used") is True
        assert data.get("mcp_tools_used") == ["call_tool_chain"]
        assert cast(dict[str, object], data.get("context_usage") or {}).get("tokens") == 42
        assert len(hook_calls) == 1
        hook_user_request, _hook_run_config = hook_calls[0]
        assert hook_user_request == "List available tools"

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_smoke_rag_non_stream_includes_citations(monkeypatch: MonkeyPatch) -> None:
    import api.routes.chat as chat
    from api.deps.request import get_graph_service

    class StubGraph:
        def invoke(self, _state: object, _run_config: dict[str, object]) -> dict[str, object]:
            return {
                "final_answer": "Stub RAG answer.",
                "standalone_question": "What is Oracle 23AI?",
                "citations": [{"source": "Doc1", "page": "1"}],
                "reranker_docs": [{"page_content": "Example text", "metadata": {"id": "d1"}}],
                "context_usage": {"tokens": 7},
                "mcp_used": False,
                "mcp_tools_used": [],
                "error": None,
            }

    hook_calls: list[tuple[str, dict[str, object]]] = []

    def _record_register_tools_for_run(user_request: str, run_config: dict[str, object]) -> None:
        hook_calls.append((user_request, run_config))

    monkeypatch.setattr(
        chat, "register_tools_for_run", _record_register_tools_for_run, raising=False
    )
    app.dependency_overrides[get_graph_service] = lambda: StubGraph()

    async def run() -> None:
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": "What is Oracle 23AI?"}],
            "stream": False,
            "mode": "rag",
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post("/api/chat", headers=headers, json=payload)
        assert response.status_code == 200
        body = cast(dict[str, object], response.json())
        citations = cast(list[dict[str, object]], body.get("citations") or [])
        assert isinstance(citations, list)
        assert citations
        context_usage = cast(dict[str, object], body.get("context_usage") or {})
        assert context_usage.get("tokens") == 7
        assert len(hook_calls) == 1
        hook_user_request, _hook_run_config = hook_calls[0]
        assert hook_user_request == "What is Oracle 23AI?"

    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
