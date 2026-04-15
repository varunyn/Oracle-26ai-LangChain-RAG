import asyncio
from typing import cast

import httpx
from _pytest.monkeypatch import MonkeyPatch
from httpx import ASGITransport

from api.main import app


def _run_payload(messages: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    input_payload: dict[str, object] = {"messages": messages}
    input_payload.update(kwargs)
    return {"assistant_id": "mcp_agent_executor", "input": input_payload}


def test_langgraph_run_validation_errors_return_4xx_json():
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
                    "/api/langgraph/threads/thread-validation/runs",
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


def test_langgraph_nonstream_json_includes_expected_fields(monkeypatch: MonkeyPatch):
    async def _stub_invoke(self: object, **_kwargs: object) -> dict[str, object]:
        return {
            "final_answer": "Deterministic answer",
            "error": None,
            "standalone_question": "Standalone Q?",
            "citations": [{"source": "Doc1", "page": "1"}],
            "reranker_docs": [{"page_content": "Para", "metadata": {"id": "d1"}}],
            "context_usage": {"tokens": 10, "prompt_tokens": 3, "completion_tokens": 7},
        }

    monkeypatch.setattr(
        "src.rag_agent.runtime.agent.RuntimeAgent.invoke", _stub_invoke, raising=True
    )

    async def run():
        headers = {"Content-Type": "application/json"}
        payload = _run_payload([{"type": "human", "content": "Hello"}])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/api/langgraph/threads/thread-fields/runs", headers=headers, json=payload
            )
            assert resp.status_code == 200
            body = cast(dict[str, object], resp.json())

        output = cast(dict[str, object], body.get("output") or {})
        assert output.get("content") == "Deterministic answer"
        assert output.get("standalone_question") == "Standalone Q?"
        citations = cast(list[dict[str, object]], output.get("citations") or [])
        assert citations and citations[0].get("source") == "Doc1"
        context_usage = cast(dict[str, object], output.get("context_usage") or {})
        assert context_usage.get("tokens") == 10

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


def test_langgraph_runs_accept_top_level_messages_and_context(monkeypatch: MonkeyPatch):
    captured_messages: list[object] = []
    captured_kwargs: dict[str, object] = {}

    async def _stub_invoke(self: object, **kwargs: object) -> dict[str, object]:
        nonlocal captured_messages
        nonlocal captured_kwargs
        captured_messages = cast(list[object], kwargs.get("messages", []))
        captured_kwargs = dict(kwargs)
        return {
            "final_answer": "Langgraph response",
            "error": None,
            "standalone_question": "Standalone",
            "citations": [],
            "reranker_docs": [],
            "context_usage": {"tokens": 4, "prompt_tokens": 2, "completion_tokens": 2},
        }

    monkeypatch.setattr(
        "src.rag_agent.runtime.agent.RuntimeAgent.invoke",
        _stub_invoke,
        raising=True,
    )

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
            resp = await client.post("/api/langgraph/threads/thread-1/runs", json=payload)
            assert resp.status_code == 200
            body = cast(dict[str, object], resp.json())
            assert body.get("thread_id") == "thread-1"
            assert "run_id" in body

    asyncio.run(run())

    assert captured_kwargs.get("mode") == "mixed"
    assert captured_kwargs.get("session_id") == "sess-1"
    assert captured_kwargs.get("collection_name") == "default"
    assert captured_kwargs.get("enable_reranker") is True
    assert len(captured_messages) == 1
    first = captured_messages[0]
    assert getattr(first, "role", None) == "user"
    assert getattr(first, "content", None) == "hello"
