import asyncio
from typing import cast

import httpx
from _pytest.monkeypatch import MonkeyPatch
from httpx import ASGITransport

from api.main import app


def test_chat_validation_errors_return_4xx_json():
    async def run():
        headers = {"Content-Type": "application/json"}
        invalid_payloads: list[dict[str, object]] = [
            {},
            {"messages": "not-a-list"},
            {"messages": []},
            {"messages": [{"role": "assistant", "content": "hi"}]},
            {"messages": [{"role": "system", "content": "hi"}]},
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            },
        ]
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            for payload in invalid_payloads:
                resp = await client.post("/api/chat", headers=headers, json=payload)
                # Our exception handler returns 422 with shaped JSON for validation errors
                assert resp.status_code == 422
                ctype: str = resp.headers.get("content-type", "") or ""
                assert "application/json" in ctype
                body = cast(dict[str, object], resp.json())
                # Structural invariants: keys and types
                detail_val = body.get("detail")
                assert isinstance(detail_val, str)
                errors_obj = body.get("errors")
                assert isinstance(errors_obj, list)
                errors_list = cast(list[object], errors_obj)
                if errors_list:
                    first = errors_list[0]
                    if isinstance(first, dict):
                        first_dict = cast(dict[str, object], first)
                        # Avoid brittle exact messages; just check typical Pydantic error fields exist
                        for k in ("loc", "msg", "type"):
                            assert k in first_dict

    asyncio.run(run())


def test_chat_nonstream_json_includes_expected_fields(monkeypatch: MonkeyPatch):
    # Deterministic non-stream response from backend logic
    def _stub_run_rag_and_get_answer(*_args: object, **_kwargs: object) -> tuple[
        str,
        str | None,
        str | None,
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, int],
        bool,
        list[object],
    ]:
        answer = "Deterministic answer"
        err = None
        standalone = "Standalone Q?"
        citations: list[dict[str, object]] = [{"source": "Doc1", "page": "1"}]
        reranker_docs: list[dict[str, object]] = [
            {"page_content": "Para", "metadata": {"id": "d1"}}
        ]
        context_usage: dict[str, int] = {"tokens": 10, "prompt_tokens": 3, "completion_tokens": 7}
        mcp_used = False
        mcp_tools_used: list[object] = []
        return (
            answer,
            err,
            standalone,
            citations,
            reranker_docs,
            context_usage,
            mcp_used,
            mcp_tools_used,
        )

    monkeypatch.setattr(
        "api.routes.chat.run_rag_and_get_answer", _stub_run_rag_and_get_answer, raising=True
    )

    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {"messages": [{"role": "user", "content": "Hello"}], "stream": False}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/chat", headers=headers, json=payload)
            assert resp.status_code == 200
            ctype: str = resp.headers.get("content-type", "") or ""
            assert "application/json" in ctype
            body = cast(dict[str, object], resp.json())

        # Flexible assertions: either top-level content or choices[].message.content
        content_top_val = body.get("content")
        content_top = content_top_val if isinstance(content_top_val, str) else None

        content_from_choice: str | None = None
        choices_obj = body.get("choices")
        if isinstance(choices_obj, list):
            for item in cast(list[object], choices_obj):
                if isinstance(item, dict):
                    item_dict = cast(dict[str, object], item)
                    msg_obj = item_dict.get("message")
                    if isinstance(msg_obj, dict):
                        msg_dict = cast(dict[str, object], msg_obj)
                        content_val = msg_dict.get("content")
                        if isinstance(content_val, str):
                            content_from_choice = content_val
                            break

        assert (content_top == "Deterministic answer") or (
            content_from_choice == "Deterministic answer"
        )

        # When backend returns these, they should be present in JSON
        citations_obj = body.get("citations")
        if isinstance(citations_obj, list):
            for cit in cast(list[object], citations_obj):
                if isinstance(cit, dict):
                    assert "source" in cit and "page" in cit
                    break

        reranker_obj = body.get("reranker_docs")
        if isinstance(reranker_obj, list):
            for doc in cast(list[object], reranker_obj):
                if isinstance(doc, dict):
                    assert "page_content" in doc and "metadata" in doc
                    break

        cu_obj = body.get("context_usage")
        if isinstance(cu_obj, dict):
            for k in ("tokens", "prompt_tokens", "completion_tokens"):
                assert k in cu_obj

    asyncio.run(run())


def test_chat_nonstream_accepts_parts_without_content(monkeypatch: MonkeyPatch):
    # Deterministic non-stream response from backend logic
    def _stub_run_rag_and_get_answer(*_args: object, **_kwargs: object) -> tuple[
        str,
        str | None,
        str | None,
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, int],
        bool,
        list[object],
    ]:
        answer = "Deterministic answer"
        err = None
        standalone = "Standalone Q?"
        citations: list[dict[str, object]] = [{"source": "Doc1", "page": "1"}]
        reranker_docs: list[dict[str, object]] = [
            {"page_content": "Para", "metadata": {"id": "d1"}}
        ]
        context_usage: dict[str, int] = {"tokens": 10, "prompt_tokens": 3, "completion_tokens": 7}
        mcp_used = False
        mcp_tools_used: list[object] = []
        return (
            answer,
            err,
            standalone,
            citations,
            reranker_docs,
            context_usage,
            mcp_used,
            mcp_tools_used,
        )

    monkeypatch.setattr(
        "api.routes.chat.run_rag_and_get_answer", _stub_run_rag_and_get_answer, raising=True
    )

    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {
                    "role": "user",
                    "id": "x",
                    "parts": [{"type": "text", "text": "Hello"}],
                }
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/chat", headers=headers, json=payload)
            assert resp.status_code == 200
            ctype: str = resp.headers.get("content-type", "") or ""
            assert "application/json" in ctype
            body = cast(dict[str, object], resp.json())

        # Should still contain deterministic answer from stub
        content_top_val = body.get("content")
        content_top = content_top_val if isinstance(content_top_val, str) else None

        content_from_choice: str | None = None
        choices_obj = body.get("choices")
        if isinstance(choices_obj, list):
            for item in cast(list[object], choices_obj):
                if isinstance(item, dict):
                    item_dict = cast(dict[str, object], item)
                    msg_obj = item_dict.get("message")
                    if isinstance(msg_obj, dict):
                        msg_dict = cast(dict[str, object], msg_obj)
                        content_val = msg_dict.get("content")
                        if isinstance(content_val, str):
                            content_from_choice = content_val
                            break

        assert (content_top == "Deterministic answer") or (
            content_from_choice == "Deterministic answer"
        )

    asyncio.run(run())


def test_chat_nonstream_accepts_message_history(monkeypatch: MonkeyPatch):
    captured_messages: list[object] = []

    def _stub_run_rag_and_get_answer(*args: object, **_kwargs: object) -> tuple[
        str,
        str | None,
        str | None,
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, int],
        bool,
        list[object],
    ]:
        nonlocal captured_messages
        captured_messages = cast(list[object], args[0])
        return (
            "Deterministic answer",
            None,
            "Standalone Q?",
            [],
            [],
            {"tokens": 2, "prompt_tokens": 1, "completion_tokens": 1},
            False,
            [],
        )

    monkeypatch.setattr(
        "api.routes.chat.run_rag_and_get_answer", _stub_run_rag_and_get_answer, raising=True
    )

    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": "Always follow user constraints."},
                {"role": "user", "content": "Answer with a table when appropriate."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "What documents mention Oracle vector search?"},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/chat", headers=headers, json=payload)
            assert resp.status_code == 200

    asyncio.run(run())

    assert len(captured_messages) == 4


def test_chat_nonstream_mcp_answer_can_return_empty_citations(monkeypatch: MonkeyPatch):
    def _stub_run_rag_and_get_answer(*_args: object, **_kwargs: object) -> tuple[
        str,
        str | None,
        str | None,
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, int],
        bool,
        list[object],
    ]:
        return (
            "Tool result",
            None,
            "Standalone Q?",
            [],
            [],
            {"tokens": 6, "prompt_tokens": 2, "completion_tokens": 4},
            True,
            ["calculator.solve_equation"],
        )

    monkeypatch.setattr(
        "api.routes.chat.run_rag_and_get_answer", _stub_run_rag_and_get_answer, raising=True
    )

    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": "Are there complex roots?"}],
            "stream": False,
            "mode": "mixed",
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/chat", headers=headers, json=payload)
            assert resp.status_code == 200
            body = cast(dict[str, object], resp.json())

        assert body.get("content") == "Tool result"
        assert body.get("citations") == []

    asyncio.run(run())


def test_legacy_mcp_chat_nonstream_returns_unavailable_error():
    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/mcp/chat", headers=headers, json=payload)
            assert resp.status_code == 200
            ctype: str = resp.headers.get("content-type", "") or ""
            assert "application/json" in ctype
            body = cast(dict[str, object], resp.json())

        assert body == {"error": "MCP integration not available"}

    asyncio.run(run())


def test_legacy_mcp_chat_stream_returns_unavailable_error_sse():
    async def run():
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post("/api/mcp/chat", headers=headers, json=payload)
            assert resp.status_code == 200
            ctype: str = resp.headers.get("content-type", "") or ""
            assert "text/event-stream" in ctype
            body = await resp.aread()

        text_body = body.decode("utf-8")
        assert '"error": "MCP integration not available"' in text_body
        assert "data: [DONE]" not in text_body

    asyncio.run(run())
