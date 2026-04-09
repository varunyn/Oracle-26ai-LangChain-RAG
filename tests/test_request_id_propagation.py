import asyncio
from typing import cast

import httpx
from _pytest.monkeypatch import MonkeyPatch
from httpx import ASGITransport

from api.main import app


def test_request_id_propagates_into_to_thread(monkeypatch: MonkeyPatch):
    """Ensure X-Request-ID bound by middleware is visible inside the worker thread.

    We stub run_rag_and_get_answer to return the currently bound request_id as the answer.
    The non-stream /api/chat handler executes the work via asyncio.to_thread with an
    explicit REQUEST_ID_CTX binding, so the stub should observe the same request ID.
    """

    def _stub_run_rag_and_get_answer(*_args: object, **_kwargs: object) -> tuple[
        str,
        str | None,
        str | None,
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, object] | None,
        bool,
        list[str],
    ]:
        from src.rag_agent.core.logging import get_request_id

        rid = get_request_id() or "-"
        # Return rid as the answer; rest are minimal defaults
        return rid, None, None, [], [], None, False, []

    monkeypatch.setattr(
        "api.routes.chat.run_rag_and_get_answer", _stub_run_rag_and_get_answer, raising=True
    )

    async def run():
        headers = {
            "Content-Type": "application/json",
            # Set a deterministic request ID to verify propagation
            "X-Request-ID": "req-test-12345",
        }
        payload = {
            "messages": [
                {
                    "role": "user",
                    # Omit content intentionally to ensure parts->content transform still applies
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
            body = cast(dict[str, object], resp.json())

        # Accept either top-level content or choices[0].message.content
        observed: str | None = None
        content_top_val = body.get("content")
        if isinstance(content_top_val, str):
            observed = content_top_val
        if observed is None:
            choices_raw = body.get("choices")
            if isinstance(choices_raw, list) and choices_raw:
                choices_list = cast(list[object], choices_raw)
                first_item: object = choices_list[0]
                if isinstance(first_item, dict):
                    first_dict = cast(dict[str, object], first_item)
                    msg_obj = first_dict.get("message")
                    if isinstance(msg_obj, dict):
                        msg_dict = cast(dict[str, object], msg_obj)
                        content_val = msg_dict.get("content")
                        if isinstance(content_val, str):
                            observed = content_val
        assert observed == "req-test-12345"

    asyncio.run(run())
