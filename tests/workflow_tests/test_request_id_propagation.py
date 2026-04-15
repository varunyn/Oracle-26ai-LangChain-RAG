import asyncio
from typing import cast

import httpx
from _pytest.monkeypatch import MonkeyPatch
from httpx import ASGITransport

from api.main import app


def test_request_id_propagates_into_to_thread(monkeypatch: MonkeyPatch):
    """Ensure X-Request-ID bound by middleware is visible inside runtime invoke path."""

    async def _stub_runtime_invoke(self: object, **_kwargs: object) -> dict[str, object]:
        from src.rag_agent.utils.logging_config import get_request_id

        rid = get_request_id() or "-"
        return {
            "final_answer": rid,
            "error": None,
            "standalone_question": None,
            "citations": [],
            "reranker_docs": [],
            "context_usage": None,
        }

    monkeypatch.setattr(
        "src.rag_agent.runtime.agent.RuntimeAgent.invoke", _stub_runtime_invoke, raising=True
    )

    async def run():
        headers = {
            "Content-Type": "application/json",
            # Set a deterministic request ID to verify propagation
            "X-Request-ID": "req-test-12345",
        }
        payload = {
            "assistant_id": "mcp_agent_executor",
            "input": {
                "messages": [
                    {
                        "type": "human",
                        # Omit content intentionally to ensure parts->content transform still applies
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            },
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/api/langgraph/threads/thread-request-id/runs",
                headers=headers,
                json=payload,
            )
            assert resp.status_code == 200
            body = cast(dict[str, object], resp.json())

        output = cast(dict[str, object], body.get("output") or {})

        # Accept either top-level content or choices[0].message.content
        observed: str | None = None
        content_top_val = output.get("content")
        if isinstance(content_top_val, str):
            observed = content_top_val
        if observed is None:
            choices_raw = output.get("choices")
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
