import asyncio

import httpx
from httpx import ASGITransport
from langchain_core.messages import AIMessage

from api.main import app


def test_suggestions_endpoint_uses_sync_llm_invoke(monkeypatch) -> None:
    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(content='["What can I customize next?", "Can I create multiple apps?"]')

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            raise RuntimeError("ainvoke should not be used for suggestions")

    monkeypatch.setattr("api.routes.suggestions.get_llm", lambda **kwargs: FakeLLM())

    async def run() -> None:
        payload = {"last_message": "You can create a visual application from App Builder."}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post("/api/suggestions", json=payload)
            assert response.status_code == 200
            body = response.json()
            assert body["suggestions"] == [
                "What can I customize next?",
                "Can I create multiple apps?",
            ]

    asyncio.run(run())


def test_suggestions_endpoint_returns_empty_when_model_output_is_invalid(monkeypatch) -> None:
    class FakeLLM:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(content="not json")

    monkeypatch.setattr("api.routes.suggestions.get_llm", lambda **kwargs: FakeLLM())

    async def run() -> None:
        payload = {
            "last_user_message": "How do I create an Oracle Visual Builder app?",
            "last_message": "You can create a visual application from App Builder.",
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post("/api/suggestions", json=payload)
            assert response.status_code == 200
            assert response.json()["suggestions"] == []

    asyncio.run(run())
