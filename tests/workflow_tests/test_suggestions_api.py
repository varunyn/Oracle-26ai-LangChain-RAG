import asyncio

import httpx
from httpx import ASGITransport

from api.main import app
from api.routes import suggestions as suggestions_route


def test_suggestions_endpoint_uses_structured_agent_response(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLLM:
        pass

    class FakeAgent:
        def invoke(self, payload: dict[str, object], config: dict[str, object] | None = None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "structured_response": {
                    "suggestions": [
                        "Which Visual Builder template should I use?",
                        "What permissions are required first?",
                    ]
                }
            }

    def fake_create_agent(*, model: object, tools: list[object], system_prompt: str, response_format: object):
        captured["model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        captured["response_format"] = response_format
        return FakeAgent()

    fake_llm = FakeLLM()
    monkeypatch.setattr(suggestions_route, "get_llm", lambda **kwargs: fake_llm)
    monkeypatch.setattr(suggestions_route, "create_agent", fake_create_agent, raising=False)

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
            assert response.json()["suggestions"] == [
                "Which Visual Builder template should I use?",
                "What permissions are required first?",
            ]

    asyncio.run(run())
    assert captured["model"] is fake_llm
    assert captured["tools"] == []
    assert captured["response_format"].__name__ == "FollowUpSuggestions"
    assert "Latest assistant answer" in str(captured["payload"])


def test_suggestions_endpoint_uses_sync_agent_invoke(monkeypatch) -> None:
    class FakeLLM:
        pass

    class FakeAgent:
        def invoke(self, payload: dict[str, object], config: dict[str, object] | None = None):
            assert payload
            assert config
            return {
                "structured_response": {
                    "suggestions": ["What can I customize next?", "Can I create multiple apps?"]
                }
            }

        async def ainvoke(self, payload: dict[str, object]):
            _ = payload
            raise RuntimeError("ainvoke should not be used for suggestions")

    monkeypatch.setattr(suggestions_route, "get_llm", lambda **kwargs: FakeLLM())
    monkeypatch.setattr(
        suggestions_route,
        "create_agent",
        lambda **kwargs: FakeAgent(),
    )

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
        pass

    class FakeAgent:
        def invoke(self, payload: dict[str, object], config: dict[str, object] | None = None):
            assert payload
            assert config
            return {"messages": []}

    monkeypatch.setattr(suggestions_route, "get_llm", lambda **kwargs: FakeLLM())
    monkeypatch.setattr(suggestions_route, "create_agent", lambda **kwargs: FakeAgent())

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
