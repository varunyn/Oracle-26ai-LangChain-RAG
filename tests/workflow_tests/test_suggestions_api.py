import asyncio
from types import SimpleNamespace

import httpx
from httpx import ASGITransport
from langchain.agents.structured_output import ToolStrategy

from api.main import app
from api.routes import suggestions as suggestions_route


def test_suggestions_endpoint_uses_structured_agent_response(monkeypatch) -> None:
    captured: dict[str, object] = {}
    llm_kwargs: dict[str, object] = {}

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

    def fake_create_agent(
        *, model: object, tools: list[object], system_prompt: str, response_format: object
    ):
        captured["model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        captured["response_format"] = response_format
        return FakeAgent()

    fake_llm = FakeLLM()
    def fake_get_llm(**kwargs: object) -> FakeLLM:
        llm_kwargs.update(kwargs)
        return fake_llm

    monkeypatch.setattr(suggestions_route, "get_llm", fake_get_llm)
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
    assert llm_kwargs["max_tokens"] == 128
    assert captured["tools"] == []
    assert isinstance(captured["response_format"], ToolStrategy)
    assert captured["response_format"].schema.__name__ == "FollowUpSuggestions"
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


def test_suggestions_endpoint_uses_tool_strategy_without_default_model_retry(
    monkeypatch,
) -> None:
    llm_calls: list[str | None] = []
    agent_models: list[str | None] = []

    class FakeLLM:
        def __init__(self, model_id: str | None) -> None:
            self.model_id = model_id

    class FakeAgent:
        def invoke(self, payload: dict[str, object], config: dict[str, object] | None = None):
            assert payload
            assert config
            return {
                "structured_response": {
                    "suggestions": [
                        "Which template should I start with?",
                        "What permissions do I need first?",
                    ]
                }
            }

    def fake_get_llm(*, model_id: str | None = None, **_: object) -> FakeLLM:
        llm_calls.append(model_id)
        return FakeLLM(model_id)

    def fake_create_agent(
        *, model: object, tools: list[object], system_prompt: str, response_format: object
    ):
        assert tools == []
        assert system_prompt
        assert isinstance(response_format, ToolStrategy)
        assert response_format.schema.__name__ == "FollowUpSuggestions"
        model_id = getattr(model, "model_id", None)
        agent_models.append(model_id)
        return FakeAgent()

    monkeypatch.setattr(suggestions_route, "get_llm", fake_get_llm)
    monkeypatch.setattr(suggestions_route, "create_agent", fake_create_agent, raising=False)

    async def run() -> None:
        payload = {
            "last_user_message": "How do I create an Oracle Visual Builder app?",
            "last_message": "You can create a visual application from App Builder.",
            "model": "xai.grok-4",
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post("/api/suggestions", json=payload)
            assert response.status_code == 200
            assert response.json()["suggestions"] == [
                "Which template should I start with?",
                "What permissions do I need first?",
            ]

    asyncio.run(run())
    assert llm_calls == ["xai.grok-4"]
    assert agent_models == ["xai.grok-4"]


def test_suggestions_trace_carries_thread_request_and_outcome_metadata(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTrace:
        trace_context = {"trace_id": "trace-1", "parent_span_id": "span-1"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def update_output(self, output: object) -> None:
            captured["output"] = output

        def update_metadata(self, metadata: dict[str, str]) -> None:
            captured["metadata_update"] = metadata

    class FakeAgent:
        def invoke(self, payload: dict[str, object], config: dict[str, object] | None = None):
            assert payload
            assert config
            return {"structured_response": {"suggestions": ["What comes next?"]}}

    monkeypatch.setattr(suggestions_route, "get_llm", lambda **kwargs: object())
    monkeypatch.setattr(suggestions_route, "create_agent", lambda **kwargs: FakeAgent())
    def fake_start_trace(**kwargs):
        captured["trace"] = kwargs
        return FakeTrace()

    monkeypatch.setattr(suggestions_route, "start_langfuse_chat_trace", fake_start_trace)
    monkeypatch.setattr(
        suggestions_route,
        "add_langfuse_callbacks",
        lambda run_config, **kwargs: (
            captured.setdefault("run_config", run_config),
            captured.setdefault("callbacks", kwargs),
        ),
    )
    monkeypatch.setattr(
        suggestions_route,
        "REQUEST_ID_CTX",
        SimpleNamespace(get=lambda: "request-1"),
        raising=False,
    )

    async def run() -> None:
        payload = {
            "last_user_message": "What is the payment policy?",
            "last_message": "Payment is due within 30 days.",
            "model": "xai.grok-4",
            "thread_id": "thread-1",
        }
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post("/api/suggestions", json=payload)
            assert response.status_code == 200
            assert response.json()["suggestions"] == ["What comes next?"]

    asyncio.run(run())
    trace_kwargs = captured["trace"]
    callback_kwargs = captured["callbacks"]
    assert trace_kwargs["session_id"] == "thread-1"
    assert trace_kwargs["thread_id"] == "thread-1"
    assert callback_kwargs["session_id"] == "thread-1"
    assert callback_kwargs["tags"] == [
        "feature:suggestions",
        "mode:suggestions",
        "model:xai.grok-4",
    ]
    assert captured["run_config"]["metadata"] == {
        "request_id": "request-1",
        "thread_id": "thread-1",
        "mode": "suggestions",
        "model_id": "xai.grok-4",
    }
    assert captured["output"] == {"suggestion_count": 1, "outcome": "success"}
    assert captured["metadata_update"] == {"suggestion_count": "1", "outcome": "success"}
