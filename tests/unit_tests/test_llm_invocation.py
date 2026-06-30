from __future__ import annotations

import asyncio

from src.rag_agent.runtime.llm_invocation import (
    ainvoke_llm_with_optional_config,
    suppress_llm_streaming,
)


class _ConfigurableModel:
    def __init__(self) -> None:
        self.configs: list[object] = []

    def with_config(self, config: object) -> _ConfigurableModel:
        self.configs.append(config)
        return self


class _KeywordConfigurableModel:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, object]] = []

    def with_config(self, *args: object, **kwargs: object) -> _KeywordConfigurableModel:
        if args:
            raise TypeError("positional config unsupported")
        self.kwargs.append(kwargs)
        return self


def test_suppress_llm_streaming_tags_model_with_nostream() -> None:
    model = _ConfigurableModel()

    configured = suppress_llm_streaming(model)

    assert configured is model
    assert model.configs == [{"tags": ["nostream"]}]


def test_suppress_llm_streaming_supports_keyword_config_models() -> None:
    model = _KeywordConfigurableModel()

    configured = suppress_llm_streaming(model)

    assert configured is model
    assert model.kwargs == [{"tags": ["nostream"]}]


def test_suppress_llm_streaming_returns_original_model_without_config_support() -> None:
    model = object()

    configured = suppress_llm_streaming(model)

    assert configured is model


class _AsyncModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object | None]] = []

    async def ainvoke(self, messages: object, config: object | None = None) -> object:
        self.calls.append((messages, config))
        return {"ok": True}


class _SyncOnlyModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object | None]] = []

    def invoke(self, messages: object, config: object | None = None) -> object:
        self.calls.append((messages, config))
        return {"ok": True}


def test_ainvoke_llm_with_optional_config_prefers_async_invoke() -> None:
    model = _AsyncModel()

    result = asyncio.run(
        ainvoke_llm_with_optional_config(
            model,
            [{"role": "user", "content": "hello"}],
            {"configurable": {"thread_id": "t1"}},
        )
    )

    assert result == {"ok": True}
    assert model.calls == [([{"role": "user", "content": "hello"}], {"configurable": {"thread_id": "t1"}})]


def test_ainvoke_llm_with_optional_config_falls_back_to_sync_invoke() -> None:
    model = _SyncOnlyModel()

    result = asyncio.run(
        ainvoke_llm_with_optional_config(
            model,
            [{"role": "user", "content": "hello"}],
            {"configurable": {"thread_id": "t1"}},
        )
    )

    assert result == {"ok": True}
    assert model.calls == [([{"role": "user", "content": "hello"}], {"configurable": {"thread_id": "t1"}})]
