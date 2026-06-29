from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.rag_agent.graphs.nodes import direct


class FakeLlm:
    model_id = "fake-direct-model"

    def invoke(self, messages: list[object], config: dict[str, object] | None = None) -> AIMessage:
        _ = messages, config
        return AIMessage(content="Direct answer")


def test_direct_node_invokes_llm_without_chat_runtime_service(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_get_llm(*, model_id: str | None = None) -> FakeLlm:
        calls.append({"model_id": model_id})
        return FakeLlm()

    def fake_invoke(llm: FakeLlm, history: list[Any], run_config: dict[str, object]) -> AIMessage:
        calls.append({"history": history, "run_config": run_config, "llm": llm})
        return AIMessage(content="Direct answer")

    monkeypatch.setattr(direct, "get_llm", fake_get_llm)
    monkeypatch.setattr(direct, "invoke_llm_with_optional_config", fake_invoke)
    monkeypatch.setattr(
        direct,
        "emit_usage_observability",
        lambda **kwargs: (None, None),
    )
    monkeypatch.setattr(
        direct,
        "start_langfuse_chat_trace",
        lambda **kwargs: _FakeTraceContextManager(),
    )

    runtime = SimpleNamespace(
        context={"model_id": "model-1", "session_id": "session-1", "enable_tracing": False},
        execution_info=SimpleNamespace(thread_id="thread-1"),
    )

    result = asyncio.run(
        direct.run_direct_node(
            {"messages": [HumanMessage(content="Hello")]},
            runtime,  # type: ignore[arg-type]
        )
    )

    assistant = result["messages"][0]
    assert isinstance(assistant, AIMessage)
    assert assistant.content == "Direct answer"
    assert assistant.additional_kwargs["mode"] == "direct"
    assert calls[0]["model_id"] == "model-1"


class _FakeTrace:
    trace_context = None
    trace_id = None


class _FakeTraceContextManager:
    def __enter__(self) -> _FakeTrace:
        return _FakeTrace()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        _ = exc_type, exc, tb
        return False
