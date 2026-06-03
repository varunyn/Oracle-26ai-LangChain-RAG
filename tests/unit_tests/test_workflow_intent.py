from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from src.rag_agent.workflows.workflow_intent import (
    WorkflowDiscoveryToolDecision,
    WorkflowIntentDecision,
    select_repeated_workflow_discovery_tools,
    should_use_repeated_workflow,
)


def test_should_use_repeated_workflow_uses_deterministic_classifier_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeStructuredModel:
        def with_structured_output(self, schema: object) -> FakeStructuredModel:
            captured["schema"] = schema
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> object:
            captured["messages"] = messages
            captured["config"] = config
            return WorkflowIntentDecision(use_repeated_workflow=True, reason="multi-item workflow")

    def fake_get_llm(**kwargs: object) -> FakeStructuredModel:
        captured["llm_kwargs"] = kwargs
        return FakeStructuredModel()

    @tool("list_work")
    def list_work() -> str:
        """List work."""
        return "[]"

    monkeypatch.setattr("src.rag_agent.workflows.workflow_intent.get_llm", fake_get_llm)

    decision = asyncio.run(
        should_use_repeated_workflow(
            question="Process every listed item and send a summary.",
            tools=[list_work],
            model_id="test-model",
            run_config={"configurable": {"thread_id": "thread-1"}},
        )
    )

    assert decision is True
    assert captured["llm_kwargs"] == {
        "model_id": "test-model",
        "temperature": 0,
        "max_tokens": 256,
    }
    assert captured["schema"] is WorkflowIntentDecision


def test_select_repeated_workflow_discovery_tools_uses_model_selected_tool_names(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeStructuredModel:
        def with_structured_output(self, schema: object) -> FakeStructuredModel:
            captured["schema"] = schema
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> object:
            captured["messages"] = messages
            captured["config"] = config
            return WorkflowDiscoveryToolDecision(
                tool_names=["list_work"],
                reason="list_work establishes the queue",
            )

    def fake_get_llm(**kwargs: object) -> FakeStructuredModel:
        captured["llm_kwargs"] = kwargs
        return FakeStructuredModel()

    @tool("list_work")
    def list_work() -> str:
        """List work."""
        return "[]"

    @tool("create_work")
    def create_work(item_id: str) -> str:
        """Create one work item."""
        return item_id

    monkeypatch.setattr("src.rag_agent.workflows.workflow_intent.get_llm", fake_get_llm)

    selected = asyncio.run(
        select_repeated_workflow_discovery_tools(
            question="Process every listed item and send a summary.",
            tools=[list_work, create_work],
            model_id="test-model",
            run_config={"configurable": {"thread_id": "thread-1"}},
        )
    )

    assert [tool.name for tool in selected] == ["list_work"]
    assert captured["llm_kwargs"] == {
        "model_id": "test-model",
        "temperature": 0,
        "max_tokens": 512,
    }
    assert captured["schema"] is WorkflowDiscoveryToolDecision


def test_should_use_repeated_workflow_falls_back_to_json_when_structured_output_is_empty(
    monkeypatch,
) -> None:
    class FakeStructuredModel:
        def __init__(self) -> None:
            self.structured = False

        def with_structured_output(self, schema: object) -> FakeStructuredModel:
            _ = schema
            self.structured = True
            return self

        def invoke(self, messages: list[object], *, config: object | None = None) -> object:
            _ = messages, config
            if self.structured:
                self.structured = False
                return None
            return '{"use_repeated_workflow": true, "reason": "explicit list and per-item work"}'

    @tool("list_work")
    def list_work() -> str:
        """List work."""
        return "[]"

    monkeypatch.setattr(
        "src.rag_agent.workflows.workflow_intent.get_llm",
        lambda **kwargs: FakeStructuredModel(),
    )

    decision = asyncio.run(
        should_use_repeated_workflow(
            question="Process every listed item and send a summary.",
            tools=[list_work],
            model_id="test-model",
        )
    )

    assert decision is True
