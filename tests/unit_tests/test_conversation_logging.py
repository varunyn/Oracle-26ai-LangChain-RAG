from __future__ import annotations

from typing import Any

from api import dependencies
from src.rag_agent.runtime import observability


def test_log_conversation_out_uses_redacted_dashboard_friendly_attributes(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_info(message: str, *args: object, **kwargs: object) -> None:
        captured["message"] = message
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(dependencies.conv_log, "info", fake_info)

    dependencies.log_conversation_out(
        final_answer="Sensitive answer text",
        error=None,
        mcp_used=True,
        mcp_tools_used=["search_docs", {"name": "calculator"}, {"tool_name": "lookup"}],
        standalone_question="Sensitive user question",
    )

    rendered_message = captured["message"] % captured["args"]
    assert "Sensitive answer text" not in rendered_message
    assert "Sensitive user question" not in rendered_message

    attributes = captured["kwargs"]["extra"]["otel_attributes"]
    assert attributes["event_type"] == "chat_out"
    assert attributes["answer_len"] == len("Sensitive answer text")
    assert attributes["standalone_len"] == len("Sensitive user question")
    assert attributes["mcp_used"] is True
    assert attributes["mcp_tool_count"] == 3
    assert attributes["mcp_tool_names"] == "search_docs,calculator,lookup"
    assert "final_answer_preview" not in attributes
    assert "standalone_preview" not in attributes


def test_emit_usage_observability_adds_dashboard_friendly_attributes(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_info(message: str, *args: object, **kwargs: object) -> None:
        captured["message"] = message
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(observability.logger, "info", fake_info)

    usage, cost_usd = observability.emit_usage_observability(
        mode="rag",
        model_id="google.gemini-2.5-pro",
        session_id="session-1",
        thread_id="thread-1",
        usage={"input": 10, "output": 20, "total": 30},
    )

    assert usage == {"input": 10, "output": 20, "total": 30}
    assert cost_usd is not None

    attributes = captured["kwargs"]["extra"]["otel_attributes"]
    assert attributes["event_type"] == "llm_usage"
    assert attributes["mode"] == "rag"
    assert attributes["model_id"] == "google.gemini-2.5-pro"
    assert attributes["session_id"] == "session-1"
    assert attributes["thread_id"] == "thread-1"
    assert attributes["input_tokens"] == 10
    assert attributes["output_tokens"] == 20
    assert attributes["total_tokens"] == 30
    assert attributes["cost_usd"] == cost_usd
    assert attributes["pricing_basis"] == "token"
