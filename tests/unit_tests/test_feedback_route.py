from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from api.routes import feedback
from api.schemas import FeedbackRequest


def test_record_langfuse_feedback_score_skips_without_trace_id(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        feedback,
        "get_langfuse_client",
        lambda: SimpleNamespace(create_score=lambda **kwargs: calls.append(kwargs)),
    )

    request = FeedbackRequest(question="What?", answer="Answer", feedback=4)

    feedback.record_langfuse_feedback_score(request)

    assert calls == []


def test_record_langfuse_feedback_score_creates_numeric_score(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    flush_calls: list[str] = []

    score_config = SimpleNamespace(id="config-1", name="user-rating")
    score_configs_response = SimpleNamespace(data=[score_config])
    client = SimpleNamespace(
        api=SimpleNamespace(score_configs=SimpleNamespace(get=lambda *, limit: score_configs_response)),
        create_score=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        feedback,
        "get_langfuse_client",
        lambda: client,
    )
    monkeypatch.setattr(feedback, "safe_flush", lambda: flush_calls.append("flush"))

    request = FeedbackRequest(question="What?", answer="Answer", feedback=5, trace_id="trace-1")

    feedback.record_langfuse_feedback_score(request)

    assert calls == [
        {
            "trace_id": "trace-1",
            "name": "user-rating",
            "value": 5,
            "data_type": "NUMERIC",
            "config_id": "config-1",
        }
    ]
    assert flush_calls == ["flush"]
