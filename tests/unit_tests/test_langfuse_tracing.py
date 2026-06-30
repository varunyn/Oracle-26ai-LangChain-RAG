from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.rag_agent.core import config
from src.rag_agent.utils import langfuse_tracing


def test_add_langfuse_callbacks_disabled_leaves_config_unchanged(monkeypatch: Any) -> None:
    """When Langfuse is disabled, add_langfuse_callbacks does not mutate run_config."""
    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", False)
    monkeypatch.setattr(langfuse_tracing, "LangfuseRuntime", object())
    langfuse_tracing.set_langfuse_client(None, disabled=False)

    run_config: dict[str, Any] = {"configurable": {"thread_id": "t-1"}}
    langfuse_tracing.add_langfuse_callbacks(run_config, session_id="sess-1", user_id=None)

    assert "callbacks" not in run_config
    assert "metadata" not in run_config
    assert run_config["configurable"]["thread_id"] == "t-1"


def test_add_langfuse_callbacks_enabled_adds_callbacks_and_metadata(monkeypatch: Any) -> None:
    """When Langfuse is enabled and client is set, run_config gets callbacks and metadata."""
    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", True)
    monkeypatch.setattr(langfuse_tracing, "LangfuseRuntime", object())
    # Use a real-looking client so get_langfuse_client() returns it; add_langfuse_callbacks
    # needs the client to be created so it proceeds to add CallbackHandler.
    try:
        from langfuse import Langfuse
    except Exception:
        import pytest

        pytest.skip("langfuse not installed")
    # Create client so it registers as singleton; use fake keys so we don't hit the network
    client = Langfuse(public_key="pk-fake", secret_key="sk-fake", host="http://localhost")
    langfuse_tracing.set_langfuse_client(client, disabled=False)

    run_config: dict[str, Any] = {"configurable": {"thread_id": "t-2"}}
    langfuse_tracing.add_langfuse_callbacks(run_config, session_id="sess-2", user_id="user-2")

    assert "callbacks" in run_config
    assert len(run_config["callbacks"]) >= 1
    assert "metadata" in run_config
    assert run_config["metadata"].get("langfuse_session_id") == "sess-2"
    assert run_config["metadata"].get("langfuse_user_id") == "user-2"
    assert run_config["configurable"]["thread_id"] == "t-2"


def test_add_langfuse_callbacks_sets_trace_context_name_and_tags(monkeypatch: Any) -> None:
    """Langfuse callbacks should attach to the request trace without inventing a user id."""
    captured: dict[str, Any] = {}

    class _CallbackHandler:
        def __init__(
            self,
            *,
            public_key: str | None = None,
            trace_context: dict[str, str] | None = None,
        ) -> None:
            captured["public_key"] = public_key
            captured["trace_context"] = trace_context

    import langfuse.langchain

    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", True)
    monkeypatch.setattr(langfuse_tracing, "LangfuseRuntime", object())
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", _CallbackHandler)
    monkeypatch.setattr(
        langfuse_tracing,
        "get_settings",
        lambda: SimpleNamespace(LANGFUSE_PUBLIC_KEY="pk-test"),
    )
    langfuse_tracing.set_langfuse_client(SimpleNamespace(public_key="pk-test"), disabled=False)

    run_config: dict[str, Any] = {"configurable": {"thread_id": "t-3", "model_id": "model-a"}}
    langfuse_tracing.add_langfuse_callbacks(
        run_config,
        session_id="sess-3",
        user_id=None,
        trace_context={"trace_id": "trace-3", "parent_span_id": "span-3"},
        trace_name="chat-rag",
        tags=["chat", "mode:rag", "model:model-a"],
    )

    metadata = run_config["metadata"]
    assert captured["public_key"] == "pk-test"
    assert captured["trace_context"] == {"trace_id": "trace-3", "parent_span_id": "span-3"}
    assert metadata["langfuse_session_id"] == "sess-3"
    assert "langfuse_user_id" not in metadata
    assert metadata["langfuse_trace_name"] == "chat-rag"
    assert metadata["langfuse_tags"] == ["chat", "mode:rag", "model:model-a"]
    assert metadata["ls_model_name"] == "model-a"


def test_add_langfuse_callbacks_falls_back_to_settings_public_key(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class _CallbackHandler:
        def __init__(
            self,
            *,
            public_key: str | None = None,
            trace_context: dict[str, str] | None = None,
        ) -> None:
            captured["public_key"] = public_key
            captured["trace_context"] = trace_context

    import langfuse.langchain

    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", True)
    monkeypatch.setattr(langfuse_tracing, "LangfuseRuntime", object())
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", _CallbackHandler)
    monkeypatch.setattr(
        langfuse_tracing,
        "get_settings",
        lambda: SimpleNamespace(LANGFUSE_PUBLIC_KEY="pk-settings"),
    )
    langfuse_tracing.set_langfuse_client(object(), disabled=False)

    run_config: dict[str, Any] = {"configurable": {"thread_id": "t-4"}}
    langfuse_tracing.add_langfuse_callbacks(run_config, session_id="sess-4", user_id=None)

    assert captured["public_key"] == "pk-settings"


def test_get_langfuse_client_passes_sample_rate_when_configured(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class _LangfuseRuntime:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", True)
    monkeypatch.setattr(langfuse_tracing, "LangfuseRuntime", _LangfuseRuntime)
    monkeypatch.setattr(
        langfuse_tracing,
        "get_settings",
        lambda: SimpleNamespace(
            LANGFUSE_HOST="http://localhost:3300",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
            LANGFUSE_TRACING_ENVIRONMENT="test",
            LANGFUSE_ENVIRONMENT=None,
            LANGFUSE_RELEASE=None,
            LANGFUSE_SAMPLE_RATE=0.25,
        ),
    )
    langfuse_tracing.set_langfuse_client(None, disabled=False)

    assert langfuse_tracing.get_langfuse_client() is not None
    assert captured["sample_rate"] == 0.25


def test_get_langfuse_client_prefers_environment_over_settings(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class _LangfuseRuntime:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", True)
    monkeypatch.setattr(langfuse_tracing, "LangfuseRuntime", _LangfuseRuntime)
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse-web:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-env")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-env")
    monkeypatch.setattr(
        langfuse_tracing,
        "get_settings",
        lambda: SimpleNamespace(
            LANGFUSE_HOST="http://localhost:3300",
            LANGFUSE_PUBLIC_KEY="pk-settings",
            LANGFUSE_SECRET_KEY="sk-settings",
            LANGFUSE_TRACING_ENVIRONMENT="test",
            LANGFUSE_ENVIRONMENT=None,
            LANGFUSE_RELEASE=None,
            LANGFUSE_SAMPLE_RATE=None,
        ),
    )
    langfuse_tracing.set_langfuse_client(None, disabled=False)

    assert langfuse_tracing.get_langfuse_client() is not None
    assert captured["base_url"] == "http://langfuse-web:3000"
    assert captured["public_key"] == "pk-env"
    assert captured["secret_key"] == "sk-env"


def test_resolve_langfuse_host_rewrites_localhost_inside_docker(monkeypatch: Any) -> None:
    monkeypatch.setattr(langfuse_tracing, "_running_in_docker", lambda: True)
    monkeypatch.setattr(
        langfuse_tracing,
        "get_settings",
        lambda: SimpleNamespace(LANGFUSE_HOST="http://localhost:3300"),
    )
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    assert langfuse_tracing._resolve_langfuse_host() == "http://langfuse-web:3000"


def test_start_langfuse_chat_trace_propagates_trace_attributes(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class _Observation:
        trace_id = "trace-1"
        id = "span-1"

    class _ObservationManager:
        def __enter__(self) -> _Observation:
            captured["observation_entered"] = True
            return _Observation()

        def __exit__(self, *args: object) -> None:
            captured["observation_exited"] = True

    class _PropagationManager:
        def __enter__(self) -> None:
            captured["propagation_entered"] = True

        def __exit__(self, *args: object) -> None:
            captured["propagation_exited"] = True

    class _Client:
        def start_as_current_observation(self, **kwargs: Any) -> _ObservationManager:
            captured["observation_kwargs"] = kwargs
            return _ObservationManager()

    def _propagate_attributes(**kwargs: Any) -> _PropagationManager:
        captured["propagation_kwargs"] = kwargs
        return _PropagationManager()

    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", True)
    monkeypatch.setattr(langfuse_tracing, "LangfuseRuntime", object())
    monkeypatch.setattr(langfuse_tracing, "LangfusePropagateAttributes", _propagate_attributes)
    langfuse_tracing.set_langfuse_client(_Client(), disabled=False)

    with langfuse_tracing.start_langfuse_chat_trace(
        enabled=True,
        mode="rag",
        model_id="model-a",
        session_id="session-a",
        thread_id="thread-a",
        input_payload={"question": "hello"},
    ) as trace:
        assert trace.trace_context == {"trace_id": "trace-1", "parent_span_id": "span-1"}

    assert captured["observation_kwargs"] == {
        "name": "chat-rag",
        "as_type": "chain",
        "input": {"question": "hello"},
        "metadata": {"mode": "rag", "model_id": "model-a", "thread_id": "thread-a"},
    }
    assert captured["propagation_kwargs"] == {
        "trace_name": "chat-rag",
        "session_id": "session-a",
        "metadata": {"mode": "rag", "model_id": "model-a", "thread_id": "thread-a"},
        "tags": ["chat", "mode:rag", "model:model-a"],
    }
    assert captured["propagation_entered"] is True
    assert captured["propagation_exited"] is True
    assert captured["observation_exited"] is True


def test_safe_flush_no_op_when_disabled(monkeypatch: Any) -> None:
    """safe_flush does not raise when Langfuse is disabled."""
    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", False)
    langfuse_tracing.set_langfuse_client(None, disabled=False)
    langfuse_tracing.safe_flush()


def test_safe_shutdown_calls_client_shutdown_and_clears_singleton(monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "ENABLE_LANGFUSE_TRACING", True)
    monkeypatch.setattr(langfuse_tracing, "LangfuseRuntime", object())

    class _Client:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    client = _Client()
    langfuse_tracing.set_langfuse_client(client, disabled=False)

    langfuse_tracing.safe_shutdown()

    assert client.shutdown_calls == 1
    assert langfuse_tracing._LANGFUSE_CLIENT is None


def test_sanitize_usage_metadata_drops_none_token_detail_values() -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "output_token_details": {
            "reasoning": None,
            "accepted_prediction": 0,
        },
    }

    sanitized = langfuse_tracing._sanitize_usage_metadata_for_langfuse(usage)

    assert sanitized == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "output_token_details": {
            "accepted_prediction": 0,
        },
    }
