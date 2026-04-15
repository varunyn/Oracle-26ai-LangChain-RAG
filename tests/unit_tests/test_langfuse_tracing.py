from __future__ import annotations

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
