import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from api.resources import create_app_resources, shutdown_app_resources


def test_create_app_resources_wires_settings_and_chat_runtime_service(monkeypatch) -> None:
    fake_settings = SimpleNamespace(
        ENABLE_PERSISTENT_MEMORY=False,
        LANGGRAPH_SQLITE_PATH=":memory:",
    )
    mock_graph = SimpleNamespace()
    mock_graph_cls = MagicMock(return_value=mock_graph)

    monkeypatch.setattr("api.resources.get_settings", lambda: fake_settings)
    monkeypatch.setattr("api.resources.ChatRuntimeService", mock_graph_cls)

    resources = asyncio.run(create_app_resources())

    assert resources.settings is fake_settings
    assert resources.chat_runtime_service is mock_graph
    mock_graph_cls.assert_called_once_with(thread_state_store=None)


def test_shutdown_app_resources_calls_langfuse_shutdown_and_adapter_cleanup(monkeypatch) -> None:
    calls: list[str] = []

    def _safe_shutdown() -> None:
        calls.append("langfuse")

    async def _clear_cache() -> None:
        calls.append("adapter")

    monkeypatch.setattr("api.resources.langfuse_safe_shutdown", _safe_shutdown)
    monkeypatch.setattr("api.resources.clear_adapter_runtime_cache", _clear_cache)

    resources = SimpleNamespace(get_state_conn=lambda: None)
    asyncio.run(shutdown_app_resources(resources))

    assert calls == ["langfuse", "adapter"]
