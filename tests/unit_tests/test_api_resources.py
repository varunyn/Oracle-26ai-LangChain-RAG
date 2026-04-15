import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from api.dependencies import messages_to_runtime_state
from api.resources import create_app_resources, shutdown_app_resources
from api.schemas import ChatMessage


def test_create_app_resources_wires_settings_and_chat_runtime_service(monkeypatch) -> None:
    fake_settings = SimpleNamespace()
    mock_graph = SimpleNamespace()
    mock_graph_cls = MagicMock(return_value=mock_graph)

    monkeypatch.setattr("api.resources.get_settings", lambda: fake_settings)
    monkeypatch.setattr("api.resources.ChatRuntimeService", mock_graph_cls)

    resources = asyncio.run(create_app_resources())

    assert resources.settings is fake_settings
    assert resources.chat_runtime_service is mock_graph
    mock_graph_cls.assert_called_once_with()


def test_messages_to_runtime_state_drops_internal_mcp_assistant_traces() -> None:
    user_request, chat_history = messages_to_runtime_state(
        [
            ChatMessage(role="user", content="what's the namespace of my tenancy?"),
            ChatMessage(
                role="assistant",
                content='oci-mcp-server_run_oci_command(command="os ns get")',
            ),
            ChatMessage(role="user", content="what's the namespace of my tenancy?"),
        ]
    )

    assert user_request == "what's the namespace of my tenancy?"
    assert len(chat_history) == 1
    assert chat_history[0].content == "what's the namespace of my tenancy?"


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
