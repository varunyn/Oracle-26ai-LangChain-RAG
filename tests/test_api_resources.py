from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.resources import create_app_resources


def test_create_app_resources_skips_code_mode_init_when_disabled(monkeypatch):
    fake_settings = SimpleNamespace(CODE_MODE_ENABLED=False)
    fake_create_async_checkpointer = AsyncMock(return_value=(object(), object()))
    fake_init_code_mode_client = AsyncMock()

    monkeypatch.setattr("api.resources.get_settings", lambda: fake_settings)
    monkeypatch.setattr("api.resources.rag_graph.create_async_checkpointer", fake_create_async_checkpointer)
    monkeypatch.setattr("api.resources.rag_graph.create_workflow", lambda checkpointer: object())
    monkeypatch.setattr("api.resources.GraphService", lambda graph: SimpleNamespace(graph=graph))
    monkeypatch.setattr("api.resources.init_code_mode_client", fake_init_code_mode_client)

    create_app_resources_sync = create_app_resources
    resources = __import__("asyncio").run(create_app_resources_sync())

    assert resources.settings is fake_settings
    fake_init_code_mode_client.assert_not_awaited()
