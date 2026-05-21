from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.config_route import router
from src.rag_agent.infrastructure import mcp_adapter_runtime
from src.rag_agent.infrastructure.mcp_config_store import (
    MCPServerConfig,
    delete_mcp_server_config,
    list_mcp_server_configs,
    resolve_enabled_mcp_servers_config,
    upsert_mcp_server_config,
)


def test_store_uses_settings_until_ui_override_exists(tmp_path) -> None:
    store_path = tmp_path / "mcp_servers.json"
    base_config = {
        "default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"},
        "context7": {"transport": "streamable-http", "url": "http://localhost:9001/mcp"},
    }

    assert list_mcp_server_configs(base_config=base_config, store_path=store_path) == [
        MCPServerConfig(
            key="default",
            transport="streamable-http",
            url="http://localhost:9000/mcp",
            enabled=True,
        ),
        MCPServerConfig(
            key="context7",
            transport="streamable-http",
            url="http://localhost:9001/mcp",
            enabled=True,
        ),
    ]

    upsert_mcp_server_config(
        MCPServerConfig(
            key="default",
            transport="streamable-http",
            url="http://127.0.0.1:9100/mcp",
            enabled=False,
        ),
        base_config=base_config,
        store_path=store_path,
    )

    listed = list_mcp_server_configs(base_config=base_config, store_path=store_path)
    assert listed[0].url == "http://127.0.0.1:9100/mcp"
    assert listed[0].enabled is False
    assert listed[1].key == "context7"
    assert listed[1].enabled is True
    assert resolve_enabled_mcp_servers_config(base_config=base_config, store_path=store_path) == {
        "context7": {"transport": "streamable-http", "url": "http://localhost:9001/mcp"}
    }


def test_store_delete_persists_removal_from_ui_managed_list(tmp_path) -> None:
    store_path = tmp_path / "mcp_servers.json"
    base_config = {
        "default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"},
        "calculator": {"transport": "streamable-http", "url": "http://localhost:9002/mcp"},
    }

    delete_mcp_server_config("default", base_config=base_config, store_path=store_path)

    assert [item.key for item in list_mcp_server_configs(base_config=base_config, store_path=store_path)] == [
        "calculator"
    ]
    assert resolve_enabled_mcp_servers_config(base_config=base_config, store_path=store_path) == {
        "calculator": {"transport": "streamable-http", "url": "http://localhost:9002/mcp"}
    }


def test_mcp_config_routes_list_upsert_toggle_and_delete(tmp_path) -> None:
    settings = SimpleNamespace(
        ENABLE_MCP_TOOLS=True,
        MCP_UI_CONFIG_FILE=str(tmp_path / "mcp_servers.json"),
        MCP_SERVERS_CONFIG={
            "default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}
        },
        REGION="us-chicago-1",
        EMBED_MODEL_ID="embed",
        MODEL_LIST=[],
        MODEL_DISPLAY_NAMES={},
        COLLECTION_LIST=[],
        DEFAULT_COLLECTION="RAG_KNOWLEDGE_BASE",
        ENABLE_USER_FEEDBACK=False,
    )
    app = FastAPI()
    app.state.resources = SimpleNamespace(settings=settings)
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/config/mcp-servers")
    assert response.status_code == 200
    assert response.json()["servers"] == [
        {
            "key": "default",
            "transport": "streamable-http",
            "url": "http://localhost:9000/mcp",
            "enabled": True,
        }
    ]

    response = client.put(
        "/api/config/mcp-servers/external",
        json={
            "transport": "streamable-http",
            "url": "https://mcp.example.com/mcp",
            "enabled": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["key"] == "external"

    response = client.patch("/api/config/mcp-servers/default/enabled", json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False

    response = client.delete("/api/config/mcp-servers/external")
    assert response.status_code == 200

    response = client.get("/api/config/mcp-servers")
    assert response.status_code == 200
    assert response.json()["servers"] == [
        {
            "key": "default",
            "transport": "streamable-http",
            "url": "http://localhost:9000/mcp",
            "enabled": False,
        }
    ]


def test_mcp_config_route_tests_draft_connection_without_persisting(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Tool:
        name = "draft.list_documents"
        description = "List documents"
        args_schema = {"type": "object"}

    async def fake_load_adapter_tools(*, server_keys=None, run_config=None):
        captured["server_keys"] = server_keys
        captured["run_config"] = run_config
        return [_Tool()]

    monkeypatch.setattr(mcp_adapter_runtime, "load_adapter_tools", fake_load_adapter_tools)

    settings = SimpleNamespace(
        ENABLE_MCP_TOOLS=True,
        MCP_UI_CONFIG_FILE=str(tmp_path / "mcp_servers.json"),
        MCP_SERVERS_CONFIG={
            "default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}
        },
        REGION="us-chicago-1",
        EMBED_MODEL_ID="embed",
        MODEL_LIST=[],
        MODEL_DISPLAY_NAMES={},
        COLLECTION_LIST=[],
        DEFAULT_COLLECTION="RAG_KNOWLEDGE_BASE",
        ENABLE_USER_FEEDBACK=False,
    )
    app = FastAPI()
    app.state.resources = SimpleNamespace(settings=settings)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/api/config/mcp-servers/draft/test",
        json={
            "transport": "streamable-http",
            "url": "http://localhost:9100/mcp",
            "enabled": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "key": "draft",
        "ok": True,
        "tool_count": 1,
        "tools": [
            {
                "name": "draft.list_documents",
                "description": "List documents",
            }
        ],
        "error": None,
    }
    assert captured["server_keys"] == ["draft"]
    assert captured["run_config"] == {
        "configurable": {
            "mcp_servers_config_override": {
                "draft": {
                    "transport": "streamable-http",
                    "url": "http://localhost:9100/mcp",
                }
            }
        }
    }
    assert not (tmp_path / "mcp_servers.json").exists()


def test_mcp_config_route_reports_connection_failure(tmp_path, monkeypatch) -> None:
    async def fake_load_adapter_tools(*, server_keys=None, run_config=None):
        _ = server_keys, run_config
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mcp_adapter_runtime, "load_adapter_tools", fake_load_adapter_tools)

    settings = SimpleNamespace(
        ENABLE_MCP_TOOLS=True,
        MCP_UI_CONFIG_FILE=str(tmp_path / "mcp_servers.json"),
        MCP_SERVERS_CONFIG={
            "default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}
        },
        REGION="us-chicago-1",
        EMBED_MODEL_ID="embed",
        MODEL_LIST=[],
        MODEL_DISPLAY_NAMES={},
        COLLECTION_LIST=[],
        DEFAULT_COLLECTION="RAG_KNOWLEDGE_BASE",
        ENABLE_USER_FEEDBACK=False,
    )
    app = FastAPI()
    app.state.resources = SimpleNamespace(settings=settings)
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/api/config/mcp-servers/default/test")

    assert response.status_code == 200
    assert response.json() == {
        "key": "default",
        "ok": False,
        "tool_count": 0,
        "tools": [],
        "error": "connection refused",
    }


def test_mcp_config_route_times_out_connection_test(tmp_path, monkeypatch) -> None:
    async def fake_load_adapter_tools(*, server_keys=None, run_config=None):
        _ = server_keys, run_config
        await asyncio.sleep(1)
        return []

    monkeypatch.setattr(mcp_adapter_runtime, "load_adapter_tools", fake_load_adapter_tools)

    settings = SimpleNamespace(
        ENABLE_MCP_TOOLS=True,
        MCP_UI_CONFIG_FILE=str(tmp_path / "mcp_servers.json"),
        MCP_CONNECTION_TEST_TIMEOUT_SECONDS=0.01,
        MCP_SERVERS_CONFIG={
            "default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}
        },
        REGION="us-chicago-1",
        EMBED_MODEL_ID="embed",
        MODEL_LIST=[],
        MODEL_DISPLAY_NAMES={},
        COLLECTION_LIST=[],
        DEFAULT_COLLECTION="RAG_KNOWLEDGE_BASE",
        ENABLE_USER_FEEDBACK=False,
    )
    app = FastAPI()
    app.state.resources = SimpleNamespace(settings=settings)
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/api/config/mcp-servers/default/test")

    assert response.status_code == 200
    assert response.json() == {
        "key": "default",
        "ok": False,
        "tool_count": 0,
        "tools": [],
        "error": "Connection test timed out after 0.01 seconds",
    }
