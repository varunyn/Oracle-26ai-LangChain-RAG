from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.config_route import router
from src.rag_agent.infrastructure import mcp_adapter_runtime
from src.rag_agent.infrastructure import mcp_config_store as mod
from src.rag_agent.infrastructure.mcp_config_store import (
    MCPServerAuthConfig,
    MCPServerConfig,
    delete_mcp_server_config,
    list_mcp_server_configs,
    resolve_enabled_mcp_servers_config,
    upsert_mcp_server_config,
)


def test_supported_transports_exclude_legacy_sse() -> None:
    assert mod.SUPPORTED_TRANSPORTS == {"http", "streamable-http", "stdio"}


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

    assert [
        item.key for item in list_mcp_server_configs(base_config=base_config, store_path=store_path)
    ] == ["calculator"]
    assert resolve_enabled_mcp_servers_config(base_config=base_config, store_path=store_path) == {
        "calculator": {"transport": "streamable-http", "url": "http://localhost:9002/mcp"}
    }


def test_store_preserves_auth_config_for_runtime(tmp_path) -> None:
    store_path = tmp_path / "mcp_servers.json"
    server = MCPServerConfig(
        key="secure",
        transport="streamable-http",
        url="https://mcp.example.com/mcp",
        auth=MCPServerAuthConfig(
            type="oauth_client_credentials",
            token_url="https://auth.example.com/oauth/token",
            client_id="client-id",
            client_secret="client-secret",
            scope="read:mcp",
            audience="mcp-api",
        ),
    )

    upsert_mcp_server_config(server, base_config={}, store_path=store_path)

    assert list_mcp_server_configs(base_config={}, store_path=store_path) == [server]
    assert resolve_enabled_mcp_servers_config(base_config={}, store_path=store_path) == {
        "secure": {
            "transport": "streamable-http",
            "url": "https://mcp.example.com/mcp",
            "auth": {
                "type": "oauth_client_credentials",
                "token_url": "https://auth.example.com/oauth/token",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "scope": "read:mcp",
                "audience": "mcp-api",
                "grant_type": "client_credentials",
                "refresh_skew_seconds": 30,
            },
        }
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
            "auth": {
                "type": "none",
                "bearer_token_set": False,
                "token_url": None,
                "client_id": None,
                "client_secret_set": False,
                "scope": None,
                "audience": None,
                "grant_type": "client_credentials",
                "refresh_skew_seconds": 30,
            },
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
            "auth": {
                "type": "none",
                "bearer_token_set": False,
                "token_url": None,
                "client_id": None,
                "client_secret_set": False,
                "scope": None,
                "audience": None,
                "grant_type": "client_credentials",
                "refresh_skew_seconds": 30,
            },
        }
    ]


def test_mcp_config_route_stores_oauth_without_echoing_secret(tmp_path) -> None:
    settings = SimpleNamespace(
        ENABLE_MCP_TOOLS=True,
        MCP_UI_CONFIG_FILE=str(tmp_path / "mcp_servers.json"),
        MCP_SERVERS_CONFIG={},
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

    response = client.put(
        "/api/config/mcp-servers/secure",
        json={
            "transport": "streamable-http",
            "url": "https://mcp.example.com/mcp",
            "enabled": True,
            "auth": {
                "type": "oauth_client_credentials",
                "token_url": "https://auth.example.com/oauth/token",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "scope": "read:mcp",
                "audience": "mcp-api",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["auth"] == {
        "type": "oauth_client_credentials",
        "bearer_token_set": False,
        "token_url": "https://auth.example.com/oauth/token",
        "client_id": "client-id",
        "client_secret_set": True,
        "scope": "read:mcp",
        "audience": "mcp-api",
        "grant_type": "client_credentials",
        "refresh_skew_seconds": 30,
    }
    assert "client-secret" not in response.text

    response = client.put(
        "/api/config/mcp-servers/secure",
        json={
            "transport": "streamable-http",
            "url": "https://mcp.example.com/mcp",
            "enabled": True,
            "auth": {
                "type": "oauth_client_credentials",
                "token_url": "https://auth.example.com/oauth/token",
                "client_id": "client-id",
                "scope": "write:mcp",
            },
        },
    )

    assert response.status_code == 200
    stored = list_mcp_server_configs(
        base_config={},
        store_path=str(tmp_path / "mcp_servers.json"),
    )[0]
    assert stored.auth.client_secret == "client-secret"
    assert stored.auth.scope == "write:mcp"


def test_config_route_exposes_observability_links_without_secrets(tmp_path) -> None:
    settings = SimpleNamespace(
        REGION="us-chicago-1",
        EMBED_MODEL_ID="embed",
        MODEL_LIST=["cohere.command-r-plus"],
        MODEL_DISPLAY_NAMES={"cohere.command-r-plus": "Command R+"},
        COLLECTION_LIST=[],
        DEFAULT_COLLECTION="RAG_KNOWLEDGE_BASE",
        ENABLE_USER_FEEDBACK=True,
        ENABLE_LANGFUSE_TRACING=True,
        LANGFUSE_HOST="http://langfuse-web:3000",
        LANGFUSE_UI_URL="http://localhost:3300",
        LANGFUSE_PUBLIC_KEY="pk-lf-real-public",
        LANGFUSE_SECRET_KEY="sk-lf-real-secret",
        ENABLE_OBSERVABILITY_STACK=True,
        GRAFANA_URL="http://localhost:3051",
        ENABLE_OCI_LOGGING_ANALYTICS=True,
        LOGGING_ANALYTICS_NAMESPACE="namespace",
        LOGGING_ANALYTICS_LOG_GROUP_ID="ocid1.loggroup.example",
        LOGGING_ANALYTICS_CONSOLE_URL="https://cloud.oracle.com/loganalytics/explorer",
    )
    app = FastAPI()
    app.state.resources = SimpleNamespace(settings=settings)
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["collection_list"] == ["RAG_KNOWLEDGE_BASE"]
    links = {item["key"]: item for item in payload["observability"]["links"]}
    assert links["langfuse"] == {
        "key": "langfuse",
        "label": "Langfuse",
        "enabled": True,
        "configured": True,
        "url": "http://localhost:3300",
        "status": "Ready",
        "details": "Tracing is enabled and the Langfuse host is configured.",
    }
    assert links["grafana"]["url"] == "http://localhost:3051"
    assert links["oracle_logging_analytics"]["url"] == (
        "https://cloud.oracle.com/loganalytics/explorer"
    )
    serialized = response.text
    assert "sk-lf-real-secret" not in serialized
    assert "pk-lf-real-public" not in serialized


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
            "auth": {
                "type": "bearer",
                "bearer_token": "draft-token",
            },
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
                    "auth": {
                        "type": "bearer",
                        "bearer_token": "draft-token",
                    },
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


def test_resolve_store_path_uses_project_root_for_relative_paths(monkeypatch) -> None:
    def _unexpected_cwd(cls) -> Path:  # pragma: no cover - should never run
        raise AssertionError("resolve_store_path should not use Path.cwd()")

    monkeypatch.setattr(mod.Path, "cwd", classmethod(_unexpected_cwd))

    resolved = mod.resolve_store_path(".local-data/mcp_servers.json")

    assert resolved == mod.PROJECT_ROOT / ".local-data/mcp_servers.json"


def test_resolve_store_path_preserves_absolute_paths() -> None:
    absolute = Path("/tmp/custom-mcp.json")

    resolved = mod.resolve_store_path(absolute)

    assert resolved == absolute
