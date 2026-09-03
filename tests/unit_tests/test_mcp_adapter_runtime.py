import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool

from src.rag_agent.infrastructure import mcp_adapter_runtime as mod
from src.rag_agent.infrastructure import mcp_oauth
from src.rag_agent.infrastructure.mcp_settings import get_mcp_settings


def test_select_server_keys_defaults_to_all_configured_when_no_filters() -> None:
    configured = {
        "default": {"url": "http://localhost:9000/mcp"},
        "calculator": {"url": "http://localhost:9001/mcp"},
    }

    selected = mod._select_server_keys(
        configured_servers=configured,
        server_keys=None,
        run_config=None,
    )

    assert selected == ["default", "calculator"]


def test_build_adapter_server_configs_applies_bearer_auth(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True),
    )
    monkeypatch.setattr(
        mod,
        "get_mcp_servers_config",
        lambda: {
            "secure": {
                "transport": "streamable-http",
                "url": "https://mcp.example.com/mcp",
                "auth": {
                    "type": "bearer",
                    "bearer_token": "server-token",
                },
            }
        },
    )

    out = asyncio.run(mod.build_adapter_server_configs(server_keys=None, run_config=None))

    assert out["secure"]["headers"] == {"Authorization": "Bearer server-token"}
    assert "auth" not in out["secure"]


def test_build_adapter_server_configs_applies_per_server_oauth(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True),
    )
    monkeypatch.setattr(
        mod,
        "get_mcp_servers_config",
        lambda: {
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
        },
    )
    captured: dict[str, object] = {}

    def fake_build_oauth_headers_supplier(**kwargs: object):
        captured.update(kwargs)
        return lambda: {"Authorization": "Bearer oauth-token"}

    monkeypatch.setattr(mod, "build_oauth_headers_supplier", fake_build_oauth_headers_supplier)

    out = asyncio.run(mod.build_adapter_server_configs(server_keys=None, run_config=None))

    assert captured["client_id"] == "client-id"
    assert captured["client_secret"] == "client-secret"
    assert captured["scope"] == "read:mcp"
    assert captured["audience"] == "mcp-api"
    assert out["secure"]["headers"] == {"Authorization": "Bearer oauth-token"}
    assert "auth" not in out["secure"]


def test_oauth_provider_cache_spans_repeated_concurrent_config_builds(monkeypatch) -> None:
    mcp_oauth.clear_oauth_provider_cache()
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True),
    )
    monkeypatch.setattr(
        mod,
        "get_mcp_servers_config",
        lambda: {
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
        },
    )
    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, access_token: str) -> None:
            self.text = json.dumps({"access_token": access_token, "expires_in": 3600})

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *, timeout: int) -> None:
            del timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            del args, kwargs
            calls["count"] += 1
            await asyncio.sleep(0)
            return FakeResponse(f"token-{calls['count']}")

    monkeypatch.setattr(mcp_oauth.httpx, "AsyncClient", FakeAsyncClient)

    async def exercise() -> None:
        await asyncio.gather(
            *(mod.build_adapter_server_configs(server_keys=None, run_config=None) for _ in range(5))
        )
        assert calls["count"] == 1
        for provider in mcp_oauth._oauth_provider_cache.values():
            provider._cached_token = mcp_oauth.OAuthTokenResponse("expired-token", "Bearer", 0.0)
        await asyncio.gather(
            *(mod.build_adapter_server_configs(server_keys=None, run_config=None) for _ in range(5))
        )

    try:
        asyncio.run(exercise())
        assert calls["count"] == 2
        assert all("client-secret" not in repr(key) for key in mcp_oauth._oauth_provider_cache)
    finally:
        mcp_oauth.clear_oauth_provider_cache()


def test_build_adapter_server_configs_uses_run_config_override(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True),
    )
    monkeypatch.setattr(
        mod,
        "get_mcp_servers_config",
        lambda: {
            "default": {
                "transport": "streamable-http",
                "url": "http://localhost:9000/mcp",
            }
        },
    )

    out = asyncio.run(
        mod.build_adapter_server_configs(
            server_keys=["draft"],
            run_config={
                "configurable": {
                    "mcp_servers_config_override": {
                        "draft": {
                            "transport": "streamable-http",
                            "url": "http://localhost:9100/mcp",
                        }
                    }
                }
            },
        )
    )

    assert out == {
        "draft": {
            "transport": "streamable-http",
            "url": "http://localhost:9100/mcp",
        }
    }


def test_empty_run_config_override_does_not_fall_back_to_saved_servers(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MCP_TOOLS", "true")
    monkeypatch.setattr(
        mod,
        "get_mcp_servers_config",
        lambda: {
            "saved": {
                "transport": "streamable-http",
                "url": "http://localhost:9000/mcp",
            }
        },
    )

    out = asyncio.run(
        mod.build_adapter_server_configs(
            run_config={"configurable": {"mcp_servers_config_override": {}}}
        )
    )

    assert out == {}


def test_get_mcp_settings_reads_enable_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.settings.get_settings",
        lambda: SimpleNamespace(ENABLE_MCP_TOOLS=True),
    )

    settings = get_mcp_settings()

    assert settings.enable_mcp_tools is True


def test_load_adapter_tools_evicts_and_closes_client_on_cancellation(monkeypatch) -> None:
    mod._client_cache.clear()
    mod._tool_cache.clear()
    closed = {"value": False}

    class CancelledAdapter:
        def __init__(self) -> None:
            self.client = SimpleNamespace(close=self.close)

        async def list_tools(self):
            raise asyncio.CancelledError

        async def close(self):
            closed["value"] = True

    client = CancelledAdapter()
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(
            enable_mcp_tools=True,
        ),
    )

    async def fake_build_adapter_server_configs(**kwargs):
        del kwargs
        return {"default": {"transport": "streamable-http", "url": "http://mcp"}}

    monkeypatch.setattr(mod, "build_adapter_server_configs", fake_build_adapter_server_configs)
    monkeypatch.setattr(mod, "_create_client", lambda connections: client)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mod.load_adapter_tools())

    assert closed["value"] is True
    assert mod._client_cache == {}
    assert mod._tool_cache == {}


def test_normalize_connection_config_passes_through_supported_optional_fields() -> None:
    out = mod._normalize_connection_config(
        {
            "transport": "streamable-http",
            "url": "http://localhost:9000/mcp",
            "timeout": 10,
            "sse_read_timeout": 15,
            "auth": object(),
            "cwd": "/tmp",
            "keep_alive": False,
        }
    )

    assert out["transport"] == "streamable-http"
    assert out["url"] == "http://localhost:9000/mcp"
    assert out["timeout"] == 10
    assert "sse_read_timeout" not in out
    assert out["cwd"] == "/tmp"
    assert out["keep_alive"] is False
    assert "auth" in out


def test_create_client_uses_first_party_adapter_and_group(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeGroup:
        @classmethod
        def from_config(cls, config):
            captured["config"] = config
            return cls()

    class FakeAdapter:
        def __init__(self, target):
            captured["target"] = target

    monkeypatch.setattr(mod, "ClientGroup", FakeGroup)
    monkeypatch.setattr(mod, "MCPAdapter", FakeAdapter)
    client = mod._create_client(
        {"default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}},
    )

    assert isinstance(client, FakeAdapter)
    assert captured["config"] == {
        "mcpServers": {
            "default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}
        }
    }
    assert captured["target"] is not None


def test_move_success_error_to_warnings_when_returncode_zero() -> None:
    payload = {
        "command": "os ns get",
        "output": {"data": "ns"},
        "returncode": 0,
        "error": "warning text",
    }

    normalized = mod._move_success_error_to_warnings(payload)

    assert "error" not in normalized
    assert normalized["warnings"] == ["warning text"]


def test_normalize_langchain_tool_result_moves_error_in_artifact_and_text_payload() -> None:
    result = (
        [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "command": "os ns get",
                        "returncode": 0,
                        "error": "warning text",
                    }
                ),
            }
        ],
        {
            "structured_content": {
                "command": "os ns get",
                "returncode": 0,
                "error": "warning text",
            }
        },
    )

    normalized = mod._normalize_langchain_tool_result(result)
    content, artifact = normalized

    structured = artifact["structured_content"]
    assert "error" not in structured
    assert structured.get("warnings") == ["warning text"]

    text_payload = content[0]["text"]
    assert isinstance(text_payload, str)
    parsed_text_payload = json.loads(text_payload)
    assert "error" not in parsed_text_payload
    assert parsed_text_payload.get("warnings") == ["warning text"]


def test_wrapped_adapter_tool_normalizes_tool_message_content_and_artifact() -> None:
    async def call_cli() -> tuple[list[dict[str, str]], dict[str, object]]:
        payload = {"returncode": 0, "error": "warning text"}
        return ([{"type": "text", "text": json.dumps(payload)}], {"structured_content": payload})

    tool = StructuredTool.from_function(
        coroutine=call_cli,
        name="cli",
        description="Run a CLI command",
        response_format="content_and_artifact",
    )
    wrapped = mod._normalize_adapter_tool(tool)
    message = asyncio.run(
        wrapped.ainvoke({"type": "tool_call", "name": "cli", "id": "call-1", "args": {}})
    )

    assert message.status == "success"
    assert message.artifact["structured_content"] == {
        "returncode": 0,
        "warnings": ["warning text"],
    }
    assert json.loads(message.content[0]["text"]) == {
        "returncode": 0,
        "warnings": ["warning text"],
    }
