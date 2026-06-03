import json
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent

from src.rag_agent.infrastructure import mcp_adapter_runtime as mod
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


def test_select_server_keys_ignores_legacy_mcp_url_filter() -> None:
    configured = {
        "default": {"url": "http://localhost:9000/mcp"},
        "calculator": {"url": "http://localhost:9001/mcp"},
    }

    selected = mod._select_server_keys(
        configured_servers=configured,
        server_keys=None,
        run_config={"configurable": {"mcp_url": "http://localhost:9000/mcp"}},
    )

    assert selected == ["default", "calculator"]


def test_build_adapter_server_configs_applies_jwt_headers_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(
            enable_mcp_tools=True,
            enable_mcp_client_jwt=True,
            jwt_headers_supplier=lambda: {"Authorization": "Bearer test-token"},
        ),
    )
    monkeypatch.setattr(
        mod,
        "get_mcp_servers_config",
        lambda: {
            "default": {
                "transport": "streamable-http",
                "url": "http://localhost:9000/mcp",
                "headers": {"x-existing": "1"},
            }
        },
    )

    out = mod.build_adapter_server_configs(server_keys=None, run_config=None)

    assert "default" in out
    headers = out["default"].get("headers")
    assert headers == {
        "x-existing": "1",
        "Authorization": "Bearer test-token",
    }


def test_build_adapter_server_configs_applies_bearer_auth(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True, enable_mcp_client_jwt=False),
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

    out = mod.build_adapter_server_configs(server_keys=None, run_config=None)

    assert out["secure"]["headers"] == {"Authorization": "Bearer server-token"}
    assert "auth" not in out["secure"]


def test_per_server_auth_overrides_global_jwt(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(
            enable_mcp_tools=True,
            enable_mcp_client_jwt=True,
            jwt_headers_supplier=lambda: {"Authorization": "Bearer global-token"},
        ),
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

    out = mod.build_adapter_server_configs(server_keys=None, run_config=None)

    assert out["secure"]["headers"] == {"Authorization": "Bearer server-token"}


def test_build_adapter_server_configs_applies_per_server_oauth(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True, enable_mcp_client_jwt=False),
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

    out = mod.build_adapter_server_configs(server_keys=None, run_config=None)

    assert captured["client_id"] == "client-id"
    assert captured["client_secret"] == "client-secret"
    assert captured["scope"] == "read:mcp"
    assert captured["audience"] == "mcp-api"
    assert out["secure"]["headers"] == {"Authorization": "Bearer oauth-token"}
    assert "auth" not in out["secure"]


def test_build_adapter_server_configs_uses_run_config_override(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(enable_mcp_tools=True, enable_mcp_client_jwt=False),
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

    out = mod.build_adapter_server_configs(
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

    assert out == {
        "draft": {
            "transport": "streamable-http",
            "url": "http://localhost:9100/mcp",
        }
    }


def test_get_mcp_settings_enables_oauth_supplier(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MCP_TOOLS", "true")
    monkeypatch.setenv("ENABLE_MCP_CLIENT_JWT", "true")
    monkeypatch.setenv("ENABLE_MCP_OAUTH", "true")
    monkeypatch.setenv("MCP_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("MCP_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("MCP_OAUTH_TOKEN_URL", "https://auth.example.com/oauth/token")
    monkeypatch.setenv("MCP_OAUTH_SCOPE", "read:mcp")

    fake_settings = SimpleNamespace(
        ENABLE_MCP_OAUTH=True,
        MCP_OAUTH_CLIENT_ID="client-id",
        MCP_OAUTH_CLIENT_SECRET="client-secret",
        MCP_OAUTH_TOKEN_URL="https://auth.example.com/oauth/token",
        MCP_OAUTH_SCOPE="read:mcp",
        MCP_OAUTH_AUDIENCE=None,
        MCP_OAUTH_GRANT_TYPE="client_credentials",
        MCP_OAUTH_REFRESH_SKEW_SECONDS=30,
    )
    monkeypatch.setattr(
        "src.rag_agent.infrastructure.mcp_settings._get_app_settings", lambda: fake_settings
    )

    settings = get_mcp_settings()

    assert callable(settings.jwt_headers_supplier)


def test_get_mcp_settings_uses_app_settings_for_oauth_supplier(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_MCP_OAUTH", raising=False)
    monkeypatch.delenv("MCP_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("MCP_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("MCP_OAUTH_TOKEN_URL", raising=False)
    monkeypatch.delenv("MCP_OAUTH_SCOPE", raising=False)

    fake_settings = SimpleNamespace(
        ENABLE_MCP_OAUTH=True,
        MCP_OAUTH_CLIENT_ID="client-id",
        MCP_OAUTH_CLIENT_SECRET="client-secret",
        MCP_OAUTH_TOKEN_URL="https://auth.example.com/oauth/token",
        MCP_OAUTH_SCOPE="read:mcp",
        MCP_OAUTH_AUDIENCE=None,
        MCP_OAUTH_GRANT_TYPE="client_credentials",
        MCP_OAUTH_REFRESH_SKEW_SECONDS=30,
    )
    monkeypatch.setattr(
        "src.rag_agent.infrastructure.mcp_settings._get_app_settings", lambda: fake_settings
    )
    monkeypatch.setenv("ENABLE_MCP_TOOLS", "true")
    monkeypatch.setenv("ENABLE_MCP_CLIENT_JWT", "true")

    settings = get_mcp_settings()

    assert callable(settings.jwt_headers_supplier)


def test_build_adapter_server_configs_raises_when_supplier_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_mcp_settings",
        lambda: SimpleNamespace(
            enable_mcp_tools=True,
            enable_mcp_client_jwt=True,
            jwt_headers_supplier=lambda: (_ for _ in ()).throw(RuntimeError("token fetch failed")),
        ),
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

    with pytest.raises(RuntimeError, match="token fetch failed"):
        mod.build_adapter_server_configs(server_keys=None, run_config=None)


def test_create_client_raises_when_callbacks_supplier_fails() -> None:
    settings = SimpleNamespace(
        mcp_client_callbacks=None,
        mcp_tool_interceptors=None,
        mcp_client_callbacks_supplier=lambda: (_ for _ in ()).throw(
            RuntimeError("callbacks failed")
        ),
        mcp_tool_interceptors_supplier=None,
    )

    with pytest.raises(RuntimeError, match="callbacks failed"):
        mod._create_client(
            {"default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}},
            settings=settings,
        )


def test_create_client_raises_when_interceptors_supplier_fails() -> None:
    settings = SimpleNamespace(
        mcp_client_callbacks=None,
        mcp_tool_interceptors=None,
        mcp_client_callbacks_supplier=None,
        mcp_tool_interceptors_supplier=lambda: (_ for _ in ()).throw(
            RuntimeError("interceptors failed")
        ),
    )

    with pytest.raises(RuntimeError, match="interceptors failed"):
        mod._create_client(
            {"default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}},
            settings=settings,
        )


def test_normalize_connection_config_passes_through_supported_optional_fields() -> None:
    out = mod._normalize_connection_config(
        {
            "transport": "streamable-http",
            "url": "http://localhost:9000/mcp",
            "timeout": 10,
            "sse_read_timeout": 15,
            "session_kwargs": {"a": 1},
            "terminate_on_close": True,
            "auth": object(),
            "httpx_client_factory": object(),
            "cwd": "/tmp",
            "encoding": "utf-8",
            "encoding_error_handler": "replace",
        }
    )

    assert out["transport"] == "streamable-http"
    assert out["url"] == "http://localhost:9000/mcp"
    assert out["timeout"] == 10
    assert out["sse_read_timeout"] == 15
    assert out["session_kwargs"] == {"a": 1}
    assert out["terminate_on_close"] is True
    assert out["cwd"] == "/tmp"
    assert out["encoding"] == "utf-8"
    assert out["encoding_error_handler"] == "replace"
    assert "auth" in out
    assert "httpx_client_factory" in out


def test_create_client_wires_callbacks_and_interceptors(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(mod, "MultiServerMCPClient", FakeClient)

    settings = SimpleNamespace(
        mcp_client_callbacks=object(),
        mcp_tool_interceptors=[object()],
        mcp_client_callbacks_supplier=None,
        mcp_tool_interceptors_supplier=None,
    )
    _ = mod._create_client(
        {"default": {"transport": "streamable-http", "url": "http://localhost:9000/mcp"}},
        settings=settings,
    )

    assert captured["tool_name_prefix"] is True
    assert "callbacks" in captured
    assert "tool_interceptors" in captured


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


def test_normalize_call_tool_result_moves_error_in_structured_and_text_payload() -> None:
    call_result = CallToolResult(
        isError=False,
        structuredContent={
            "command": "os ns get",
            "returncode": 0,
            "error": "warning text",
        },
        content=[
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "command": "os ns get",
                        "returncode": 0,
                        "error": "warning text",
                    }
                ),
            )
        ],
    )

    normalized = mod._normalize_call_tool_result(call_result)
    assert isinstance(normalized, CallToolResult)

    structured = normalized.structuredContent or {}
    assert "error" not in structured
    assert structured.get("warnings") == ["warning text"]

    assert isinstance(normalized.content[0], TextContent)
    text_payload = normalized.content[0].text
    assert isinstance(text_payload, str)
    parsed_text_payload = json.loads(text_payload)
    assert "error" not in parsed_text_payload
    assert parsed_text_payload.get("warnings") == ["warning text"]
