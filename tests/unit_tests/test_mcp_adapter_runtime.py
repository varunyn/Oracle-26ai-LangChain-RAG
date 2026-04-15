import json
from types import SimpleNamespace

from mcp.types import CallToolResult, TextContent

from src.rag_agent.infrastructure import mcp_adapter_runtime as mod


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

    text_payload = normalized.content[0].text
    assert isinstance(text_payload, str)
    parsed_text_payload = json.loads(text_payload)
    assert "error" not in parsed_text_payload
    assert parsed_text_payload.get("warnings") == ["warning text"]
