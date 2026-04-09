import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from src.rag_agent.infrastructure import code_mode_client


def test_register_mcp_bundle_falls_back_on_streamable_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers = cast(
        code_mode_client.MCPServersConfig,
        {
            "default": {
                "transport": "streamable-http",
                "url": "http://test",
            }
        },
    )
    built_templates: list[code_mode_client.MCPServersConfig] = []

    def fake_build_call_template(servers_arg: code_mode_client.MCPServersConfig) -> object:
        built_templates.append(servers_arg)
        return object()

    register_manual = AsyncMock(
        side_effect=[Exception("transport streamable-http unsupported"), None]
    )
    mock_client = cast(
        Any,
        cast(object, SimpleNamespace(register_manual=register_manual)),
    )

    monkeypatch.setattr(code_mode_client, "_build_call_template", fake_build_call_template)

    asyncio.run(
        code_mode_client._register_mcp_bundle(  # pyright: ignore[reportPrivateUsage]
            mock_client,
            servers,
        )
    )

    assert len(built_templates) == 2
    first_default = cast(dict[str, object], built_templates[0]["default"])
    assert first_default["transport"] == "streamable-http"
    assert first_default["url"] == "http://test"
    assert "auth" in first_default and first_default["auth"] is None
    mapped_default = cast(dict[str, object], built_templates[1]["default"])
    assert mapped_default["transport"] == "http"
    assert mapped_default["url"] == "http://test"
    assert register_manual.await_count == 2
