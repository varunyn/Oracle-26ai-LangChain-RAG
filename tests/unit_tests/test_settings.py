from __future__ import annotations

from api.settings import Settings


def test_mcp_max_rounds_loads_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("MCP_MAX_ROUNDS", "5")

    settings = Settings(_env_file=None)

    assert settings.MCP_MAX_ROUNDS == 5
