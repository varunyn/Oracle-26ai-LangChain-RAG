from __future__ import annotations

from api.settings import Settings


def test_mcp_max_rounds_loads_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("MCP_MAX_ROUNDS", "5")
    monkeypatch.setenv("MCP_USE_LLM_TOOL_SELECTOR", "true")

    settings = Settings(_env_file=None)

    assert settings.MCP_MAX_ROUNDS == 5
    assert settings.MCP_USE_LLM_TOOL_SELECTOR is True


def test_search_mode_settings_only_allow_vector(monkeypatch) -> None:
    monkeypatch.setenv("RAG_SEARCH_MODE", "hybrid")
    monkeypatch.setenv("MCP_SEARCH_MODE", "text")

    settings = Settings(_env_file=None)

    assert settings.RAG_SEARCH_MODE == "vector"
    assert settings.MCP_SEARCH_MODE == "vector"
