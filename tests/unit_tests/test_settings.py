from __future__ import annotations

from api.settings import Settings


def test_mcp_max_rounds_default_allows_mixed_retrieval_and_tools() -> None:
    settings = Settings(_env_file=None)

    assert settings.MCP_MAX_ROUNDS == 4


def test_mcp_max_rounds_loads_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("MCP_MAX_ROUNDS", "5")

    settings = Settings(_env_file=None)

    assert settings.MCP_MAX_ROUNDS == 5


def test_search_mode_settings_only_allow_vector(monkeypatch) -> None:
    monkeypatch.setenv("RAG_SEARCH_MODE", "hybrid")

    settings = Settings(_env_file=None)

    assert settings.RAG_SEARCH_MODE == "vector"
