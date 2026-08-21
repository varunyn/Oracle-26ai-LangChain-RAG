from pathlib import Path

import pytest
from pydantic import ValidationError

from api.settings import Settings


def test_oracle_knowledge_mapping_and_allowlist_parse_from_json(monkeypatch):
    monkeypatch.setenv("ORACLE_KNOWLEDGE_BASES", '{"docs":"RAW_DOCS","handbook":"RAW_HANDBOOK"}')
    monkeypatch.setenv("ORACLE_KNOWLEDGE_ALLOWED_KEYS", '["docs", "handbook"]')

    settings = Settings(_env_file=None)

    assert settings.ORACLE_KNOWLEDGE_BASES == {
        "docs": "RAW_DOCS",
        "handbook": "RAW_HANDBOOK",
    }
    assert settings.ORACLE_KNOWLEDGE_ALLOWED_KEYS == ["docs", "handbook"]


def test_oracle_knowledge_namespaced_defaults_are_independent():
    settings = Settings(_env_file=None)

    assert settings.ORACLE_KNOWLEDGE_TRANSPORT == "stdio"
    assert settings.ORACLE_KNOWLEDGE_HOST == "127.0.0.1"
    assert settings.ORACLE_KNOWLEDGE_PORT == 9000
    assert settings.ORACLE_KNOWLEDGE_ENABLE_OTEL_TRACING is False


def test_oracle_compose_otel_is_namespaced_and_container_safe():
    text = (Path(__file__).parents[2] / "docker-compose.oracle-knowledge.yml").read_text()
    assert "ENABLE_OTEL_TRACING: ${ORACLE_KNOWLEDGE_ENABLE_OTEL_TRACING:-false}" in text
    assert "host.docker.internal:4318" in text
    assert '"host.docker.internal:host-gateway"' in text


@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
def test_oracle_knowledge_accepts_only_supported_transports(monkeypatch, transport):
    monkeypatch.setenv("ORACLE_KNOWLEDGE_TRANSPORT", transport)
    assert Settings(_env_file=None).ORACLE_KNOWLEDGE_TRANSPORT == transport


@pytest.mark.parametrize("transport", ["sse", "http", "grpc", "invalid"])
def test_oracle_knowledge_rejects_unsupported_transports(monkeypatch, transport):
    monkeypatch.setenv("ORACLE_KNOWLEDGE_TRANSPORT", transport)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_oracle_knowledge_bounds_are_validated(monkeypatch):
    monkeypatch.setenv("ORACLE_KNOWLEDGE_MAX_QUERY_LENGTH", "100000")
    monkeypatch.setenv("ORACLE_KNOWLEDGE_MAX_RESULT_LIMIT", "100")
    monkeypatch.setenv("ORACLE_KNOWLEDGE_MAX_CANDIDATE_LIMIT", "100")
    monkeypatch.setenv("ORACLE_KNOWLEDGE_MAX_METADATA_FILTERS", "32")
    monkeypatch.setenv("ORACLE_KNOWLEDGE_TIMEOUT_SECONDS", "300")
    settings = Settings(_env_file=None)
    assert settings.ORACLE_KNOWLEDGE_MAX_QUERY_LENGTH == 100000
    assert settings.ORACLE_KNOWLEDGE_TIMEOUT_SECONDS == 300

    monkeypatch.setenv("ORACLE_KNOWLEDGE_MAX_RESULT_LIMIT", "101")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
