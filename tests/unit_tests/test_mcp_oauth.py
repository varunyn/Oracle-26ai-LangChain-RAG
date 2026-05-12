import json
from urllib.parse import parse_qs

import pytest

from src.rag_agent.infrastructure.mcp_oauth import (
    MCPAuthError,
    OAuthClientCredentialsProvider,
    OAuthTokenResponse,
)


def test_get_headers_fetches_bearer_token(monkeypatch):
    provider = OAuthClientCredentialsProvider(
        token_url="https://auth.example.com/oauth/token",
        client_id="client-id",
        client_secret="client-secret",
        scope="read:mcp",
        audience=None,
        grant_type="client_credentials",
        refresh_skew_seconds=30,
    )

    def fake_request_token() -> OAuthTokenResponse:
        return OAuthTokenResponse(
            access_token="token-123",
            token_type="Bearer",
            expires_at=2_000_000_000,
        )

    monkeypatch.setattr(provider, "_request_token", fake_request_token)

    assert provider.get_headers() == {"Authorization": "Bearer token-123"}


def test_get_headers_reuses_cached_token_when_not_near_expiry(monkeypatch):
    provider = OAuthClientCredentialsProvider(
        token_url="https://auth.example.com/oauth/token",
        client_id="client-id",
        client_secret="client-secret",
        scope="read:mcp",
        audience=None,
        grant_type="client_credentials",
        refresh_skew_seconds=30,
    )

    calls = {"count": 0}

    def fake_request_token() -> OAuthTokenResponse:
        calls["count"] += 1
        return OAuthTokenResponse(
            access_token="cached-token",
            token_type="Bearer",
            expires_at=2_000_000_000,
        )

    monkeypatch.setattr(provider, "_request_token", fake_request_token)

    first = provider.get_headers()
    second = provider.get_headers()

    assert first == {"Authorization": "Bearer cached-token"}
    assert second == {"Authorization": "Bearer cached-token"}
    assert calls["count"] == 1


def test_get_headers_refreshes_when_token_is_near_expiry(monkeypatch):
    provider = OAuthClientCredentialsProvider(
        token_url="https://auth.example.com/oauth/token",
        client_id="client-id",
        client_secret="client-secret",
        scope="read:mcp",
        audience=None,
        grant_type="client_credentials",
        refresh_skew_seconds=30,
    )

    issued_tokens = iter(
        [
            OAuthTokenResponse("old-token", "Bearer", 100.0),
            OAuthTokenResponse("new-token", "Bearer", 1_000.0),
        ]
    )

    monkeypatch.setattr(provider, "_request_token", lambda: next(issued_tokens))
    monkeypatch.setattr("src.rag_agent.infrastructure.mcp_oauth.time.time", lambda: 90.0)

    provider.get_headers()
    refreshed = provider.get_headers()

    assert refreshed == {"Authorization": "Bearer new-token"}


def test_parse_token_response_requires_access_token():
    with pytest.raises(MCPAuthError, match="missing access_token"):
        OAuthClientCredentialsProvider.parse_token_response(
            {"token_type": "Bearer", "expires_in": 3600},
            now=100.0,
        )


def test_parse_token_response_requires_numeric_expires_in():
    with pytest.raises(MCPAuthError, match="invalid expires_in"):
        OAuthClientCredentialsProvider.parse_token_response(
            {
                "access_token": "token-123",
                "token_type": "Bearer",
                "expires_in": object(),
            },
            now=100.0,
        )


def test_parse_token_response_rejects_negative_expires_in():
    with pytest.raises(MCPAuthError, match="invalid expires_in"):
        OAuthClientCredentialsProvider.parse_token_response(
            {
                "access_token": "token-123",
                "token_type": "Bearer",
                "expires_in": -1,
            },
            now=100.0,
        )


def test_parse_token_response_defaults_blank_token_type_to_bearer():
    response = OAuthClientCredentialsProvider.parse_token_response(
        {
            "access_token": "token-123",
            "token_type": "",
            "expires_in": 3600,
        },
        now=100.0,
    )

    assert response.token_type == "Bearer"


def test_request_token_posts_client_credentials_form(monkeypatch):
    provider = OAuthClientCredentialsProvider(
        token_url="https://auth.example.com/oauth/token",
        client_id="client-id",
        client_secret="client-secret",
        scope="read:mcp",
        audience="https://mcp.example.com",
        grant_type="client_credentials",
        refresh_skew_seconds=30,
    )

    captured: dict[str, object] = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200

        def read(self) -> bytes:
            return json.dumps(
                {
                    "access_token": "token-123",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.rag_agent.infrastructure.mcp_oauth.urlopen", fake_urlopen)
    monkeypatch.setattr("src.rag_agent.infrastructure.mcp_oauth.time.time", lambda: 100.0)

    response = provider._request_token()
    body = captured["body"]
    assert isinstance(body, str)
    parsed_body = parse_qs(body)

    assert captured["url"] == "https://auth.example.com/oauth/token"
    assert captured["method"] == "POST"
    assert parsed_body["grant_type"] == ["client_credentials"]
    assert parsed_body["client_id"] == ["client-id"]
    assert parsed_body["client_secret"] == ["client-secret"]
    assert parsed_body["scope"] == ["read:mcp"]
    assert parsed_body["audience"] == ["https://mcp.example.com"]
    assert response.access_token == "token-123"
