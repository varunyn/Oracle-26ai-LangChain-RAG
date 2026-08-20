import asyncio
import json
import socket
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

    async def fake_request_token() -> OAuthTokenResponse:
        return OAuthTokenResponse(
            access_token="token-123",
            token_type="Bearer",
            expires_at=2_000_000_000,
        )

    monkeypatch.setattr(provider, "_request_token", fake_request_token)

    assert asyncio.run(provider.get_headers()) == {"Authorization": "Bearer token-123"}


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

    async def fake_request_token() -> OAuthTokenResponse:
        calls["count"] += 1
        return OAuthTokenResponse(
            access_token="cached-token",
            token_type="Bearer",
            expires_at=2_000_000_000,
        )

    monkeypatch.setattr(provider, "_request_token", fake_request_token)

    first = asyncio.run(provider.get_headers())
    second = asyncio.run(provider.get_headers())

    assert first == {"Authorization": "Bearer cached-token"}
    assert second == {"Authorization": "Bearer cached-token"}
    assert calls["count"] == 1


def test_secret_rotation_during_refresh_cannot_publish_old_token(monkeypatch):
    provider = OAuthClientCredentialsProvider(
        token_url="https://auth.example.com/oauth/token",
        client_id="client-id",
        client_secret="old-secret",
        scope=None,
        audience=None,
        grant_type="client_credentials",
        refresh_skew_seconds=30,
    )
    request_secrets: list[str] = []
    refresh_started = asyncio.Event()
    release_old_refresh = asyncio.Event()

    class FakeResponse:
        def __init__(self, token: str) -> None:
            self.text = json.dumps({"access_token": token, "expires_in": 3600})

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *, timeout: int) -> None:
            del timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, content, headers):
            del url, headers
            form = parse_qs(content.decode("utf-8"))
            secret = form["client_secret"][0]
            request_secrets.append(secret)
            if secret == "old-secret":
                refresh_started.set()
                await release_old_refresh.wait()
                return FakeResponse("old-token")
            return FakeResponse("new-token")

    monkeypatch.setattr("src.rag_agent.infrastructure.mcp_oauth.httpx.AsyncClient", FakeAsyncClient)

    async def exercise() -> dict[str, str]:
        refresh_task = asyncio.create_task(provider.get_headers())
        await refresh_started.wait()
        provider.update_credentials("new-secret", 30)
        release_old_refresh.set()
        return await refresh_task

    headers = asyncio.run(exercise())

    assert request_secrets == ["old-secret", "new-secret"]
    assert headers == {"Authorization": "Bearer new-token"}
    assert provider._cached_token is not None
    assert provider._cached_token.access_token == "new-token"


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

    async def fake_request_token() -> OAuthTokenResponse:
        return next(issued_tokens)

    monkeypatch.setattr(provider, "_request_token", fake_request_token)
    monkeypatch.setattr("src.rag_agent.infrastructure.mcp_oauth.time.time", lambda: 90.0)

    asyncio.run(provider.get_headers())
    refreshed = asyncio.run(provider.get_headers())

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
        text = json.dumps({"access_token": "token-123", "token_type": "Bearer", "expires_in": 3600})

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, content, headers):
            captured["url"] = url
            captured["method"] = "POST"
            captured["headers"] = headers
            captured["body"] = content.decode("utf-8")
            return FakeResponse()

    monkeypatch.setattr("src.rag_agent.infrastructure.mcp_oauth.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("src.rag_agent.infrastructure.mcp_oauth.time.time", lambda: 100.0)

    response = asyncio.run(provider._request_token())
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


def test_async_token_request_does_not_use_blocking_socket_io(monkeypatch):
    provider = OAuthClientCredentialsProvider(
        token_url="https://auth.example.com/oauth/token",
        client_id="client-id",
        client_secret="client-secret",
        scope=None,
        audience=None,
        grant_type="client_credentials",
        refresh_skew_seconds=30,
    )

    class FakeResponse:
        text = json.dumps({"access_token": "token-123", "expires_in": 3600})

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    def fail_if_socket_connects(*args, **kwargs):
        raise AssertionError("OAuth token retrieval performed blocking socket I/O")

    monkeypatch.setattr("src.rag_agent.infrastructure.mcp_oauth.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(socket.socket, "connect", fail_if_socket_connects)

    response = asyncio.run(provider._request_token())

    assert response.access_token == "token-123"
