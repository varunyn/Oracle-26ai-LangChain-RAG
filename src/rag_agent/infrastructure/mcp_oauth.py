import asyncio
import json
import time
from dataclasses import dataclass
from typing import TypeAlias
from urllib.parse import urlencode

import httpx


class MCPAuthError(RuntimeError):
    """Raised when MCP auth setup or token retrieval fails."""


OAuthProviderIdentity: TypeAlias = tuple[str, str, str | None, str | None, str]
_oauth_provider_cache: dict[OAuthProviderIdentity, "OAuthClientCredentialsProvider"] = {}


@dataclass(slots=True)
class OAuthTokenResponse:
    access_token: str
    token_type: str
    expires_at: float

    def is_expiring_within(self, skew_seconds: int, *, now: float) -> bool:
        return now >= self.expires_at - skew_seconds


class OAuthClientCredentialsProvider:
    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str | None,
        audience: str | None,
        grant_type: str,
        refresh_skew_seconds: int,
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._audience = audience
        self._grant_type = grant_type
        self._refresh_skew_seconds = refresh_skew_seconds
        self._cached_token: OAuthTokenResponse | None = None
        self._credential_generation = 0
        self._refresh_lock = asyncio.Lock()

    def update_credentials(self, client_secret: str, refresh_skew_seconds: int) -> None:
        if (
            client_secret == self._client_secret
            and refresh_skew_seconds == self._refresh_skew_seconds
        ):
            return
        self._client_secret = client_secret
        self._refresh_skew_seconds = refresh_skew_seconds
        self._cached_token = None
        self._credential_generation += 1

    @staticmethod
    def parse_token_response(payload: dict[str, object], *, now: float) -> OAuthTokenResponse:
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise MCPAuthError("MCP OAuth token response missing access_token")

        token_type = payload.get("token_type")
        normalized_token_type = (
            token_type if isinstance(token_type, str) and token_type.strip() else "Bearer"
        )

        expires_in_raw = payload.get("expires_in", 3600)
        if not isinstance(expires_in_raw, (int, float, str)):
            raise MCPAuthError("MCP OAuth token response has invalid expires_in")
        try:
            expires_in = int(expires_in_raw)
        except (TypeError, ValueError) as exc:
            raise MCPAuthError("MCP OAuth token response has invalid expires_in") from exc
        if expires_in <= 0:
            raise MCPAuthError("MCP OAuth token response has invalid expires_in")

        return OAuthTokenResponse(
            access_token=access_token,
            token_type=normalized_token_type,
            expires_at=now + expires_in,
        )

    async def get_headers(self) -> dict[str, str]:
        while True:
            token = self._cached_token
            if token is not None and not token.is_expiring_within(
                self._refresh_skew_seconds, now=time.time()
            ):
                return {"Authorization": f"{token.token_type} {token.access_token}"}

            async with self._refresh_lock:
                token = self._cached_token
                if token is not None and not token.is_expiring_within(
                    self._refresh_skew_seconds, now=time.time()
                ):
                    return {"Authorization": f"{token.token_type} {token.access_token}"}
                generation = self._credential_generation
                token = await self._request_token()
                if generation != self._credential_generation:
                    continue
                self._cached_token = token
                return {"Authorization": f"{token.token_type} {token.access_token}"}

    async def _request_token(self) -> OAuthTokenResponse:
        form_fields: dict[str, str] = {
            "grant_type": self._grant_type,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scope:
            form_fields["scope"] = self._scope
        if self._audience:
            form_fields["audience"] = self._audience

        body = urlencode(form_fields).encode("utf-8")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    self._token_url,
                    content=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                raw_payload = response.text
        except (httpx.HTTPError, TimeoutError) as exc:
            raise MCPAuthError("MCP OAuth token request failed") from exc
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise MCPAuthError("MCP OAuth token response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise MCPAuthError("MCP OAuth token response was not a JSON object")
        return self.parse_token_response(payload, now=time.time())


def build_oauth_headers_supplier(
    *,
    token_url: str | None,
    client_id: str | None,
    client_secret: str | None,
    scope: str | None,
    audience: str | None,
    grant_type: str,
    refresh_skew_seconds: int,
):
    if not token_url or not client_id or not client_secret:
        raise MCPAuthError("MCP OAuth requires token URL, client ID, and client secret")

    identity: OAuthProviderIdentity = (token_url, client_id, scope, audience, grant_type)
    provider = _oauth_provider_cache.get(identity)
    if provider is None:
        provider = OAuthClientCredentialsProvider(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
            audience=audience,
            grant_type=grant_type,
            refresh_skew_seconds=refresh_skew_seconds,
        )
        _oauth_provider_cache[identity] = provider
    else:
        provider.update_credentials(client_secret, refresh_skew_seconds)
    return provider.get_headers


def clear_oauth_provider_cache() -> None:
    _oauth_provider_cache.clear()
