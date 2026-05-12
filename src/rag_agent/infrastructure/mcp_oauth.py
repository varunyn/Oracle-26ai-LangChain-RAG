import json
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class MCPAuthError(RuntimeError):
    """Raised when MCP auth setup or token retrieval fails."""


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

    def get_headers(self) -> dict[str, str]:
        now = time.time()
        token = self._cached_token
        if token is None or token.is_expiring_within(self._refresh_skew_seconds, now=now):
            token = self._request_token()
            self._cached_token = token
        return {"Authorization": f"{token.token_type} {token.access_token}"}

    def _request_token(self) -> OAuthTokenResponse:
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
        request = Request(
            self._token_url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                raw_payload = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
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

    provider = OAuthClientCredentialsProvider(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
        audience=audience,
        grant_type=grant_type,
        refresh_skew_seconds=refresh_skew_seconds,
    )
    return provider.get_headers
