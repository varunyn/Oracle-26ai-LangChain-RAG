"""
Request-scoped context middleware for FastAPI.

Provides request ID binding and context clearing to prevent leaks between requests.
"""

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.dependencies import generate_request_id

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware:
    """Set request ID from header or generate one; inject into context and response.

    Ensures context is cleared in finally block to prevent leaks between requests.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = Headers(scope=scope).get(REQUEST_ID_HEADER) or generate_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        # Bind context and get token for restoration
        from src.rag_agent.utils.logging_config import REQUEST_ID_CTX

        token = REQUEST_ID_CTX.set(request_id)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            # Clear context to prevent leaks
            REQUEST_ID_CTX.reset(token)
