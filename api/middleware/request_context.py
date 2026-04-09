"""
Request-scoped context middleware for FastAPI.

Provides request ID binding and context clearing to prevent leaks between requests.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from api.dependencies import generate_request_id

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Set request ID from header or generate one; inject into context and response.

    Ensures context is cleared in finally block to prevent leaks between requests.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or generate_request_id()
        # Bind context and get token for restoration
        from src.rag_agent.utils.logging_config import REQUEST_ID_CTX

        token = REQUEST_ID_CTX.set(request_id)
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            # Clear context to prevent leaks
            REQUEST_ID_CTX.reset(token)
