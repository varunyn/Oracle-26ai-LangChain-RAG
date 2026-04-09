import logging
from collections.abc import Sequence
from typing import cast

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _sanitize_validation_errors(errors: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Sanitize Pydantic validation errors to be JSON serializable."""
    sanitized: list[dict[str, object]] = []
    for error in errors:
        sanitized_error: dict[str, object] = dict(error)  # copy
        if "ctx" in sanitized_error:
            ctx = sanitized_error["ctx"]
            if isinstance(ctx, dict):
                ctx_dict = cast(dict[str, object], ctx)
                sanitized_ctx: dict[str, object] = {}
                for k, v in ctx_dict.items():
                    if isinstance(v, Exception):
                        sanitized_ctx[k] = str(v)
                    else:
                        sanitized_ctx[k] = v
                sanitized_error["ctx"] = sanitized_ctx
        sanitized.append(sanitized_error)
    return sanitized


def request_validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 422 validation errors with shaped JSON.

    Returns at least `detail` and `errors` keys. Does not leak internals beyond
    Pydantic's structured error objects. Logs with request context.
    """
    if isinstance(exc, RequestValidationError):
        # Log at info level to avoid noisy stack traces for expected 4xx
        logger.info("422 RequestValidationError at %s: %s", request.url.path, exc.errors())
        payload = {
            "detail": "Validation error",
            "errors": _sanitize_validation_errors(exc.errors()),
        }
        return JSONResponse(status_code=422, content=payload)

    # Fallback (should not occur via registration), but stay safe
    logger.warning("Validation handler received non-validation exception at %s", request.url.path)
    return JSONResponse(status_code=422, content={"detail": "Validation error", "errors": []})


def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle FastAPI/Starlette HTTPException with consistent envelope.

    Preserves status code and the `detail` message per requirement.
    """
    status = 500
    detail: object = "Internal Server Error"
    if isinstance(exc, HTTPException):
        status = exc.status_code
        detail_obj = exc.detail
        # Preserve dict/list details; stringify other types
        detail = detail_obj if isinstance(detail_obj, (dict, list)) else str(detail_obj)

        # Log at warning for 4xx and error for 5xx
        if 400 <= status < 500:
            logger.warning("HTTPException %s at %s: %s", status, request.url.path, detail)
        else:
            logger.error("HTTPException %s at %s: %s", status, request.url.path, detail)
    else:
        logger.error(
            "Non-HTTP exception passed to http_exception_handler at %s",
            request.url.path,
            exc_info=exc,
        )

    payload = {"detail": detail}
    return JSONResponse(status_code=status, content=payload)


def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions without leaking raw exception strings.

    Logs full stack trace server-side. Returns a generic error message to clients.
    Streaming routes should avoid propagating raw exceptions after headers are sent
    – this handler is for non-stream JSON responses.
    """
    logger.error("Unhandled exception at %s", request.url.path, exc_info=exc)
    payload = {"detail": "An internal error occurred. Please try again later."}
    return JSONResponse(status_code=500, content=payload)
