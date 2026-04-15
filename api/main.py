"""
RAG Agent API – main FastAPI app.

Expose the RAG runtime and supporting API endpoints.
Entrypoint for uvicorn: api.main:app
"""

import sys
from pathlib import Path

from api.bootstrap import add_project_root_to_sys_path, configure_default_oci_config

_PROJECT_ROOT = add_project_root_to_sys_path(__file__)
configure_default_oci_config(_PROJECT_ROOT)

# Set up OTel + LangSmith OTEL before importing routers (so LangChain runtime spans are traced)
try:
    from src.rag_agent.utils.otel_tracing import setup_otel_tracing_early

    _ = setup_otel_tracing_early()
except Exception:  # noqa: BLE001
    pass

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from api.errors import (
    generic_exception_handler,
    http_exception_handler,
    request_validation_exception_handler,
)
from api.middleware.request_context import RequestIdMiddleware
from api.resources import create_app_resources, shutdown_app_resources
from api.routes import api
from api.settings import get_settings
from src.rag_agent.infrastructure.db_utils import close_pool, get_pooled_connection
from src.rag_agent.utils.logging_config import setup_logging


def _warm_pool_sync() -> None:
    """Run in thread: warm Oracle pool so first request doesn't pay connection cost."""
    try:
        with get_pooled_connection() as _conn:
            pass
    except Exception as e:
        logging.getLogger(__name__).debug("Pool warm-up failed (non-fatal): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging; warm Oracle pool in background. Close pool on shutdown."""
    setup_logging()
    _log = logging.getLogger(__name__)
    _log.info("Process Python: %s", sys.executable)

    # Initialize OpenTelemetry tracing if enabled (fail-open)
    try:
        from src.rag_agent.utils.otel_tracing import setup_otel_tracing

        _ = setup_otel_tracing(app)
    except Exception as _e:  # noqa: BLE001
        _log.warning("OTel tracing setup failed (non-fatal): %s", _e)

    _this_venv = _PROJECT_ROOT / ".venv"
    _venv_str = _this_venv.resolve().as_posix()
    _exe_raw = sys.executable
    _exe_resolved = Path(sys.executable).resolve().as_posix()
    _using_this_venv = _venv_str in _exe_raw or _venv_str in _exe_resolved
    if not _using_this_venv:
        _log.warning(
            "Process Python is not from this app's .venv (expected under %s). MCP will fail. Use: ./run_api.sh or ./.venv/bin/python -m uvicorn api.main:app --reload --port 3002",
            _this_venv,
        )
    _warm_thread = threading.Thread(target=_warm_pool_sync, daemon=True)
    _warm_thread.start()

    # Initialize app-scoped resources (chat runtime + settings)
    try:
        app.state.resources = await create_app_resources()
    except Exception as e:  # noqa: BLE001
        _log.warning("App resources init failed (non-fatal): %s", e)

    yield

    # Shutdown app-scoped resources then DB pool
    try:
        await shutdown_app_resources(getattr(app.state, "resources", None))
    except Exception as e:  # noqa: BLE001
        _log.warning("App resources shutdown failed (non-fatal): %s", e)
    close_pool(force=True)


app = FastAPI(lifespan=lifespan)
_settings = get_settings()
app.add_middleware(RequestIdMiddleware)
if _settings.ENABLE_CORS:
    # Keep CORS outermost so browser clients receive CORS headers even when
    # downstream middleware or handlers return error responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.CORS_ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Centralized exception handlers
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(api.router)
