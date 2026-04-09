"""
RAG Agent API – main FastAPI app.

Expose the RAG agent as a REST API. Uses OpenAI-compatible Chat Completions format.
Entrypoint for uvicorn: api.main:app
"""

import os
import sys
from pathlib import Path

_API_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _API_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_oci_config = _PROJECT_ROOT / "local-config" / "oci" / "config"
if not os.environ.get("OCI_CONFIG_FILE") and _oci_config.is_file():
    os.environ["OCI_CONFIG_FILE"] = str(_oci_config)

# Set up OTel + LangSmith OTEL before importing routers (so LangGraph/LangChain are traced)
try:
    from src.rag_agent.core.otel import setup_otel_tracing_early

    _ = setup_otel_tracing_early()
except Exception:  # noqa: BLE001
    pass

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

from api.errors import (
    generic_exception_handler,
    http_exception_handler,
    request_validation_exception_handler,
)
from api.middleware.request_context import RequestIdMiddleware
from api.resources import create_app_resources, shutdown_app_resources
from api.routes import api
from src.rag_agent.core.logging import setup_logging
from src.rag_agent.infrastructure.db_utils import close_pool, get_pooled_connection


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
        from src.rag_agent.core.otel import setup_otel_tracing

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

    # Initialize app-scoped resources (async checkpointer + graph)
    try:
        app.state.resources = await create_app_resources()
        # Wire legacy dependency singleton to lifespan resource for compatibility
        try:
            import api.dependencies as _deps

            func = getattr(_deps, "set_graph_service_singleton", None)
            if callable(func):
                _ = func(app.state.resources.graph_service)
            else:
                _log.debug("No public setter for graph service singleton; skip wiring")
        except Exception as _e:  # noqa: BLE001
            _log.debug("Could not wire legacy dependency singleton: %s", _e)
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
app.add_middleware(RequestIdMiddleware)

# Centralized exception handlers
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(api.router)
