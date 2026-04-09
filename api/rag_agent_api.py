"""
RAG Agent API – backward-compatible re-export.

The app and routes live in api.main and api.routes. This module re-exports the app
and commonly used symbols so existing references (e.g. uvicorn api.rag_agent_api:app,
tests importing ChatMessage, run_rag_and_get_answer) keep working.
"""

# Ensure project root is on sys.path (same bootstrap as main).
import os
import sys
from pathlib import Path

_API_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _API_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Single OCI config: local-config/oci/config (same as config.OCI_CONFIG_FILE)
_oci_config = _PROJECT_ROOT / "local-config" / "oci" / "config"
if not os.environ.get("OCI_CONFIG_FILE") and _oci_config.is_file():
    os.environ["OCI_CONFIG_FILE"] = str(_oci_config)

from api.main import app
from api.routes.chat import run_rag_and_get_answer
from api.schemas import ChatMessage

__all__ = ["app", "ChatMessage", "run_rag_and_get_answer"]
