"""Shared API bootstrap helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_project_root_to_sys_path(api_file: str) -> Path:
    """Ensure project root is importable and return it."""
    api_dir = Path(api_file).resolve().parent
    project_root = api_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def configure_default_oci_config(project_root: Path) -> None:
    """Default OCI_CONFIG_FILE to local-config/oci/config when present."""
    oci_config = project_root / "local-config" / "oci" / "config"
    if not os.environ.get("OCI_CONFIG_FILE") and oci_config.is_file():
        os.environ["OCI_CONFIG_FILE"] = str(oci_config)

