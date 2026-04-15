from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ingest_documents_cli_exposes_main_and_process_file_paths() -> None:
    module = load_module(
        "ingest_documents_cli",
        Path("scripts/ingest_documents.py"),
    )

    assert callable(module.main)
    assert callable(module.process_file_paths)
