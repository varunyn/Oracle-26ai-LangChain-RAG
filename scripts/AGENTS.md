# Scripts — AGENTS.md (scoped to scripts/)

OVERVIEW
- One-off utilities for development, testing, and operations. Applies ONLY to scripts/.

KEY SCRIPTS
- regression_guard.sh — runs lint/type/tests + streaming smoke for quick gate
- release_checklist.sh — verifies release docs, generated API docs, and repository checks
- streaming_smoke_test.sh — Agent Server SSE smoke test against the `chat_agent` graph on port 2024
- export_openapi.py — exports current OpenAPI for comparison/diff
- manage_stacks.py — orchestrates docker compose profiles (observability/langfuse)
- ingest_documents.py (primary CLI wrapper over `src/rag_agent/ingestion.py`) / create_rag_table.py / truncate_table.py / drop_table.py — DB utilities
- bm25_search.py — local BM25 helper for experimentation
- verify_oci_logging_analytics.py — validate OCI logging analytics routing

RUNNING
- Make executable if needed: chmod +x scripts/*.sh
- Execute: ./scripts/regression_guard.sh
- Python scripts must run via uv to pick the project venv: uv run python scripts/<name>.py [...]
- Frontend/docs-site commands belong to their own directories; do not add Node workflows to Python scripts.

CONVENTIONS
- Keep scripts idempotent; print clear status and non-zero exit on failure
- Avoid secrets in CLI flags; prefer env vars or settings files
- Document usage at top of the script (help/usage comment) when non-trivial

ANTI-PATTERNS (FORBIDDEN)
- Committing logs, temporary outputs, or secrets
- Running docker compose directly for observability; prefer manage_stacks.py
- Silent failures; always exit non-zero on error

NOTES
- See root AGENTS.md for full command matrix and testing gates
- Add tests (where reasonable) before wiring new scripts into regression_guard.sh
