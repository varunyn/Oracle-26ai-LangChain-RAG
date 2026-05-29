# AGENTS.md - Local Codex Hints

This file is local-only orientation for Codex work in this repo. The closest `AGENTS.md` to the file being edited wins. Explicit chat instructions override these notes.

## Repo Map

| Area | Stack | Primary Commands |
| --- | --- | --- |
| `src`, `api`, `scripts`, `mcp_servers`, `tests`, `ui` | Python 3.11, LangChain, FastAPI, MCP | `uv run python ...`, `uv run pytest`, `./run_api.sh` |
| `frontend` | Next.js 16, React 19, Tailwind CSS | `PORT=4000 pnpm dev`, `pnpm build`, `pnpm lint` |
| Docker stacks | Core app, observability, Langfuse | `uv run python scripts/manage_stacks.py up --stacks <name>` |

Scoped hints:

- `api/AGENTS.md`: FastAPI route/runtime boundaries and API contracts.
- `frontend/AGENTS.md`: Next.js layout, frontend commands, and UI/backend contracts.
- `src/AGENTS.md`: LangChain documentation rule for runtime/source work.
- `tests/AGENTS.md`: LangChain testing categories and fake-model guidance.
- `scripts/AGENTS.md`, `mcp_servers/AGENTS.md`, `observability/AGENTS.md`: scoped operational notes.

## Environment

- Use `uv` for Python commands so the project-managed virtualenv is active.
- Sync Python dependencies with `uv sync` or `uv sync --group dev`.
- Use `pnpm` for frontend workflows. Do not use npm/yarn for Next.js work.
- Keep secrets in `.env` or environment variables. Never commit `.env`, wallet files, tokens, or credentials.
- Root `.env.example` documents backend/runtime settings. `frontend/env.example` documents frontend `.env.local`.

## Runtime Commands

- FastAPI API: `./run_api.sh`
- Direct API fallback: `uv run uvicorn api.main:app --host 127.0.0.1 --port 3002 --reload`
- API health: `curl -s http://127.0.0.1:3002/health`
- Frontend dev: `cd frontend && PORT=4000 pnpm dev`
- Check running Compose services first: `docker compose ps`
- Inspect Compose logs before guessing: `docker compose logs --tail 120 <service>`
- Observability stack: `uv run python scripts/manage_stacks.py up --stacks observability`
- Langfuse stack: `uv run python scripts/manage_stacks.py up --stacks langfuse`

## Checks

Use targeted checks while iterating, then broaden based on risk.

Python:

- Lint: `uv run ruff check`
- Format check: `uv run black --check .`
- Type check: `uv run mypy src api tests scripts`
- Full tests: `uv run pytest`
- API docs drift: `uv run python scripts/sync_api_docs.py --check`
- Regression guard: `./scripts/regression_guard.sh`

Frontend:

- Lint: `cd frontend && pnpm lint`
- Build/type check: `cd frontend && pnpm build`
- E2E: `cd frontend && pnpm test:e2e`

Test categories:

- `tests/unit_tests`: fast deterministic units, no real providers or network.
- `tests/workflow_tests`: deterministic orchestration with mocked/fake external boundaries.
- `tests/integration_tests`: live provider/backend/runtime tests, explicitly gated.
- `tests/run_*.py`: manual scripts, not pytest suites.

## Repo-Wide Rules

- Read the closest scoped `AGENTS.md` before editing in a subdirectory.
- Keep response contracts stable unless frontend, tests, and docs move together.
- Runtime response fields such as `final_answer`, `citations`, `reranker_docs`, `context_usage`, and `mcp_*` are public-facing within this app.
- Citation normalization belongs in `src/rag_agent/core/citations.py`.
- Chat runtime state is owned by `src/rag_agent/runtime/chat_service.py`; request config helpers live in `api/dependencies.py`.
- For agentic workflow control, avoid regex or hardcoded prompt-keyword routing. Prefer model/tool-calling semantics, structured state, and deterministic validation of already-structured data.
- For LangChain-related source or test changes, check the current official docs through the configured LangChain docs MCP before recommending APIs or changing patterns.
- For agentic AI, tool-calling, MCP, routing, or LLM behavior debugging, inspect live traces with the installed Langfuse CLI before guessing from code. Prefer `langfuse --env .env api traces list --limit 5 --json`, `langfuse --env .env api traces get <trace-id> --fields core,io,scores,observations,metrics --json`, and `langfuse --env .env api observations list --filter '[{"type":"string","column":"traceId","operator":"=","value":"<trace-id>"}]' --fields core,basic,io,model,usage,metadata --json`. Never print public/secret key values.
- For runtime bugs, prefer live evidence first: check whether the Docker Compose app is already running, use the existing containers/surface when available, inspect Docker logs before guessing, then use browser/devtools output, actual tool schemas, request IDs, and repro commands.

## Contribution Notes

- Main should stay deployable; use focused branches and commits for shared work.
- Use conventional, descriptive commit subjects. Do not add `Co-authored-by`.
- Before opening PRs, run the relevant checks above and report any failures with concrete logs.
- For releases, use `./scripts/release_checklist.sh`.
