# AGENTS.md

A **README for agents**: a dedicated, predictable place for context and instructions to help AI coding agents work on this project. Follow every instruction below; when in doubt, ask for clarification before editing.

- **Precedence**: The closest `AGENTS.md` to the file being edited wins. Explicit user chat prompts override everything.
- **Living doc**: Treat this file as living documentation; keep it updated when workflows, commands, or rules change. Agents should treat omissions as bugs and contribute improvements proactively.

## 1. Project overview & topology

| Area                                                  | Stack                                 | Entrypoints                                          |
| ----------------------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| `src`, `api`, `scripts`, `mcp_servers`, `tests`, `ui` | Python 3.11 (LangChain, FastAPI, MCP) | `uv run python ...`, `uv run pytest`, `./run_api.sh` |
| `frontend`                                            | Next.js 16 / React 19 / Tailwind CSS  | `PORT=4000 pnpm dev`, `pnpm build`, `pnpm lint`      |
| Docker stacks                                         | OTEL/Observability, Langfuse          | `uv run python scripts/manage_stacks.py up           |

## 2. Environment & Dependency Management

1. Install [uv](https://github.com/astral-sh/uv) locally. Sync deps with:
   ```bash
   uv sync              # prod deps
   uv sync --group dev  # +pytest, black, ruff, mypy
   ```
2. Python commands **must** run through uv to ensure the project-managed virtualenv is activated automatically (or call `.venv/bin/python ...`).
3. Frontend workflows use **pnpm** for Next.js work. Prefer:
   ```bash
   cd frontend
   pnpm install
   PORT=4000 pnpm dev | pnpm build | pnpm lint
   ```
4. Docker stacks are defined in api/settings.py (DOCKER_STACKS); override in .env if needed. Enable the stacks you need (e.g., `core`, `observability`, `langfuse`) and run `uv run python scripts/manage_stacks.py up --stacks core`. The script validates compose/env files and shell-outs to the correct `docker compose` incantation.

## 3. Service Runtimes

| Purpose                    | Command                                                            | Notes                                                                                                                                                              |
| -------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FastAPI API w/ MCP tooling | `./run_api.sh`                                                     | Ensures `.venv`; runs `uvicorn api.main:app --reload --port ${PORT:-3002}`. If `ENABLE_OBSERVABILITY_STACK=True`, auto-starts the OTEL docker profile before boot. |
| Next.js web UI             | `PORT=4000 pnpm dev`                                               | Local dev server lives on http://localhost:4000; Docker compose maps 4000→3000.                                                                                    |
| Observability stack        | `uv run python scripts/manage_stacks.py up --stacks observability` | Starts Grafana + Tempo + Loki + OTEL collector (uses compose profile).                                                                                             |
| Langfuse stack             | `uv run python scripts/manage_stacks.py up --stacks langfuse`      | Runs Langfuse web/worker on http://localhost:3300; requires `.env` in `observability/langfuse/`.                                                                   |

## 4. Build / Lint / Test Matrix

### Python

| Task             | Command                                                                                       | Extra Details                                                                                              |
| ---------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Lint (Ruff)      | `uv run ruff check`                                                                           | Config: `pyproject.toml` (`line-length=100`, `select=[E,F,I,N,W,UP]`, `ignore=[E501]`)                     |
| Format (Black)   | `uv run black .`                                                                              | `line-length=100`, target py311                                                                            |
| Type Check       | `uv run mypy src api tests scripts`                                                           | `warn_return_any=true`, `warn_unused_configs=true`, `disallow_untyped_defs=false`                          |
| Test Suite       | `uv run pytest`                                                                               | Pytest config lives in `pyproject.toml` (`testpaths=["tests/unit_tests", "tests/workflow_tests", "tests/integration_tests"]`, default markers, `addopts=--record-mode=once`) |
| Single Test      | `uv run pytest tests/unit_tests/test_file.py::test_case`, `uv run pytest tests/workflow_tests/test_file.py -k "keyword"`, or `uv run pytest tests/integration_tests -m integration` | Integration tests are marked `@pytest.mark.integration`; workflow tests stay deterministic and do not hit real providers by default |
| Manual MCP tools | `uv run python tests/run_mcp_semantic_search.py` etc.                                         | Use to validate MCP servers without pytest                                                                 |

### JavaScript / TypeScript

| Task              | Command         | Notes                                                        |
| ----------------- | --------------- | ------------------------------------------------------------ |
| Dev server        | `pnpm dev`      | Next.js app dir `frontend/`                                  |
| Production build  | `pnpm build`    | Executes `next build`; fails on lint/type issues             |
| Start prod server | `pnpm start`    | Runs `next start` after build                                |
| ESLint            | `pnpm lint`     | Uses `eslint.config.mjs` (Next core-web-vitals + TypeScript) |
| E2E tests         | `pnpm test:e2e` | Playwright is configured in `frontend/playwright.config.ts`  |

## 5. Baseline Config Files

- `pyproject.toml`: authoritative Python metadata, dependencies, Ruff/Black/Mypy/Pytest settings.
- `uv.lock`: lockfile generated by uv; update via `uv lock`.
- `frontend/eslint.config.mjs`: extends Next core-web-vitals + TS, custom ignores.
- `frontend/tsconfig.json`: `strict=true`, `noEmit`, bundler resolution, `paths: { "@/*": "./src/*" }`.
- `.env.example`: copy to `.env` (gitignored) and set OCI + Oracle vectors + MCP settings; see docs/CONFIGURATION.md.
- `docs/OBSERVABILITY_ROUTING.md`: explains how to route traces/logs to Grafana/Tempo, OCI APM, and Logging Analytics simultaneously.

## 7. Code Style — Python

1. **Imports**: Standard library → third-party → local (`from __future__ import annotations` when needed). Group blocks with blank line separation and keep alphabetical order (Ruff `I` rules enforce this).
2. **Typing**:
   - Use type hints everywhere (functions, class attrs, TypedDict fields).
   - Use TypedDict, Literal, and pydantic.BaseModel for structured schemas. Use Annotated where metadata-driven validation or framework integration benefits from it.
   - Use `collections.abc` types (`Sequence`, `Mapping`) instead of `typing.List` in new code.
3. **Functions & Classes**: snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants. Keep functions focused on one responsibility. Extract helper functions or services when branching, state mutation, or error handling starts to obscure the main flow; push complex logic into helper services (see `src/rag_agent/*.py`).
4. **Error Handling**: Catch the most specific exception possible, log via module logger, and either rethrow with context (`raise CustomError(...) from exc`) or return typed error payloads (e.g., `search_error_response`). Never swallow exceptions silently.
5. **Logging**: Always create module loggers via `logging.getLogger(__name__)`. Prefer logs that describe the operation, outcome, and identifiers; avoid logging secrets, full tokens, or large raw payloads.
6. **State & Data Flow**: Chat runtime state is managed in `api/services/graph_service.py` and request config in `api/dependencies.py`. Keep response contracts stable (`final_answer`, `citations`, `reranker_docs`, `context_usage`, `mcp_*`) and centralize citation normalization in `src/rag_agent/core/citations.py`.
7. **Security**: Never check `.env` into git; use `.env.example` as reference. Use OCI wallet paths from settings/env, not hardcoded strings.
8. **Testing**: Tests live under `tests/` with categorized suites: `tests/unit_tests` for deterministic unit tests, `tests/workflow_tests` for deterministic orchestration tests with mocked boundaries, and `tests/integration_tests` for real external/provider/backend tests. Respect `pytest` markers (`integration`, `vcr`, `langsmith`). Use VCR for HTTP recording (`--record-mode=once`). Detailed LangChain testing guidance lives in `tests/AGENTS.md`.

## 8. Code Style — TypeScript / Next.js

1. **Project Layout**: Next.js App Router inside `frontend/app` (inspect actual layout before editing). Components belong in `frontend/src/components` or similar; use `@/` alias.
2. **Imports**: Third-party modules first, absolute `@/` paths second, relative paths last. Keep CSS/tailwind imports at the top-level layout files only.
3. **Components**: Prefer function components with explicit prop types. Use `React.FC` only when generics are required; otherwise annotate props inline (`type ButtonProps = React.ComponentProps<"button"> & { variant?: "ghost" }`).
4. **Hooks**: Keep `use` hooks at file top-level; guard SSR-only APIs inside `useEffect`. Memoize expensive computations with `useMemo` and event handlers with `useCallback` when passing down.
5. **Styling**: Tailwind v4 + shadcn primitives. Compose classes with `clsx` or `tailwind-merge`. For math rendering use `katex` + `@streamdown/math` components.
6. **Type Safety**: Strict TS mode; `any` is disallowed. When dealing with streaming AI payloads, define discriminated unions rather than generic `Record<string, unknown>`.
7. **Data Fetching**: Favor Next.js server actions / route handlers for backend calls. Client components interacting with RAG API must read base URLs from env (e.g., `process.env.NEXT_PUBLIC_API_BASE`).
8. **Testing**: Playwright is configured for frontend e2e coverage. Before adding a new JS/TS unit-test harness, document the choice and config in the repo.

## 9. Error Handling & Observability

1. **FastAPI**: Wrap router handlers with structured errors; return `JSONResponse` with `status_code` set explicitly. Use Pydantic response models from `api/schemas.py` to guarantee shape.
2. **Runtime Steps**: Each runtime path (`rag`, `mcp`, `mixed`, `direct`) should validate prerequisites and write meaningful `error` fields so API responses and streams surface failures clearly. Keep operations idempotent where possible.
3. **Core boundary**: Prefer importing observability/config/logging wrappers from `src.rag_agent.core.*`.
   - Compatibility note: `src.rag_agent.utils.*` remains in place as intentional shims during migration.
4. **Logging IDs**: Request ID header `X-Request-ID` is injected by middleware. Ensure downstream logs include this context.

## 10. Security Expectations

1. Secrets (OCI, DB wallets, JWT keys) live in `.env` or env vars; never commit secrets.
2. Enforce Oracle DB credentials via wallet files referenced in config, not code.

## 11. Contribution Workflow

1. **Branching**: Create feature branches; main should stay deployable.
2. **Commits**: Use conventional, descriptive subjects. Never add `Co-authored-by` (per Cursor rule).
3. **Testing Gate**: Before opening PRs, run `uv run ruff check`, `uv run black --check .`, `uv run mypy src api tests scripts`, `uv run pytest`, `uv run python scripts/sync_api_docs.py --check`, `./scripts/regression_guard.sh`, `cd frontend && pnpm lint`, and `cd frontend && pnpm build`. Attach logs for failures or document flaky tests.
4. **PR Content**: Note pnpm requirement for frontend, list env vars touched, and describe any MCP config changes.

## 12. Release & Publication Workflow

1. The local publication gate script is `./scripts/release_checklist.sh`.

## 13. Reference Snippets

```bash
# Single pytest by node id
uv run pytest tests/unit_tests/test_mcp_agent_executor.py::test_build_middleware_skips_llm_selector_for_oci_models

# Workflow-focused pytest
uv run pytest tests/workflow_tests/test_ai_sdk_stream.py -k "references"

# Integration-only pytest
uv run pytest tests/integration_tests -m integration

# Ruff autofix import order only
uv run ruff check --select I --fix

# Frontend type check (Next.js build performs this)
pnpm build
```
