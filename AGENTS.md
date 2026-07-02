# AGENTS.md - Local Codex Hints

This file is local-only orientation for Codex work in this repo. The closest `AGENTS.md` to the file being edited wins. Explicit chat instructions override these notes.

## Repo Map

| Area                            | Purpose                                                                | Primary commands                        |
| ------------------------------- | ---------------------------------------------------------------------- | --------------------------------------- |
| `src/rag_agent/graphs/`         | LangGraph `chat_agent` and direct/RAG/MCP/mixed nodes                  | `uv run langgraph dev`                  |
| `src/rag_agent/runtime/`        | Runtime helpers, memory/message normalization, streaming, retrieval    | `uv run pytest tests/unit_tests`        |
| `src/rag_agent/infrastructure/` | Oracle/OCI, MCP adapters/config, retrieval, model integrations         | `uv run python ...`                     |
| `api/`                          | FastAPI product APIs mounted into the LangGraph Agent Server           | `uv run langgraph dev`                  |
| `frontend/`                     | Next.js 16/React 19 chat and Settings UI                               | `cd frontend && pnpm dev`               |
| `mcp_servers/`                  | Standalone FastMCP semantic-search and RAG servers                     | `uv run python mcp_servers/<server>.py` |
| `docs/` and `docs-site/`        | Markdown source and Astro/ReallySimpleDocs site                        | `cd docs-site && npm run dev`           |
| `observability/`                | Grafana, Tempo, Loki, OTEL, and Langfuse stack configuration           | `make observability-up`                 |
| `scripts/`                      | Ingestion, database, stack, API-doc, smoke, and release utilities      | `uv run python scripts/<name>.py`       |

Scoped hints:

- `api/AGENTS.md`: FastAPI route/runtime boundaries and API contracts.
- `frontend/AGENTS.md`: Next.js layout, frontend commands, and UI/backend contracts.
- `src/AGENTS.md`: LangChain documentation rule and runtime/graph ownership.
- `tests/AGENTS.md`: LangChain testing categories and fake-model guidance.
- `scripts/AGENTS.md`, `mcp_servers/AGENTS.md`, `observability/AGENTS.md`, `docs-site/AGENTS.md`: scoped operational notes.

## Environment

- Use `uv` for Python commands so the project-managed virtualenv is active.
- Sync Python dependencies with `uv sync` or `uv sync --group dev`.
- Use `pnpm` for frontend workflows. Do not use npm/yarn for Next.js work.
- Keep secrets in `.env` or environment variables. Never commit `.env`, wallet files, tokens, or credentials.
- Root `.env.example` documents backend/runtime settings. `frontend/env.example` documents frontend `.env.local`.

## Runtime Commands

- LangGraph Agent Server and product APIs: `uv run langgraph dev`
- Product API health: `curl -s http://127.0.0.1:2024/health`
- Frontend dev: `cd frontend && PORT=4000 pnpm dev`
- Check running Compose services first: `docker compose ps`
- Inspect Compose logs before guessing: `docker compose logs --tail 120 <service>`
- Observability stack: `uv run python scripts/manage_stacks.py up --stacks observability`
- Langfuse stack: `uv run python scripts/manage_stacks.py up --stacks langfuse`
- Core stack: `make core-up`; stop with `make core-down`; inspect with `make status`

## Runtime Debugging

Start from the running system:

1. Check whether Docker Compose services are already running.
2. Inspect relevant logs before changing code.
3. For browser-visible regressions, collect:
   - console errors
   - network response
   - request ID
   - backend log lines
   - response or stream payload
4. When Docker is the user’s test surface, treat Docker as the source of truth.
5. Rebuild or hot-load frontend changes before trusting browser results.

## Runtime Boundaries

- The browser chat uses the LangGraph Agent Server at `http://localhost:2024` and graph id `chat_agent` from `langgraph.json`.
- `api/` owns product endpoints, not the primary chat stream. Keep chat orchestration in `src/rag_agent/graphs/` and runtime helpers in `src/rag_agent/runtime/`.
- MCP server definitions are managed through the Settings UI/config store and resolved by the MCP infrastructure layer. Do not hardcode consumed server URLs in graph or standalone server code.
- LangGraph Agent Server persistence owns chat thread history. Preserve structured `AIMessage` content and stable message ids across stream and replay changes.

## Agentic AI Changes

- For agentic AI, MCP, tool-calling, routing, retry, memory/state, streaming, or output-quality bugs, inspect live Langfuse traces before coding. Use `langfuse --env .env api traces list --limit 5 --json`, then fetch the trace with embedded observations using `langfuse --env .env api traces get <trace-id> --fields core,io,scores,observations,metrics --json`.
- Do not hardcode prompt-keyword or tool-name scenarios to make one example pass. Prefer model tool-calling semantics, tool descriptions, structured state, middleware, and deterministic checks on already-structured tool results.
- For provider errors, inspect the real package behavior and live payload shape first. Keep compatibility fixes at the app/provider boundary and avoid patching vendored packages unless explicitly requested.
- Unit tests are not enough for LangChain/LangGraph changes. Run a workflow test plus an Agent Server or frontend path that exercises the actual graph/agent flow; use real model calls and real tool calls when the environment allows.
- If real calls are not possible, say exactly what was validated instead and what remains unproven. Keep deterministic unit/workflow tests for guardrails, but do not present them as proof of live agent behavior.
- Streaming fixes must be verified at the stream contract and UI layers: Agent Server SSE/events, frontend stream state, visible tool progress, thread replay, and final answer/citation rendering.

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

- Lint (ESLint): `cd frontend && pnpm lint`
- Check (Ultracite/Biome): `cd frontend && pnpm check`
- Build/type check: `cd frontend && pnpm build`
- Dead code/unused deps: `cd frontend && pnpm knip`
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
- Chat graph state and thread persistence are owned by the LangGraph Agent Server graph/runtime; request config helpers live in `api/dependencies.py`.
- For agentic workflow control, avoid regex or hardcoded prompt-keyword routing. Prefer model/tool-calling semantics, structured state, and deterministic validation of already-structured data.
- For LangChain-related source or test changes, check the current official docs through the configured LangChain docs MCP before recommending APIs or changing patterns.
- For agentic AI, tool-calling, MCP, routing, or LLM behavior debugging, inspect live traces with the installed Langfuse CLI before guessing from code. Prefer `langfuse --env .env api traces list --limit 5 --json` followed by `langfuse --env .env api traces get <trace-id> --fields core,io,scores,observations,metrics --json`. The local self-hosted stack is Langfuse v3; do not use `api observations list` here because that endpoint requires Langfuse v4 write mode. Never print public/secret key values.
- For runtime bugs, prefer live evidence first: check whether the Docker Compose app is already running, use the existing containers/surface when available, inspect Docker logs before guessing, then use browser/devtools output, actual tool schemas, request IDs, and repro commands.
- For frontend chat regressions, check both `frontend/src/hooks/chat/` projection/state code and the Agent Server payload before changing rendering components.
- Significant features, fixes, refactorings, specification updates, deployment changes, and documentation updates must be recorded in CHANGELOG.md under the current date.

## Contribution Notes

- Main should stay deployable; use focused branches and commits for shared work.
- Use conventional, descriptive commit subjects. Do not add `Co-authored-by`.
- Before opening PRs, run the relevant checks above and report any failures with concrete logs.
- For releases, use `./scripts/release_checklist.sh`.

## AGENTS.md Maintenance

- Keep this root file limited to repo-wide rules and architectural landmarks; put subsystem-specific rules in the nearest scoped file.
- Prefer references to stable directories, entry points, and commands over exhaustive file inventories.
- When a command, ownership boundary, or generated-file rule changes, update the nearest affected guide in the same change.
- Keep scoped guides tracked and review them like code; do not hide project guidance with local ignore rules.

Don't leave fallbacks and legacy code. Implementation must be thorough
