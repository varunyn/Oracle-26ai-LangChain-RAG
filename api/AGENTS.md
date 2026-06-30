# API AGENTS.md - Local Codex Hints

This file is local-only guidance for edits under `api/`. Root `AGENTS.md` owns repo-wide style, testing, security, Langfuse, and contribution rules.

## API Map

- `main.py`: FastAPI app, lifespan, middleware, exception handlers, router include.
- `routes/api.py`: router aggregation.
- `routes/`: HTTP handlers for health, config, suggestions, feedback, documents, and shared response shaping.
- `dependencies.py`: public route helper surface; imports request-scoped DI and owns chat config/logging helpers.
- `deps/request.py`: request-scoped DI from `app.state.resources` (`get_settings`).
- `resources.py`: app-scoped FastAPI product-API resources (`Settings`).
- `schemas.py`: shared Pydantic request/response models.

Runtime execution belongs outside the FastAPI layer:

- `src/rag_agent/graphs/`: active LangGraph Agent Server graph and mode-specific nodes.
- `src/rag_agent/runtime/agent_server_checkpointer.py`: local SQLite Agent Server checkpointer factory configured by `langgraph.json`.

## Boundaries

- Add public HTTP endpoints under `api/routes/*`; keep route modules focused on validation, response shaping, and delegation.
- Keep provider/workflow execution in `src/rag_agent/runtime/*` or lower-level `src/rag_agent/*` modules.
- Use app/request DI instead of constructing runtime services in handlers.
- Use `api/settings.py`/`get_settings()` for config; do not add parallel config modules.
- Keep `api` from importing FastAPI into `src/rag_agent/*`.

## Public Contracts

Keep these stable unless frontend/tests/docs are updated together:

- `/api/config`
- `/api/suggestions`
- `/api/feedback`
- `/api/documents/*`

Chat streaming/thread contracts live in the LangGraph Agent Server `chat_agent` surface rather than FastAPI. Keep FastAPI focused on product APIs (`/api/config`, `/api/suggestions`, `/api/feedback`, `/api/documents/*`) and verify frontend chat changes against port 2024.

## Useful Checks

- OpenAPI: `uv run pytest tests/workflow_tests/test_openapi_baseline.py -q`
- Graph/server contract: `uv run pytest tests/workflow_tests/test_langgraph_chat_contract.py tests/workflow_tests/test_langgraph_server_bootstrap.py -q`
- Non-streaming/validation: `uv run pytest tests/workflow_tests/test_chat_nonstream_and_validation.py -q`
- Request IDs: `uv run pytest tests/workflow_tests/test_request_id_propagation.py -q`
