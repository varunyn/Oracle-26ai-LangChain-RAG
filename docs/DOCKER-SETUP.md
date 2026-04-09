# Docker setup

This page describes how to run the backend, frontend, and optional stacks using Docker. The **Makefile** is the preferred entrypoint for all commands.

## Prerequisites

- Docker Desktop (or Docker Engine) + Docker Compose
- `.env` in the repo root (copy from `.env.example`)
- OCI config + wallet in `./local-config/` (used by the backend container)

## 1. Makefile (preferred)

From the repo root, use `make` to run core and optional stacks:

```bash
make help
```

**Core app (backend + frontend):**

```bash
make core-up          # Start backend + frontend
make core-logs        # Follow backend logs
make core-down       # Stop and remove containers
```

**Full run (core + stacks enabled in .env):**

Stacks are auto-selected from `.env` flags (`ENABLE_OBSERVABILITY_STACK`, `ENABLE_OTEL_TRACING`, `ENABLE_LANGFUSE_TRACING`). You do not need to set `DOCKER_STACKS`; see `api/settings.py` for defaults.

```bash
make up               # core-up, then start enabled stacks
make status           # docker compose ps + stack status
make down             # Stop stacks, then core-down
```

**Optional stacks (one at a time):**

```bash
make observability-up
make observability-status
make observability-down

make langfuse-up
make langfuse-status
make langfuse-down
```

**All enabled stacks (no core):**

```bash
make stacks-up
make stacks-status
make stacks-down
```

Ports:

- API: <http://localhost:3002>
- UI: <http://localhost:4000>
- Langfuse (if run): <http://localhost:3300>

## 2. Backend + frontend (without Makefile)

If you prefer to call Docker Compose directly:

```bash
docker compose up -d backend frontend
docker compose logs -f backend
docker compose down
```

## 3. Configure OCI + wallet for Docker

The backend container mounts `./local-config` and uses the wallet and OCI key from there:

- `./local-config/wallet` → mounted at `/app/wallet`
- `./local-config/oci_api_key.pem` → mounted as a Docker secret (`/run/secrets/oci_api_key`)
- `./local-config/oci/config` → referenced by `OCI_CONFIG_FILE`

See `docs/CONFIGURATION.md` for the required env vars.

## 4. Stack manager script (under the hood)

The Makefile delegates optional stacks to `scripts/manage_stacks.py`. You can call it directly for finer control:

- `ENABLE_OBSERVABILITY_STACK=true` **or** `ENABLE_OTEL_TRACING=true` → adds `observability`
- `ENABLE_LANGFUSE_TRACING=true` → adds `langfuse`

```bash
uv run python scripts/manage_stacks.py up --stacks observability
uv run python scripts/manage_stacks.py up --stacks langfuse
uv run python scripts/manage_stacks.py status
uv run python scripts/manage_stacks.py down
```

Stack names:

- `core` → backend + frontend (started by `make core-up` / `docker compose`; the script can manage it if configured)
- `observability` → Loki, Tempo, OTEL collector, Grafana
- `langfuse` → Langfuse web + worker (compose under `observability/langfuse/`)

The script uses `DOCKER_STACKS` from `.env` if set, otherwise defaults in `api/settings.py` (with stacks auto-enabled from the `ENABLE_*` flags above).

## 5. Observability stack (optional)

Start via Makefile (recommended):

```bash
make observability-up
```

Or via the script or compose profile:

```bash
uv run python scripts/manage_stacks.py up --stacks observability
# or
docker compose --profile observability up -d loki tempo otel-collector grafana
```

See `docs/OBSERVABILITY.md` for ports and verification.

## 6. Langfuse stack (optional)

Start via Makefile (recommended):

```bash
make langfuse-up
```

Or via the script or compose file:

```bash
uv run python scripts/manage_stacks.py up --stacks langfuse
# or
docker compose -f observability/langfuse/docker-compose.yml up -d
```

Langfuse UI: <http://localhost:3300>. For the backend (e.g. rag-backend) to send traces to Langfuse, ensure the Langfuse stack is on `rag-network` and `langfuse-web` was started with the network alias and `HOSTNAME=0.0.0.0` (see `observability/langfuse/docker-compose.yml`). Start the main stack first so `rag-network` exists, then run `make langfuse-up`.
