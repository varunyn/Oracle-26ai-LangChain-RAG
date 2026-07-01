# Langfuse Local Stack

Self-hosted Langfuse (web + worker + ClickHouse + Postgres + Redis + MinIO) that you can run independently of the main `docker-compose.yml`. Keeping it in `observability/langfuse/` avoids polluting the primary compose file while still making it easy to opt-in when you want Langfuse SDK traces.

## Quick start

```bash
cd /path/to/custom-rag-agent-app
cp observability/langfuse/.env.example observability/langfuse/.env
# edit observability/langfuse/.env and change every `changeme-*` value
docker compose -f observability/langfuse/docker-compose.yml up -d
```

The compose file binds the Langfuse UI to `http://localhost:3300` (default) and exposes MinIO on `http://localhost:9090` (S3 API) and `http://localhost:9091` (console). All internal databases stay on `127.0.0.1` so the stack is only reachable from your machine.

ClickHouse is capped to 1 CPU and 2 GB of memory by default for local laptop use. It also mounts `clickhouse-config.d/low-resource.xml`, which turns down ClickHouse's own internal logs so background system-log merges do not dominate CPU or disk. Adjust the caps in `observability/langfuse/.env` if you import a large trace history:

```bash
CLICKHOUSE_CPUS=1.0
CLICKHOUSE_MEM_LIMIT=2g
```

Docker `json-file` log rotation is also enabled to prevent local Docker/Colima disks from filling up. ClickHouse keeps three 10 MB files; Langfuse web/worker, Redis, and MinIO keep two 5 MB files. Adjust these in `observability/langfuse/.env`:

```bash
CLICKHOUSE_DOCKER_LOG_MAX_SIZE=10m
CLICKHOUSE_DOCKER_LOG_MAX_FILE=3
LANGFUSE_DOCKER_LOG_MAX_SIZE=5m
LANGFUSE_DOCKER_LOG_MAX_FILE=2
```

The local MinIO bucket is configured with lifecycle rules so raw ingestion blobs do not accumulate indefinitely. By default, raw event uploads expire after one day, while media and export objects expire after seven days:

```bash
MINIO_EVENT_RETENTION_DAYS=1
MINIO_MEDIA_RETENTION_DAYS=7
MINIO_EXPORT_RETENTION_DAYS=7
```

Keep SDK trace payloads small for local development. The application defaults in `.env.example` sample only a fraction of traces and truncate very large exported OTEL attributes before they reach Langfuse:

```bash
LANGFUSE_SAMPLE_RATE=0.25
```

Set `LANGFUSE_SAMPLE_RATE=1.0` only when you need every run.

To stop everything:

```bash
docker compose -f observability/langfuse/docker-compose.yml down
```

Volumes (`langfuse_*`) keep your data between restarts. Add `-v` to `down` if you want to delete everything.

## Configuration checklist

- **Secrets** – generate strong values before running:
  - `SALT`, `NEXTAUTH_SECRET`, `ENCRYPTION_KEY` (`openssl rand -hex 32`)
  - `POSTGRES_PASSWORD`, `CLICKHOUSE_PASSWORD`, `REDIS_AUTH`, `MINIO_ROOT_PASSWORD`
  - MinIO-related access keys (`LANGFUSE_S3_*_SECRET_ACCESS_KEY`)
- **Langfuse bootstrap (optional)** – set `LANGFUSE_INIT_*` in `.env` to create the first org/user/project on startup. **You must set `LANGFUSE_INIT_ORG_ID`** (e.g. `my-org`); otherwise `LANGFUSE_INIT_USER_EMAIL`, `LANGFUSE_INIT_USER_NAME`, and `LANGFUSE_INIT_USER_PASSWORD` are ignored. See `.env.example` for the full list.
- **Backend integration** – after the stack is running, set `ENABLE_LANGFUSE_TRACING=true` and `LANGFUSE_HOST=http://localhost:3300` in `.env` so the FastAPI app sends traces to Langfuse.

See [`docs/TRACING.md`](../../docs/TRACING.md#6-langfuse-sdk-optional) for the SDK wiring details.
