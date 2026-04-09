#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Public Release Checklist ===${NC}"

echo "1. Running backend release gates..."
uv run ruff check
uv run black --check .
uv run mypy src api tests scripts
uv run pytest
uv run python scripts/export_openapi.py /tmp/current_openapi.json
uv run python scripts/sync_api_docs.py --check
./scripts/regression_guard.sh

echo "2. Running frontend release gates..."
(
  cd frontend
  pnpm lint
  pnpm build
)

echo "3. Confirming trust-signal files exist..."
test -f LICENSE

echo -e "${GREEN}✅ Public release checklist passed${NC}"
