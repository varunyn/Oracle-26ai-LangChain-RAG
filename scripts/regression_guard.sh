#!/bin/bash
set -euo pipefail

# Establish regression guard + OpenAPI snapshot baseline
# This script exports OpenAPI, diffs against baseline, runs tests, and performs streaming smoke test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Regression Guard ===${NC}"

# Step 1: Export OpenAPI JSON
echo "1. Exporting OpenAPI JSON..."
if ! uv run python scripts/export_openapi.py /tmp/current_openapi.json; then
    echo -e "${RED}❌ Failed to export OpenAPI${NC}"
    exit 1
fi
echo -e "${GREEN}✅ OpenAPI exported${NC}"

# Step 2: Diff against baseline
echo "2. Diffing against baseline..."
BASELINE="tests/fixtures/openapi-baseline.json"
if [ ! -f "$BASELINE" ]; then
    echo -e "${RED}❌ Baseline file not found: $BASELINE${NC}"
    echo -e "${YELLOW}To create baseline, run: uv run python scripts/export_openapi.py $BASELINE${NC}"
    exit 1
fi

if ! diff -u "$BASELINE" /tmp/current_openapi.json; then
    echo -e "${RED}❌ OpenAPI spec differs from baseline${NC}"
    echo -e "${YELLOW}To update baseline: cp /tmp/current_openapi.json $BASELINE${NC}"
    exit 1
fi
echo -e "${GREEN}✅ OpenAPI matches baseline${NC}"

# Step 3: Check API docs sync
echo "3. Checking API docs sync..."
if ! uv run python scripts/sync_api_docs.py --check; then
    echo -e "${RED}❌ API docs artifacts are out of sync${NC}"
    exit 1
fi
echo -e "${GREEN}✅ API docs artifacts are in sync${NC}"

# Step 4: Run pytest
echo "4. Running tests..."
if ! uv run pytest; then
    echo -e "${RED}❌ Tests failed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ All tests passed${NC}"

# Step 5: Streaming smoke test
if ! ./scripts/streaming_smoke_test.sh; then
    echo -e "${RED}❌ Streaming smoke test failed${NC}"
    exit 1
fi
echo -e "${GREEN}🎉 All regression checks passed!${NC}"