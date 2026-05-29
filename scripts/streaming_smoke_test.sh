#!/bin/bash
set -euo pipefail

# Streaming smoke test script for regression guard
# Checks that the LangGraph stream returns SSE values events and closes cleanly.

API_HOST="127.0.0.1"
API_PORT="3002"
API_URL="http://${API_HOST}:${API_PORT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running streaming smoke test...${NC}"

# Check if API is running
if ! curl -s --max-time 5 "${API_URL}/health" > /dev/null; then
    echo -e "${RED}❌ API not running on ${API_URL}${NC}"
    echo -e "${YELLOW}Start API with: ./run_api.sh${NC}"
    exit 1
fi

# Perform streaming request and capture output
SMOKE_OUTPUT=$(mktemp)
HEADER_FILE=$(mktemp)

if ! curl -N -X POST "${API_URL}/api/langgraph/threads/smoke-thread/runs/stream" \
    -H "Content-Type: application/json" \
    -d '{"assistant_id":"mcp_agent_executor","input":{"messages":[{"type":"human","content":"Hello"}]}}' \
    --dump-header "$HEADER_FILE" \
    --max-time 30 2>/dev/null > "$SMOKE_OUTPUT"; then
    echo -e "${RED}❌ Streaming request failed${NC}"
    rm -f "$SMOKE_OUTPUT" "$HEADER_FILE"
    exit 1
fi

# Check SSE content type
if ! grep -qi '^content-type: text/event-stream' "$HEADER_FILE"; then
    echo -e "${RED}❌ Missing required content type: text/event-stream${NC}"
    echo -e "${YELLOW}Response headers:${NC}"
    cat "$HEADER_FILE"
    rm -f "$SMOKE_OUTPUT" "$HEADER_FILE"
    exit 1
fi

# Check stream emits LangGraph values events. Completion is stream close.
if ! grep -q "^event: values$" "$SMOKE_OUTPUT"; then
    echo -e "${RED}❌ Stream did not emit a values event${NC}"
    echo -e "${YELLOW}Last 10 lines of stream output:${NC}"
    tail -10 "$SMOKE_OUTPUT"
    rm -f "$SMOKE_OUTPUT" "$HEADER_FILE"
    exit 1
fi

# Clean up
rm -f "$SMOKE_OUTPUT" "$HEADER_FILE"

echo -e "${GREEN}✅ Streaming smoke test passed${NC}"
