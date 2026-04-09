#!/usr/bin/env bash
# Quick check: do Loki and Tempo have data from the RAG API?
# Run from project root. Requires: stack up, API running with ENABLE_OTEL_TRACING=1.

set -e
LOKI="${LOKI:-http://localhost:3100}"
TEMPO="${TEMPO:-http://localhost:3200}"
API="${API:-http://localhost:3002}"
COLLECTOR="${COLLECTOR:-http://localhost:4318}"

echo "=== 0. Collector reachable? (API must send to this endpoint) ==="
# OTLP receiver only accepts POST on /v1/traces and /v1/logs; GET to root may 404/405. Use POST to /v1/traces.
CODE=$(curl -s --connect-timeout 2 -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/x-protobuf" --data-binary "" "$COLLECTOR/v1/traces" 2>/dev/null || echo "000")
if [ "$CODE" != "000" ]; then
  echo "  Collector at $COLLECTOR is reachable (POST /v1/traces → HTTP $CODE). API uses $COLLECTOR/v1/traces and /v1/logs."
else
  echo "  WARNING: Cannot reach collector at $COLLECTOR. Start stack: docker compose --profile observability up -d loki tempo otel-collector grafana"
  echo "  If the API runs in Docker, set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://host.docker.internal:4318/v1/traces"
fi

echo ""
echo "=== 1. Generate traffic (API must be running with OTel enabled) ==="
if curl -sf --connect-timeout 2 -o /dev/null "$API/health"; then
  echo "  API is up. Sending 3 health requests..."
  for i in 1 2 3; do curl -sf -o /dev/null "$API/health"; sleep 0.5; done
else
  echo "  WARNING: API not reachable at $API/health. Start it with ENABLE_OTEL_TRACING=1 and run_api.sh"
fi

echo ""
echo "  Waiting 10s for collector batch flush..."
sleep 10

echo ""
echo "=== 2. Loki: recent log streams (OTLP uses service_name=rag-api) ==="
START=$(python3 -c 'import time; print(int((time.time()-300)*1e9))')
END=$(python3 -c 'import time; print(int(time.time()*1e9))')
# OTLP logs use index label service_name (service.name); try that first, then {}
LOKI_RESP=$(curl -sS -G "$LOKI/loki/api/v1/query_range" \
  --data-urlencode 'query={service_name="rag-api"}' \
  --data-urlencode "start=$START" \
  --data-urlencode "end=$END" \
  --data-urlencode "limit=20") || true
LOKI_PARSE=$(echo "$LOKI_RESP" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('data', {}).get('result', [])
    print('count', len(r))
    for s in r[:3]:
        labels = s.get('stream', {})
        print('  labels:', labels)
        vals = s.get('values', [])
        if vals:
            print('  sample:', vals[0][1][:80] if len(vals[0][1]) > 80 else vals[0][1])
except Exception as e:
    print('error', str(e))
" 2>/dev/null || echo "error parse failed")
LOKI_COUNT=$(echo "$LOKI_PARSE" | sed -n 's/^count \([0-9]*\)/\1/p')
if [ -z "$LOKI_COUNT" ] || [ "$LOKI_COUNT" = "0" ] || echo "$LOKI_PARSE" | grep -q "^error "; then
  echo "  No log streams in last 5 minutes (or Loki returned an error)."
  echo "  → In Grafana Explore (Loki) try: {service_name=\"rag-api\"} or {} with 'Last 1 hour'."
else
  echo "  Found $LOKI_COUNT log stream(s). Sample:"
  echo "$LOKI_PARSE" | sed -n '2,$p'
fi

echo ""
echo "=== 3. Tempo: search tags (should include service.name if traces exist) ==="
TAGS=$(curl -sS "$TEMPO/api/v2/search/tags?limit=50" 2>/dev/null || true)
if echo "$TAGS" | python3 -c "import sys, json; d=json.load(sys.stdin); exit(0 if d.get('tagNames') else 1)" 2>/dev/null; then
  echo "  Tag names: $(echo "$TAGS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('tagNames', []))" 2>/dev/null)"
else
  echo "  Could not read tags (Tempo may have no traces yet)."
fi

echo ""
echo "=== 4. Tempo: recent traces (service name = rag-api) ==="
# Try legacy /api/search?limit=&serviceName= and v2-style tags
TRACES=$(curl -sS -G "$TEMPO/api/search" \
  --data-urlencode "limit=5" \
  --data-urlencode "serviceName=rag-api" 2>/dev/null) || true
TRACE_COUNT=$(echo "$TRACES" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('traces', [])
    print(len(t))
    for x in t[:2]:
        print('  traceID:', x.get('traceID'))
except Exception:
    print(0)
" 2>/dev/null || echo "0")
if [ "$TRACE_COUNT" = "0" ] || [ -z "$TRACE_COUNT" ]; then
  echo "  No traces found for service.name=rag-api."
  echo "  → In Grafana Explore (Tempo): pick 'Search', set Service name to 'rag-api', time range 'Last 15 minutes'."
  echo "  → Restart API with ENABLE_OTEL_TRACING=1 (or true in .env) and send a few requests."
else
  echo "  Found $TRACE_COUNT trace(s). Sample traceIDs above."
fi

echo ""
echo "=== Done ==="
