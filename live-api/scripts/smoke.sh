#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://localhost:4000}
PY=${PY:-http://localhost:8001}
SYM=${SYM:-SPY}

echo "== Health"
curl -sf "$BASE/api/health" | jq -C . | head -n 50
curl -sf "$BASE/api/indicators/health" | jq -C . | head -n 50
curl -sf "$PY/health" | jq -C . | head -n 50

echo "== Score"
S=$(curl -sf -X POST "$BASE/api/brain/score" -H 'content-type: application/json' -d "{\"symbol\":\"$SYM\",\"snapshot_ts\":\"NOW\"}")
echo "$S" | jq -C . | head -n 50
FALL=$(echo "$S" | jq -r '.fallback // false')
if [ "$FALL" = "true" ]; then echo "WARN: fallback active"; fi

echo "== Plan (no halt)"
P=$(curl -sf -X POST "$BASE/api/brain/plan" -H 'content-type: application/json' -d "{\"symbol\":\"$SYM\",\"final_score\":9.6,\"conf\":0.78,\"account_state\":{\"equity\":3000,\"buying_power\":9000,\"day_drawdown_pct\":0.5}}")
echo "$P" | jq -C . | head -n 50

echo "== Halt drill (DD)"
PH=$(curl -sf -X POST "$BASE/api/brain/plan" -H 'content-type: application/json' -d "{\"symbol\":\"$SYM\",\"final_score\":9.6,\"conf\":0.78,\"account_state\":{\"equity\":3000,\"buying_power\":9000,\"day_drawdown_pct\":9.9}}")
echo "$PH" | jq -C . | head -n 50

echo "== Metrics"
curl -sf "$BASE/metrics" | head -n 30
echo
echo "OK"
