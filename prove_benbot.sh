#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://localhost:4000}"
OUT="${OUT:-evidence}"
mkdir -p "$OUT"

echo "==> Health & roster"
curl -sS "$BASE/api/health"            | tee "$OUT/health.json"            >/dev/null
curl -sS "$BASE/api/strategies"        | tee "$OUT/strategies.json"        >/dev/null
curl -sS "$BASE/api/decisions/recent"  | tee "$OUT/decisions_recent.json"  >/dev/null
curl -sS "$BASE/api/paper/account"     | tee "$OUT/paper_account.json"     >/dev/null

echo "==> Audit & risk routes (newly implemented)"
for u in \
  /api/audit/coordination \
  /api/audit/risk-rejections \
  /api/audit/allocations/current \
  /api/audit/autoloop/status \
  /api/audit/system
do
  echo "GET $u"
  curl -sS "$BASE$u" | tee "$OUT$(echo "$u" | tr '/?' '__')".json >/dev/null || true
done

echo "==> Orders/positions/trades endpoints"
for u in \
  /api/paper/orders \
  /api/paper/positions \
  /api/trades \
  /api/fills \
  /api/pnl \
  /api/pnl/daily
do
  echo "GET $u"
  curl -sS "$BASE$u" | tee "$OUT$(echo "$u" | tr '/?' '__')".json >/dev/null || true
done

echo "==> Light schema checks (jq required)"
have_jq=1; command -v jq >/dev/null || have_jq=0
if [ $have_jq -eq 1 ]; then
  echo "Health ok?";         jq -r '.ok' "$OUT/health.json"
  echo "Breaker status:";    jq -r '.breaker' "$OUT/health.json"
  echo "Strategies found:";  jq -r '.items[]?.id' "$OUT/strategies.json" 2>/dev/null || jq -r '.[]?.id' "$OUT/strategies.json" || true
  echo "Autoloop status:";   jq '.[0]? // .' "$OUT__api_audit_autoloop_status.json" 2>/dev/null || true
  echo "Allocations:";        jq '.' "$OUT__api_audit_allocations_current.json" 2>/dev/null || true
fi

echo "==> Done. Evidence in $OUT/"
ls -1 "$OUT"
