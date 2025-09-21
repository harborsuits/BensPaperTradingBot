#!/bin/bash

# Test script for dashboard endpoints
# Run this after starting the backend server on localhost:4000

BASE_URL="http://localhost:4000"

echo "🧪 Testing Dashboard Endpoints"
echo "================================"

# Health + Autopilot
echo "📊 Health + Autopilot:"
echo "curl -s $BASE_URL/api/health | jq '{ok,breaker,broker_ok:(.broker.ok)}'"
curl -s "$BASE_URL/api/health" | jq '{ok,breaker,broker_ok:(.broker.ok)}' 2>/dev/null || echo "❌ Health endpoint failed"
echo ""

echo "curl -s $BASE_URL/api/audit/autoloop/status | jq '{mode,status:(.status//.running),tick:(.tick_ms//.interval_ms)}'"
curl -s "$BASE_URL/api/audit/autoloop/status" | jq '{mode,status:(.status//.running),tick:(.tick_ms//.interval_ms)}' 2>/dev/null || echo "❌ Autoloop status failed"
echo ""

# Portfolio (server authoritative)
echo "💰 Portfolio (server authoritative):"
echo "curl -s $BASE_URL/api/portfolio/summary | jq '{equity, cash, day_pnl, open_pnl, positions_len:(.positions|length)}'"
curl -s "$BASE_URL/api/portfolio/summary" | jq '{equity, cash, day_pnl, open_pnl, positions_len:(.positions|length)}' 2>/dev/null || echo "❌ Portfolio summary failed"
echo ""

# Dashboard summaries (what cards render)
echo "📈 Dashboard summaries:"
echo "curl -s '$BASE_URL/api/brain/flow/summary?window=15m' | jq"
curl -s "$BASE_URL/api/brain/flow/summary?window=15m" | jq 2>/dev/null || echo "❌ Brain flow summary failed"
echo ""

echo "curl -s '$BASE_URL/api/decisions/summary?window=15m' | jq"
curl -s "$BASE_URL/api/decisions/summary?window=15m" | jq 2>/dev/null || echo "❌ Decisions summary failed"
echo ""

echo "curl -s '$BASE_URL/api/paper/orders?limit=3' | jq 'length'"
curl -s "$BASE_URL/api/paper/orders?limit=3" | jq 'length' 2>/dev/null || echo "❌ Paper orders failed"
echo ""

# Decisions page data
echo "🎯 Decisions page data:"
echo "curl -s '$BASE_URL/api/decisions/recent?stage=proposed&limit=50' | jq '.[0] // (.items[0])'"
curl -s "$BASE_URL/api/decisions/recent?stage=proposed&limit=50" | jq '.[0] // (.items[0])' 2>/dev/null || echo "❌ Proposals failed"
echo ""

echo "curl -s '$BASE_URL/api/decisions/recent?stage=intent&limit=50' | jq 'length'"
curl -s "$BASE_URL/api/decisions/recent?stage=intent&limit=50" | jq 'length' 2>/dev/null || echo "❌ Intents failed"
echo ""

# Telemetry (which cards actually mounted)
echo "📊 Telemetry (which cards actually mounted):"
echo "curl -s $BASE_URL/api/telemetry/cards | jq"
curl -s "$BASE_URL/api/telemetry/cards" | jq 2>/dev/null || echo "❌ Telemetry failed"
echo ""

echo "✅ Dashboard endpoint testing complete!"
echo "=========================================="
echo "💡 Pro tip: Run this script after starting your backend server"
echo "   to verify all dashboard endpoints are working correctly."
