#!/usr/bin/env bash
set -euo pipefail

echo "== Health & Metrics =="
curl -s http://localhost:3003/api/health | jq .
curl -s http://localhost:3003/metrics | jq .

echo "== Strategies & Context =="
curl -s http://localhost:3003/api/strategies/active | jq '.[0] // .items[0]'
curl -s http://localhost:3003/api/context/regime | jq .
curl -s http://localhost:3003/api/context/volatility | jq .
curl -s http://localhost:3003/api/context/sentiment | jq .

echo "== Paper account/positions =="
curl -s http://localhost:3003/api/paper/account | jq .
curl -s http://localhost:3003/api/paper/positions | jq '.items // .'

echo "== Place sample paper order =="
OID=$(curl -s -X POST http://localhost:3003/api/paper/orders -H "Content-Type: application/json" --data-binary '{"symbol":"AAPL","side":"buy","qty":5,"type":"market"}' | jq -r .id)
echo "Order ID: $OID"
curl -s "http://localhost:3003/api/paper/orders/$OID" | jq .

echo "== Trades feed =="
curl -s "http://localhost:3003/api/trades?limit=5" | jq '.items // .'


