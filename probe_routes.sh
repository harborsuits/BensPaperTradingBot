#!/bin/bash
cd /Users/bendickinson/Desktop/benbot

# Array of URLs to probe
urls=(
  "/api/metrics"
  "/api/metrics/strategies"
  "/api/metrics/strategies?all=true"
  "/api/strategies/news_momo_v2"
  "/api/strategies/news_momo_v2/metrics"
  "/api/strategies/news_momo_v2/trades"
  "/api/strategies/mean_rev"
  "/api/strategies/mean_rev/metrics"
  "/api/strategies/mean_rev/trades"
  "/api/orders?limit=50"
  "/api/positions"
  "/api/trades?limit=100"
  "/api/fills?limit=100"
  "/api/pnl"
  "/api/pnl/daily"
  "/api/live/status"
  "/api/live/ai/status"
  "/api/ai/status"
  "/api/ai/logs?limit=50"
  "/api/ai/orchestrator/status"
  "/api/strategies/manager/status"
  "/api/tournament/status"
  "/api/websocket/status"
  "/api/market/recent"
  "/api/data/alerts/count"
  "/api/data/evotester/stats"
)

echo "Starting API route probe..." > evidence/_probe.log

for url in "${urls[@]}"; do
  echo ">> GET $url" | tee -a evidence/_probe.log
  # Create safe filename by replacing special chars
  fname="evidence$(echo "$url" | tr '/?&=\\' '_____' ).json"
  # Make the request and capture HTTP status
  http_code=$(curl -sS -o "$fname" -w "%{http_code}" "http://localhost:4000$url")
  echo "HTTP:$http_code" | tee -a evidence/_probe.log
  echo "" | tee -a evidence/_probe.log
done

echo "Probe complete. Results in evidence/_probe.log"
