# 🚀 Monday Market Open Checklist

## Pre-Market (9:00 AM ET)

### 1. Start Core Services
```bash
# Kill any stale processes
ps aux | grep "node.*server.js" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

# Start backend with all features
cd ~/Desktop/benbot/live-api
TRADIER_API_KEY=KU2iUnOZIUFre0wypgyOn8TgmGxI \
TRADIER_ACCOUNT_ID=VA1201776 \
PORT=4000 \
AUTOLOOP_ENABLED=1 \
STRATEGIES_ENABLED=1 \
AI_ORCHESTRATOR_ENABLED=1 \
AUTO_EVOLUTION_ENABLED=1 \
node server.js > ../backend-live.log 2>&1 &

# Start Python brain (optional - will use fallback if not running)
cd ~/Desktop/benbot
python ai_scoring_service.py > ai_scoring.log 2>&1 &
```

### 2. Reset Circuit Breaker
```bash
# Reset circuit breaker if it's blocking trades
curl -X POST http://localhost:4000/api/circuit-breaker/reset
```

### 3. Verify System Health
```bash
# Check all systems
curl http://localhost:4000/api/health | jq '.'
curl http://localhost:4000/api/autoloop/status | jq '.'
curl http://localhost:4000/api/brain/flow | jq '.'
```

## Market Open (9:30 AM ET)

### 4. Monitor Initial Activity
```bash
# Watch for trading signals
tail -f ~/Desktop/benbot/backend-live.log | grep -E "Brain decision|Signal generated|Order submitted"

# Check decisions being made
watch -n 10 'curl -s http://localhost:4000/api/decisions/recent | jq ". | length"'
```

### 5. Force Test Trade (if needed)
```bash
# If no trades after 30 minutes, force AutoLoop
curl -X POST http://localhost:4000/api/test/autoloop/runonce
```

## Expected Behavior

When markets open, you should see:
1. **Brain Decisions**: "Brain decision for [SYMBOL]: BUY/SELL (score: 0.XX)"
2. **Signal Generation**: "Signal generated for [SYMBOL]"
3. **Order Submission**: "Order submitted: [ID]"
4. **Evolution Triggers**: After 50 trades

## Troubleshooting

### If No Trades Execute:
- Check circuit breaker: `curl http://localhost:4000/api/circuit-breaker/status`
- Lower thresholds temporarily in `/live-api/config/tradingThresholds.js`
- Check for rate limit errors in logs

### If Rate Limited:
- Increase intervals in `computeQuotesInterval()` 
- Use production API key instead of sandbox

### Emergency Stop:
```bash
curl -X POST http://localhost:4000/api/autoloop/stop
curl -X POST http://localhost:4000/api/circuit-breaker/trip
```

## Success Metrics

✅ AutoLoop generating signals every cycle
✅ Brain scoring all candidates
✅ Orders executing in paper account
✅ Evolution system counting trades
✅ No rate limit errors

## Notes

- The system uses simulated brain scoring (0.45-0.65) when Python brain is offline
- Buy threshold: score > 0.45
- Sell threshold: score < 0.35
- Evolution triggers after 50 trades
- Circuit breaker trips at 2% daily loss or 5% drawdown
