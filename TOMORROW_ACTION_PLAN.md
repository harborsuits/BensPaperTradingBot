# 🎯 ACTION PLAN FOR TOMORROW'S TESTING

## IMMEDIATE FIXES (Do These First at Market Open)

### 1. Force AutoLoop to Start
```bash
# At 9:30 AM ET, manually start if scheduler fails:
curl -X POST http://localhost:4000/api/autoloop/start
```

### 2. Monitor Evolution Winners
```bash
# Check which strategies are winning:
curl http://localhost:4000/api/tournament/leaderboard | jq '.'
```

### 3. Enable Dynamic Discovery
In `/live-api/lib/autoLoop.js`, the code already checks `this.dynamicDiscovery` but falls back to regular scanner. Need to ensure dynamic discovery endpoint is working:
```bash
curl http://localhost:4000/api/discovery/dynamic-symbols
```

## POWER FEATURES TO ACTIVATE

### 1. **Connect Memory to Decisions**
The bot should query its memory before making decisions:
```javascript
// In autoLoop._determineSide(), add:
const pastPerformance = await performanceRecorder.getStrategyStats(strategyId);
const marketMemory = geneticInheritance.getBestGenesForConditions(marketConditions);
```

### 2. **Use Tournament Winners**
Currently strategies in R1/R2/R3 aren't promoted to live trading:
```javascript
// Query tournament winners
const winners = await tournamentController.getPromotionCandidates('R3');
// Add their strategies to AutoLoop
```

### 3. **Enable Contextual Bandit Learning**
RouteSelector has Thompson Sampling but isn't connected:
```javascript
// After each trade
routeSelector.updateBanditModel({
  outcome: { route, pnl, friction, drawdown }
});
```

### 4. **Fix News Integration**
News nudge exists but news scanner not feeding candidates:
```javascript
// In AutoLoop, replace static scanner with:
const newsMovers = await fetch('/api/context/news/movers');
const candidates = newsMovers.symbols;
```

## MONITORING COMMANDS

### Every 5 Minutes During Market Hours:
```bash
# Check if bot is making decisions
watch -n 300 'curl -s http://localhost:4000/api/decisions/recent | jq length'

# Monitor brain activity
curl http://localhost:4000/api/brain/flow/summary

# Check evolution progress
curl http://localhost:4000/api/evo/status | jq '.generation'

# Verify AutoLoop is running
curl http://localhost:4000/api/autoloop/status | jq '.is_running'
```

## EXPECTED TIMELINE

### Hour 1 (9:30-10:30 AM):
- AutoLoop starts automatically (or manually)
- Should see brain scoring activity
- Evolution should trigger if configured

### Hour 2-3 (10:30 AM-12:30 PM):
- First trades should execute if thresholds met
- Memory systems start recording outcomes
- Bot competition trades accumulate

### Hour 4-6 (12:30-3:30 PM):
- Evolution cycle completes (after 50 trades)
- New strategies generated from winners
- Performance patterns emerge in memory

### End of Day (3:30-4:00 PM):
- Tournament evaluations run
- Successful strategies promoted
- Learning consolidated for next day

## SUCCESS METRICS

✅ **Minimum Success** (What we MUST see):
- [ ] AutoLoop runs during market hours
- [ ] Brain scores symbols continuously
- [ ] At least 1 trade executes

📈 **Good Progress** (What we SHOULD see):
- [ ] 10+ trades executed
- [ ] Evolution triggers at least once
- [ ] Memory systems record outcomes

🚀 **Excellent Progress** (What we HOPE to see):
- [ ] 50+ trades trigger evolution
- [ ] Tournament promotes strategies
- [ ] Bot learns and adapts thresholds

## TROUBLESHOOTING

### If No Trades Execute:
1. Check brain scores: `curl http://localhost:4000/api/brain/flow/summary`
2. Lower thresholds further in tradingThresholds.js
3. Force a test trade: `curl -X POST http://localhost:4000/api/paper/orders -d '{"symbol":"AAPL","qty":1,"side":"buy"}'`

### If AutoLoop Doesn't Start:
1. Check scheduler: `pm2 logs benbot-backend | grep "Market open"`
2. Manually start: `curl -X POST http://localhost:4000/api/autoloop/start`
3. Verify environment: `pm2 env 0 | grep AUTOLOOP_ENABLED`

### If Evolution Doesn't Trigger:
1. Check trade count: `curl http://localhost:4000/api/evo/status | jq '.tradesSinceLastEvolution'`
2. Force evolution: `curl -X POST http://localhost:4000/api/bot-competition/evolve`
3. Verify bot competition: `curl http://localhost:4000/api/bot-competition/active`

## THE CORE ISSUE

**The bot has a Ferrari engine but no transmission!** All the advanced AI components exist but aren't connected:
- Brain ✅ → Decisions ❌
- Memory ✅ → Learning ❌  
- Evolution ✅ → Trading ❌
- News ✅ → Discovery ❌

Tomorrow's test will reveal if connecting these systems enables true autonomous trading and learning.
