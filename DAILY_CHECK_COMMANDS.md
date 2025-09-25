# 📊 Daily Benbot Check Commands

## Quick Status Check (Run these tomorrow)

### 1. Check Today's Trades
```bash
curl -s http://localhost:4000/api/trades | jq '.items[:5]'
```

### 2. See Today's Performance
```bash
curl -s http://localhost:4000/api/portfolio/summary | jq '.'
```

### 3. View Active Positions
```bash
curl -s http://localhost:4000/api/paper/positions | jq '.'
```

### 4. Check Decision Log (Why trades were made)
```bash
curl -s http://localhost:4000/api/decisions | jq '.items[:5] | .[] | {symbol, action, confidence, reason}'
```

### 5. See AutoLoop Activity
```bash
pm2 logs benbot-backend --lines 100 | grep -E "AutoLoop|Order placed|Signal generated"
```

### 6. View Evolution Progress
```bash
curl -s http://localhost:4000/api/bot-competition/active | jq '.competitions[0] | {id, status, leaderboard: .leaderboard[:3]}'
```

### 7. Check System Health
```bash
curl -s http://localhost:4000/api/health | jq '{status, broker, marketData}'
```

## 📈 Dashboard Views

Open http://localhost:3003 and check:

1. **Main Dashboard**: Overall performance
2. **Trade Decisions**: See every trade with reasoning
3. **Evolution Lab**: Watch bots compete and evolve
4. **Market Data**: Real-time quotes and indicators

## 📊 Performance Report

### Generate Daily Summary
```bash
# See today's summary
pm2 logs benbot-backend | grep -A 20 "DailyReport"

# Check win/loss ratio
curl -s http://localhost:4000/api/trades | jq '[.items[] | select(.timestamp > "'$(date -u +%Y-%m-%d)'T00:00:00Z")] | {total: length, wins: [.[] | select(.pnl > 0)] | length}'
```

## 🔍 Detailed Analysis

### Why Each Trade Was Made
```bash
# Get detailed decision reasoning
curl -s http://localhost:4000/api/decisions | jq '.items[:10] | .[] | {time: .timestamp, symbol, action, confidence, brain_score, reason}'
```

### Strategy Performance
```bash
# See which strategies are working
curl -s http://localhost:4000/api/strategies | jq '.items[] | {name, stage, performance}'
```

## 🚨 Quick Troubleshooting

### If No Trades Happened
```bash
# Check if AutoLoop is running
curl -s http://localhost:4000/api/autoloop/status | jq '.'

# Check for errors
pm2 logs benbot-backend --err --lines 50
```

### If You Want More Activity
```bash
# Start a bot competition
curl -X POST http://localhost:4000/api/bot-competition/news-triggered

# Check circuit breaker status
curl -s http://localhost:4000/api/circuit-breaker/status | jq '.'
```

## 📱 Mobile Check

You can even check from your phone:
1. Open browser to `http://[your-mac-ip]:3003`
2. View dashboard on mobile
3. All real-time updates work!

---

**Remember**: First trades execute at 9:30 AM EST when market opens!
