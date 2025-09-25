# ✅ BenBot Stock Paper Trading - READY TO TEST

## 🚀 Quick Start (5 minutes)

### 1. Restart Clean
```bash
# Kill any existing processes
ps aux | grep node | grep minimal_server | awk '{print $2}' | xargs kill -9

# Start fresh with all systems
cd ~/Desktop/benbot/live-api
TRADIER_API_KEY=KU2iUnOZIUFre0wypgyOn8TgmGxI \
TRADIER_ACCOUNT_ID=VA1201776 \
PORT=4000 \
AUTOLOOP_ENABLED=1 \
STRATEGIES_ENABLED=1 \
AI_ORCHESTRATOR_ENABLED=1 \
AUTO_EVOLUTION_ENABLED=1 \
node minimal_server.js
```

### 2. Monitor Key Metrics
```bash
# In another terminal - watch for issues
watch -n 5 'curl -s http://localhost:4000/api/capital/status | jq .'
```

### 3. What to Expect at Market Open (9:30 AM ET)
- ✅ 459 pending orders will start processing
- ✅ Capital tracker will prevent overtrading
- ✅ Bot competition continues evolving strategies
- ✅ Only 10 new strategies promoted per cycle

## ⚠️ Critical Issues to Watch

1. **Overcommitted Capital** - You have $206K in pending orders on $100K account
   - System will fill ~220 orders then stop
   - Rest will auto-cancel

2. **No Stop Losses** - Strategies don't have risk management yet
   - Monitor closely for first day
   - Add stops after seeing behavior

3. **Empty Symbol Discovery** - AutoLoop needs symbols
   ```bash
   # Check if finding symbols
   curl http://localhost:4000/api/discovery/dynamic-symbols
   ```

## 📊 Success Metrics

After 1 day of trading, check:
- [ ] Trades executed successfully
- [ ] No capital overruns (stays under 95%)
- [ ] Bot competition evolved new strategies
- [ ] No system crashes

## 🔮 Future (After Testing Works)

1. **Enable Options Trading**
   - You already have all the infrastructure
   - Just need to activate endpoints
   - Add to UI

2. **Enable Crypto**
   - Providers ready (Coinbase, Binance)
   - Need orchestrator integration
   - 24/7 trading capability

3. **Deploy to Cloud**
   - Use deployment guide
   - $20-40/month VPS
   - Runs with laptop closed

## 🚨 Emergency Controls

If things go wrong:
```bash
# Stop all trading
curl -X POST http://localhost:4000/api/autoloop/stop

# Cancel all orders
curl -X POST http://localhost:4000/api/capital/emergency-release

# Check what happened
curl http://localhost:4000/api/trades | jq '.[-10:]'
```
