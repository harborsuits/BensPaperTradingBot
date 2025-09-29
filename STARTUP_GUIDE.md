# BenBot Startup Guide

## Quick Start (Recommended)
Just run this single command:
```bash
cd ~/Desktop/benbot/live-api && ./start-benbot.sh
```

This will:
- Kill any old processes
- Start the API server with all features
- Activate all strategies
- Start the dashboard
- Open http://localhost:3003 in your browser

## Manual Start (If needed)
1. **Kill old processes:**
   ```bash
   lsof -ti:4000 | xargs kill -9
   lsof -ti:3003 | xargs kill -9
   ```

2. **Start API server:**
   ```bash
   cd ~/Desktop/benbot/live-api
   ./start-server.sh
   ```

3. **Activate strategies:**
   ```bash
   curl -X POST http://localhost:4000/api/strategies/activate-all
   ```

4. **Start dashboard:**
   ```bash
   cd ~/Desktop/benbot/new-trading-dashboard
   npm run dev
   ```

5. **Open dashboard:**
   http://localhost:3003

## Troubleshooting

### Dashboard shows errors?
1. Hard refresh: Cmd+Shift+R (Mac) or Ctrl+F5 (Windows)
2. Check server: `curl http://localhost:4000/api/health`
3. Restart using: `./start-benbot.sh`

### "Port already in use" error?
```bash
lsof -ti:4000 | xargs kill -9
lsof -ti:3003 | xargs kill -9
```

### Verify fixes are intact:
```bash
cd ~/Desktop/benbot/live-api
./verify-dashboard-fixes.sh
```

## What Each Error Means
- **"Cannot read properties of null"**: Strategies not activated
- **NaN in positions**: Old server running without fixes
- **Stale prices**: No price provider or old server

## Remember
- ALWAYS use `start-benbot.sh` or `start-server.sh`
- NEVER run just `node minimal_server.js` without env vars
- If another AI assistant helps, point them to `AI_ASSISTANT_GUIDE.md`
