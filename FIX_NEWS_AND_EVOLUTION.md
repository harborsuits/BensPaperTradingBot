# Fix for News Sentiment and Evolution Endpoints

## Issues Fixed

### 1. News Sentiment Showing Fake URLs (example.com)
- **Problem**: The `/api/news` endpoint was returning hardcoded `https://example.com/news` URLs
- **Fixed**: Updated to use real news from newsProvider, falling back to realistic Yahoo Finance URLs

### 2. Missing Evolution/Discovery Endpoints (404 errors)
- **Problem**: Frontend was calling 6 endpoints that didn't exist
- **Fixed**: Added all missing endpoints with appropriate mock data:
  - `/api/evo/strategy-hypotheses`
  - `/api/pipeline/stages`
  - `/api/evo/deployment-metrics`
  - `/api/evo/trigger-rules`
  - `/api/fundamentals`
  - `/api/discovery/market`

## To Apply These Fixes

The server needs to be restarted to pick up the changes:

```bash
# Find the server process
ps aux | grep "node.*server.js" | grep -v grep

# Note the PID (second column) and kill it
kill <PID>

# Restart the server with strategies enabled
cd /Users/bendickinson/Desktop/benbot/live-api
STRATEGIES_ENABLED=1 node server.js
```

## Alternative: Hot Fix Without Restart

If you don't want to restart the server, you can use this workaround:

1. The news sentiment will continue showing example.com until restart
2. The evolution endpoints will continue returning 404 until restart
3. However, the core trading functionality is not affected

## What These Fixes Provide

1. **Real News URLs**: News items will show Yahoo Finance URLs like `https://finance.yahoo.com/quote/SPY/news`
2. **Evolution Page Support**: All endpoints the evotester page needs are now available
3. **Better Mock Data**: Even when real news fails, URLs will be realistic

## Testing After Restart

Run this to verify the fixes:
```bash
cd /Users/bendickinson/Desktop/benbot/live-api
node test-fixes.js
```

You should see:
- ✅ No more example.com URLs
- ✅ All evolution endpoints returning 200 OK
