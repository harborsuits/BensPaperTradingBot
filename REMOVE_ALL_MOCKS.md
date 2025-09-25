# 🚫 Mock Data Removal Checklist

## Critical Mock Data to Remove:

### 1. WebSocket Mock Prices/Decisions
- File: `minimal_server.js` lines 3995-4023
- Remove all `mockPrice` and `mockDecision` generation

### 2. Brain Service Fallbacks
- File: `src/services/BrainService.js` lines 154-201
- Change fallbacks to throw errors instead

### 3. Mock Equity Account
- File: `minimal_server.js` lines 65-136
- Remove `mockEquityAccount` completely

### 4. Default Symbol Lists
- File: `minimal_server.js` line 2953
- Return empty array instead of ['SPY','QQQ','AAPL']

### 5. Market Indicators Mock
- File: `minimal_server.js` line 380
- Pass real data source or let it fail

### 6. Diamond Scorer Mock Data
- File: `src/services/diamondsScorer.js` lines 186-207
- Remove mock volume/price data generation

### 7. Sample Strategies
- File: `minimal_server.js` line 175
- Remove hardcoded "sample" strategies

## Environment Variables to Set:
```bash
export FORCE_NO_MOCKS=true
export PAPER_MOCK_MODE=false
```

## Expected Behavior After Removal:
- If Tradier fails → Error, not mock data
- If Brain fails → Error, not fallback scoring
- If News fails → Empty results, not defaults
- If WebSocket connects → Real data only

## Why This Matters:
"We need to see what's not producing" - if something breaks, we MUST know immediately, not hide behind fake data!
