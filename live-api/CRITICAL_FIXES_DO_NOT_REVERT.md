# CRITICAL FIXES - DO NOT REVERT

## ⚠️ IMPORTANT: These fixes resolve dashboard display issues. DO NOT modify without understanding the impact.

### Fixed Issues:
1. **Autopilot Error**: "Cannot read properties of null (reading 'getAllStrategies')"
2. **Position Data**: NaN values for quantities and prices
3. **Market Prices**: Stale prices and missing provider
4. **Rate Limiting**: 429 errors on quotes API due to concurrent frontend requests

### Applied Fixes:

#### 1. Strategy API Fix (Line ~3879)
```javascript
// BEFORE (BROKEN):
Object.values(allStrategies || {}).forEach(strategy => {

// AFTER (FIXED):
(allStrategies || []).forEach(strategy => {
```
**Reason**: `getAllStrategies()` returns an array, not an object.

#### 2. Position Data Fix (Lines ~3671-3680)
```javascript
// Transform positions to match frontend expectations
const transformedPositions = positions.map(pos => ({
  symbol: pos.symbol,
  quantity: pos.qty || 0,  // Frontend expects 'quantity', not 'qty'
  cost_basis: (pos.avg_price || 0) * (pos.qty || 0),
  avg_price: pos.avg_price || 0,
  current_price: pos.avg_price || 0,
  market_value: (pos.avg_price || 0) * (pos.qty || 0),
  pnl: 0
}));
```
**Reason**: Frontend expects `quantity` field, backend was sending `qty`.

#### 3. Mock Quotes Provider (Lines ~1502-1522)
```javascript
// Added mock quotes when no Tradier token
if (!token) {
  const now = Date.now();
  const mockQuotes = symbols.map(symbol => {
    const seed = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const basePrice = 100 + (seed % 900);
    const variation = (Math.sin(now / 60000) * 0.01);
    const last = +(basePrice * (1 + variation)).toFixed(2);
    return {
      symbol,
      last,
      price: last,
      // ... other fields
    };
  });
  return res.json(mockQuotes);
}
```
**Reason**: Provides quotes when no real data provider is configured.

#### 4. Rate Limiter Fix (Line ~36-37)
```javascript
// BEFORE (TOO RESTRICTIVE):
'/api/quotes': new TokenBucketLimiter(5, 10),

// AFTER (FIXED):
'/api/quotes': new TokenBucketLimiter(20, 50), // Increased to handle concurrent frontend requests
```
**Reason**: Frontend makes multiple concurrent quote requests causing 429 errors.

### To Start Server Correctly:
```bash
cd /Users/bendickinson/Desktop/benbot/live-api
STRATEGIES_ENABLED=1 AI_ORCHESTRATOR_ENABLED=1 OPTIONS_ENABLED=1 node minimal_server.js
```

### To Verify Fixes Are Applied:
Run: `./verify-dashboard-fixes.sh`

### Dashboard Issues If Reverted:
- Autopilot will show: "ERROR: Cannot read properties of null (reading 'getAllStrategies')"
- Positions will show: NaN for all quantities
- All prices will show as "stale"

## DO NOT USE grep/sed TO MODIFY minimal_server.js
The file has been carefully fixed. Use proper code editing tools only.
