# Protected Files Manifest - DO NOT MODIFY WITHOUT AUTHORIZATION

## Critical System Files - NEVER MODIFY OR DELETE

### Core Server Files
- **`minimal_server.js`** - Main API server with critical fixes
  - Contains dashboard fixes that MUST NOT be reverted
  - Backup: `minimal_server.js.fixed-backup`
  - Verification: `verify-dashboard-fixes.sh`

### Trading System Core
- **`lib/autoLoop.js`** - Autonomous trading loop with parallel evaluation
- **`lib/PaperBroker.js`** - Paper trading engine with real price integration
- **`lib/brainIntegrator.js`** - Unified decision-making system
- **`config/tradingThresholds.js`** - Trading thresholds (currently in testing mode)
- **`services/enhancedPerformanceRecorder.js`** - Learning system with hot hand detection

### Critical Scripts
- **`start-benbot.sh`** - System startup script with verification
- **`verify-dashboard-fixes.sh`** - Verifies critical fixes are intact
- **`start-server.sh`** - Legacy startup script (use start-benbot.sh instead)

### Documentation - READ BEFORE ANY CHANGES
- **`CRITICAL_FIXES_DO_NOT_REVERT.md`** - Explains critical fixes
- **`AI_ASSISTANT_GUIDE.md`** - Guide for AI assistants
- **`CIRCUIT_BREAKER_RESET.md`** - Circuit breaker reset procedures
- **`STARTUP_GUIDE.md`** - User startup guide

## Files That Should NEVER Be Brought Back
These files contain old/broken code and should NEVER be used:
- Any file in `server-archive/` directory
- Any file ending with `.backup` (except `.fixed-backup`)
- Any file in `backup/` directory
- Files with timestamps like `server-backup-1758429442.js`

## Protected Patterns
These code patterns MUST remain in place:

### In minimal_server.js:
```javascript
// Strategy API fix - MUST use array iteration
(allStrategies || []).forEach(strategy => {

// Position data transformation - MUST include quantity/cost_basis
const transformedPositions = positions.map(pos => ({
  symbol: pos.symbol,
  quantity: pos.qty || 0,
  cost_basis: (pos.avg_price || 0) * (pos.qty || 0),

// Rate limiter fix - MUST have higher limits
'/api/quotes': new TokenBucketLimiter(20, 50),

// BrainIntegrator minConfidence - Testing mode
minConfidence: 0.25, // TESTING MODE: Lower confidence
```

### In lib/PaperBroker.js:
```javascript
// MUST use real quotes from getQuotesCache
const { getQuotesCache } = require('../minimal_server');

// MUST emit proper event structure
this.emit('orderFilled', {
  ...order,
  fillPrice: order.filled_price,
```

### In lib/autoLoop.js:
```javascript
// MUST use parallel evaluation
const [highValueCandidates, regularCandidates] = await Promise.all([

// MUST process in batches
for (let i = 0; i < testPromises.length; i += BATCH_SIZE) {
```

## Verification Commands
Run these before making ANY changes:
```bash
cd /Users/bendickinson/Desktop/benbot/live-api
./verify-dashboard-fixes.sh
./verify-system-integrity.sh
```

## Emergency Rollback
If something breaks:
```bash
cd /Users/bendickinson/Desktop/benbot/live-api
cp minimal_server.js.fixed-backup minimal_server.js
./start-benbot.sh
```

## WARNING TO AI ASSISTANTS
1. NEVER modify files listed here without explicit user permission
2. NEVER bring back old code from archive/backup directories
3. ALWAYS run verification scripts before and after changes
4. If unsure, ASK THE USER before proceeding
5. This system is in PRODUCTION - breaking changes affect real trading

Last Updated: September 29, 2025
Version: 2.0
