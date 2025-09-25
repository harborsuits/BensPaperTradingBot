# BenBot Trading Principles & Shared Rules

## Core Principles (All Components MUST Follow)

### 1. ❌ NEVER Trust Internal State
- **Strategies MUST NOT track their own positions** (no `lastSignal`, `hasPosition`, etc.)
- **Always query actual positions** from PaperBroker/Tradier
- **The broker is the single source of truth** for positions

### 2. 🛡️ Defense-in-Depth Validation
Every trade must pass ALL validation layers:
1. **Position Validation** - Can't sell what we don't have
2. **Circuit Breaker** - System protection gates
3. **Capital Tracker** - Sufficient funds check
4. **Data Validator** - Fresh, valid market data
5. **Expected Value** - Positive EV after costs
6. **Risk Limits** - Position size, daily loss, etc.

### 3. 📊 Data Freshness Requirements
- **Quotes**: Max 5 seconds old
- **Bars**: Max 5 minutes old
- **News**: Max 1 hour old
- **NO TRADING on stale data** (sandbox override for testing only)

### 4. 💰 Risk Management Rules
- **Max Position Size**: 10% of portfolio
- **Max Daily Loss**: 2% of account
- **Min Cash Reserve**: 5% buffer
- **Stop Loss Required**: All positions need exit plan

### 5. 🔄 State Synchronization
- **Positions refreshed** before every decision
- **Capital updated** every 5 seconds
- **Circuit breaker** checks continuously
- **No cached state** beyond TTL limits

### 6. 🚦 Order Flow Rules
```
Signal → Brain Score → Validation → Risk Check → Circuit Breaker → Submit
         ↓ FAIL        ↓ FAIL       ↓ FAIL       ↓ FAIL
         REJECT        REJECT       REJECT       REJECT
```

### 7. 🎯 Strategy Requirements
All strategies MUST:
- Return **signals**, not execute trades
- Include **confidence** scores
- Provide **clear reasons** for trades
- Respect **position limits**
- Use **shared validation**

### 8. 📝 Audit Trail
Every decision must include:
- Timestamp
- Strategy source
- Confidence score
- Risk assessment
- Validation results
- Rejection reasons (if any)

## Implementation Checklist

### For Strategy Developers:
```javascript
// ❌ WRONG - Don't track state
this.lastSignal = 'BUY';
if (this.lastSignal === 'BUY') { sell(); }

// ✅ RIGHT - Check actual positions
const position = paperBroker.getPositions().find(p => p.symbol === symbol);
if (position && position.qty > 0) { /* can sell */ }
```

### For Order Validation:
```javascript
// All orders MUST pass through /api/paper/orders which enforces:
1. Position validation (can't sell non-existent positions)
2. Circuit breaker status
3. Capital availability
4. Data freshness
5. Expected value check
```

### For Risk Management:
```javascript
// Use existing systems:
- capitalTracker.getAvailableCapital()
- circuitBreaker.canTrade()
- dataValidator.validatePreTrade()
- positionSizer.calculatePosition()
```

## Monitoring & Alerts

### Health Checks:
- Circuit breaker status
- Capital utilization
- Position count & exposure
- Data freshness
- Strategy performance

### Alert Conditions:
- Circuit breaker tripped
- Daily loss > 1.5%
- Capital utilization > 80%
- Stale data detected
- Multiple validation failures

## Evolution Rules

### Strategy Promotion (per ai_policy.yaml):
- **R1 → R2**: 40+ trades, Sharpe > 0.9, PF > 1.15
- **R2 → R3**: Sharpe > 1.1, PF > 1.25, DD < 7%
- **R3 → Live**: Sharpe > 1.2, PF > 1.3, DD < 6%

### Demotion Triggers:
- Sharpe < 0.8 over 10 days
- Drawdown > 6%
- 3+ breaches in 24h
- Slippage > 8bps over model

## Emergency Procedures

### Circuit Breaker Trips When:
- 5+ failures in 1 minute
- Daily loss > 2%
- API errors > 3 consecutive
- Data staleness > 30 seconds

### Recovery Process:
1. Wait cooldown period (5 min)
2. Verify data freshness
3. Check account status
4. Resume with reduced position sizes
5. Monitor closely for 30 minutes

## Testing Guidelines

### Before Going Live:
1. Verify all validation layers active
2. Test circuit breaker triggers
3. Confirm position sync works
4. Check capital tracking accuracy
5. Validate data freshness checks

### Continuous Monitoring:
- Log all validation failures
- Track rejection reasons
- Monitor strategy state sync
- Alert on unusual patterns
