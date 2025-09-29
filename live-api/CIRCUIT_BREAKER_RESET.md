# Circuit Breaker Reset Guide

## Quick Reset Commands

If the AutoLoop circuit breaker blocks trading, run these commands:

```bash
# 1. Stop the AutoLoop
curl -X POST http://localhost:4000/api/autoloop/stop

# 2. Wait 2 seconds
sleep 2

# 3. Start the AutoLoop (this resets its circuit breaker)
curl -X POST http://localhost:4000/api/autoloop/start

# 4. Activate all strategies
curl -X POST http://localhost:4000/api/strategies/activate-all
```

## Understanding Circuit Breakers

Your system has TWO circuit breakers:
1. **Main Circuit Breaker** - Protects the entire system
2. **AutoLoop Circuit Breaker** - Protects the automated trading loop

## Common Triggers
- Failed orders
- API errors exceeding limits (3 in 1 minute)
- Order failures exceeding limits (5 in 1 minute)
- Drawdown exceeding 5%
- Daily loss exceeding 2%

## Circuit Breaker States
- **CLOSED**: Normal operation (trading allowed)
- **OPEN**: Blocked (no trading, cooling down)
- **HALF-OPEN**: Testing recovery (limited trading)

## Recovery Time
- Cooldown period: 5 minutes
- After cooldown, it enters HALF-OPEN state
- If test trades succeed, returns to CLOSED
- If test trades fail, returns to OPEN
