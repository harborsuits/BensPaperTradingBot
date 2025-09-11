# Micro-Live Rollout Plan
## Production-Ready Scaling with Maximum Safety

### 🎯 Objective
Gradually introduce live trading with **\$5-25 net-debit trades** while maintaining **16/16 green proof validation** and **immediate emergency shutdown** capability.

---

## 📅 Phase 1: Day 1-2 (Single Bot, Maximum Caution)

### Safety Gates (All Must Pass)
```javascript
const MICRO_LIVE_GATES = {
  soak_test_passed: true,           // 5+ days 16/16 green required
  single_bot_only: true,            // Only 1 bot active
  options_only: true,               // No crypto/equity yet
  net_debit_max: 25,                // $25 max per trade
  equity_subcap: 0.01,              // 1% of total equity max
  friction_p90_max: 0.06,           // 6% max friction
  kill_switch_tested: true,         // E2E kill switch verified
  reconciliation_active: true       // Nightly reconciliation running
};
```

### Trade Parameters
- **Max Position Size**: $25 net debit
- **Max Portfolio Exposure**: 1% of equity
- **Entry Criteria**:
  - Pre-trade proof: ✅ GREEN (fail-closed gating enforced)
  - NBBO freshness: ≥95%
  - Greeks headroom: ≥10% buffer
  - Friction estimate: ≤6%

### Monitoring & Alerts
```javascript
// Real-time monitoring during live trading
const LIVE_MONITORING = {
  proof_validation: 'continuous',    // Every trade validated
  pnl_tracking: 'real-time',         // Mark-to-market every 5min
  greeks_monitoring: 'continuous',   // Headroom alerts
  kill_switch: 'hot',                // 1-click emergency stop
  reconciliation: 'nightly',         // Position verification
  alerts: {
    critical: ['proof_failure', 'greeks_violation', 'pnl_threshold'],
    warning: ['friction_spike', 'nbbo_stale'],
    info: ['trade_execution', 'pnl_update']
  }
};
```

### Daily Runbook (Live Trading)
```bash
# Pre-Market (08:00 ET)
curl localhost:4000/api/observability/health | jq '.overall_status'
# Expected: "healthy"

# Market Open (09:30 ET)
# Monitor live trades
watch -n 30 'curl -s localhost:4000/api/competition/ledger | jq ".[] | select(.status==\"active\") | {id: .allocId, pnl: .totalPnL}"'

# Midday Check (12:00 ET)
curl -s "localhost:4000/api/proofs/fills?since=-2h" | jq '.summary.passed'
# Expected: >= 95%

# Emergency Stop (Anytime)
npm run kill-switch  # Instant shutdown
```

---

## 📅 Phase 2: Day 3-5 (Two Bots, Controlled Scaling)

### Expanded Safety Gates
```javascript
const PHASE2_GATES = {
  ...MICRO_LIVE_GATES,
  two_bots_max: true,               // Allow up to 2 bots
  friction_p95_max: 0.08,           // 8% max friction (95th percentile)
  position_correlation_max: 0.75,   // Max correlation between bots
  drawdown_daily_max: 0.005,        // 0.5% max daily drawdown
  trade_frequency_max: 3            // Max 3 trades per hour per bot
};
```

### Risk Management Enhancements
- **Correlation Guard**: Bots cannot hold >75% correlated positions
- **Drawdown Limits**: Daily 0.5% max, intraday 0.2% max
- **Trade Frequency**: Max 3 trades/hour/bot to prevent overtrading
- **Position Sizing**: Dynamic based on volatility and correlation

### Advanced Monitoring
```javascript
const ADVANCED_MONITORING = {
  correlation_tracking: 'real-time',     // Monitor position correlations
  volatility_adjustment: 'dynamic',      // Adjust sizes based on VIX
  drawdown_tracking: 'continuous',       // Real-time P&L monitoring
  trade_frequency: 'per-bot-per-hour',   // Rate limiting
  cross_bot_hedging: 'allowed'           // Bots can hedge each other
};
```

---

## 🚨 Emergency Protocols

### Immediate Shutdown Triggers
```javascript
const EMERGENCY_TRIGGERS = {
  proof_failure_rate: 0.05,        // 5% proof failure rate
  greeks_headroom_min: 0.05,       // 5% headroom minimum
  drawdown_intraday: 0.002,        // 0.2% intraday drawdown
  nbbo_freshness_min: 0.90,        // 90% NBBO freshness minimum
  reconciliation_drift: 0.001,     // 0.1% reconciliation drift
  manual_kill_switch: true         // Always available
};
```

### Automated Safety Responses
```javascript
const SAFETY_AUTOMATION = {
  // Reduce position sizes
  high_volatility: 'size_reduction_50%',
  correlation_spike: 'hedge_positions',
  drawdown_threshold: 'pause_trading',

  // Emergency actions
  critical_alert: 'pause_all_bots',
  proof_failure: 'cancel_pending_orders',
  system_degraded: 'reduce_trade_frequency'
};
```

---

## 📊 Success Metrics

### Phase 1 Success Criteria (Day 1-2)
- ✅ **16/16 green proofs** maintained during live trading
- ✅ **Zero emergency shutdowns** required
- ✅ **PnL within expectations** (±2% of backtest)
- ✅ **All safety gates** functioning correctly
- ✅ **Reconciliation clean** (≤0.05% drift)

### Phase 2 Success Criteria (Day 3-5)
- ✅ **Two-bot correlation** maintained <75%
- ✅ **Drawdown limits** never breached
- ✅ **Trade frequency** within limits
- ✅ **Cross-bot efficiency** demonstrated
- ✅ **All Phase 1 criteria** maintained

---

## 🔧 Implementation Checklist

### Pre-Rollout (Complete Before Day 1)
- [x] **5-day soak test passed** (16/16 green)
- [x] **Kill switch E2E tested** (during market hours)
- [x] **Rate limit handling verified** (no stale NBBO routing)
- [x] **Greeks headroom alerts** implemented
- [x] **Reconciliation monitoring** active
- [x] **Clock discipline** enforced
- [x] **WORM audit trail** tested

### Day 1 Setup
- [ ] **Single bot configuration** prepared
- [ ] **$25 max trade size** enforced
- [ ] **1% equity sub-cap** configured
- [ ] **Real-time monitoring** dashboards active
- [ ] **Emergency contacts** on standby

### Live Trading Verification
- [ ] **Pre-trade proofs** all green
- [ ] **NBBO freshness** ≥95%
- [ ] **Greeks headroom** ≥10%
- [ ] **Friction estimates** ≤6%
- [ ] **Kill switch** tested and ready

---

## 🎯 Go/No-Go Decision Framework

### Go Criteria (All Must Pass)
```javascript
const GO_DECISION = {
  soak_test: '5_days_16_16_green',
  kill_switch: 'e2e_tested_market_hours',
  monitoring: 'all_dashboards_green',
  reconciliation: 'nightly_clean',
  emergency_protocols: 'tested_and_documented',
  team_readiness: '24_7_coverage'
};
```

### No-Go Triggers
- Any safety system test failure
- Reconciliation drift >0.1%
- Proof failure rate >5%
- Team not ready for 24/7 monitoring

---

## 📈 Scaling Path Forward

### Post-Phase 2 Success
- **Week 2**: Increase to $50 max trade size
- **Week 3**: Add equity trading capability
- **Week 4**: Increase to 3-4 bots
- **Month 2**: Add crypto trading
- **Month 3**: Full production scaling

### Key Milestones
1. **Phase 2 Complete**: 5 days successful micro-live
2. **Size Increase**: $50 max trades validated
3. **Asset Expansion**: Options + Equity validated
4. **Bot Scaling**: 4-bot portfolio validated
5. **Full Production**: All safety systems proven

---

## 🛡️ Risk Mitigation

### Technical Safeguards
- **Circuit Breakers**: Automatic shutdown on threshold breaches
- **Position Limits**: Hard caps on exposure
- **Rate Limiting**: Prevent overtrading
- **Manual Override**: Kill switch always available

### Operational Safeguards
- **24/7 Monitoring**: Team coverage during market hours
- **Gradual Scaling**: Never more than 1% equity at risk
- **Conservative Sizing**: Always smaller than backtest positions
- **Quick Exit**: All positions liquid and hedgeable

---

## 🚀 Execution Command

When ready to start micro-live:

```bash
# Final pre-flight check
npm run safety-check
npm run real-data-status
curl localhost:4000/api/observability/health

# Start micro-live trading
export MICRO_LIVE_ENABLED=true
export MAX_NET_DEBIT=25
export MAX_EQUITY_SUBCAP=0.01

# Monitor continuously
watch -n 30 'curl -s localhost:4000/api/competition/ledger'
```

**This plan ensures maximum safety while proving live trading capability with real capital.** 🎯✨
