const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Health endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Opportunities endpoint
app.get('/api/opportunities', (req, res) => {
  const opportunities = [
    {
      id: '1',
      symbol: 'SPY',
      kind: 'etf',
      timestamp: Date.now() / 1000,
      reason_tags: ['tariff', 'semiconductors'],
      novelty_score: 0.85,
      sentiment_z: -2.1,
      iv_change_1d: 0.12,
      regime: 'risk_off',
      regime_alignment: 'bearish',
      fees_bps: 6,
      slip_bps: 3,
      spread_bps: 24,
      spread_cap_bps: 30,
      cluster_heat_delta: 0.01,
      plan_strategy: 'put_debit_spread',
      plan_risk_usd: 25,
      plan_target_r: 2,
      plan_horizon: '1–2 days',
      meta_prob: 0.61,
      meta_threshold: 0.62,
      decision: 'PROBE',
    },
    {
      id: '2',
      symbol: 'NVDA',
      kind: 'equity',
      timestamp: Date.now() / 1000,
      reason_tags: ['earnings', 'ai_demand'],
      novelty_score: 0.72,
      sentiment_z: 1.8,
      iv_change_1d: 0.25,
      regime: 'mixed',
      regime_alignment: 'bullish',
      fees_bps: 8,
      slip_bps: 5,
      spread_bps: 18,
      spread_cap_bps: 30,
      cluster_heat_delta: 0.015,
      plan_strategy: 'call_debit_spread',
      plan_risk_usd: 35,
      plan_target_r: 1.8,
      plan_horizon: '3–5 days',
      meta_prob: 0.68,
      meta_threshold: 0.62,
      decision: 'PASS',
    },
    {
      id: '3',
      symbol: 'SOFI',
      kind: 'equity',
      timestamp: Date.now() / 1000,
      reason_tags: ['fintech', 'user_growth'],
      novelty_score: 0.35,
      sentiment_z: 0.9,
      iv_change_1d: -0.05,
      regime: 'risk_on',
      regime_alignment: 'neutral',
      fees_bps: 12,
      slip_bps: 15,
      spread_bps: 92,
      spread_cap_bps: 90,
      cluster_heat_delta: 0.008,
      meta_prob: 0.58,
      meta_threshold: 0.62,
      decision: 'SKIP',
      skip_codes: ['SPREAD_TOO_WIDE', 'META_PROB_LOW'],
    },
  ];
  
  res.json(opportunities);
});

// Handle probe and paper order actions
app.post('/api/opportunities/:id/probe', (req, res) => {
  res.json({ success: true, message: `Started probe for opportunity ${req.params.id}` });
});

app.post('/api/opportunities/:id/paper-order', (req, res) => {
  res.json({ success: true, message: `Submitted paper order for opportunity ${req.params.id}` });
});

const PORT = 4001;
app.listen(PORT, () => {
  console.log(`Opportunities server running on http://localhost:${PORT}`);
});
