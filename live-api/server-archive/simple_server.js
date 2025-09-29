const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 4000;

app.use(cors());
app.use(express.json());

// Basic health endpoint
app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    timestamp: new Date().toISOString(),
    server: 'simple-dashboard-api'
  });
});

// Autopilot status endpoint
app.get('/api/audit/autoloop/status', (req, res) => {
  res.json({
    mode: process.env.AUTOLOOP_MODE || 'discovery',
    running: true,
    tick_ms: 30000,
    timestamp: new Date().toISOString()
  });
});

// Portfolio endpoint
app.get('/api/paper/account', (req, res) => {
  res.json({
    equity: 10000.50,
    cash: 2500.75,
    buying_power: 12500.25,
    day_pnl: 45.30,
    open_pnl: 120.80,
    timestamp: new Date().toISOString()
  });
});

// Strategies endpoint
app.get('/api/strategies', (req, res) => {
  const limit = parseInt(req.query.limit) || 10;
  const strategies = [
    {
      id: 'news_momo_v2',
      name: 'News Momentum V2',
      status: 'active',
      win_rate: 0.68,
      sharpe_ratio: 1.24,
      trades_count: 245
    },
    {
      id: 'mean_rev',
      name: 'Mean Reversion',
      status: 'paper',
      win_rate: 0.55,
      sharpe_ratio: 0.78,
      trades_count: 189
    }
  ];

  res.json(strategies.slice(0, limit));
});

// Orders endpoint
app.get('/api/paper/orders', (req, res) => {
  const limit = parseInt(req.query.limit) || 10;
  const orders = [
    {
      id: 'ord_001',
      symbol: 'SPY',
      side: 'BUY',
      qty: 10,
      price: 443.12,
      status: 'FILLED',
      strategy_id: 'news_momo_v2',
      timestamp: new Date().toISOString()
    },
    {
      id: 'ord_002',
      symbol: 'QQQ',
      side: 'SELL',
      qty: 5,
      price: 398.45,
      status: 'PENDING',
      strategy_id: 'mean_rev',
      timestamp: new Date().toISOString()
    },
    {
      id: 'ord_003',
      symbol: 'AAPL',
      side: 'BUY',
      qty: 8,
      price: 175.30,
      status: 'PARTIAL',
      strategy_id: 'news_momo_v2',
      timestamp: new Date().toISOString()
    }
  ];

  res.json(orders.slice(0, limit));
});

// Brain flow summary endpoint
app.get('/api/brain/flow/summary', (req, res) => {
  const window = req.query.window || '15m';
  res.json({
    window,
    counts: {
      ingest_ok: 87,
      context_ok: 85,
      candidates_ok: 82,
      gates_passed: 41,
      gates_failed: 46,
      plan_ok: 2,
      route_ok: 1,
      manage_ok: 1,
      learn_ok: 1
    },
    by_mode: {
      discovery: 100,
      shadow: 0,
      live: 0
    },
    latency_ms: {
      p50: 120,
      p95: 340
    },
    timestamp: new Date().toISOString()
  });
});

// Decisions summary endpoint
app.get('/api/decisions/summary', (req, res) => {
  const window = req.query.window || '15m';
  res.json({
    window,
    proposals_per_min: 4.2,
    unique_symbols: 7,
    last_ts: new Date().toISOString(),
    by_stage: {
      proposed: 15,
      intent: 2,
      executed: 8
    },
    timestamp: new Date().toISOString()
  });
});

// Brain status endpoint
app.get('/api/brain/status', (req, res) => {
  res.json({
    mode: process.env.AUTOLOOP_MODE || 'discovery',
    running: true,
    tick_ms: 30000,
    breaker: null,
    recent_pf_after_costs: 1.05,
    sharpe_30d: 0.42,
    sharpe_90d: 0.38,
    timestamp: new Date().toISOString()
  });
});

// Evo status endpoint
app.get('/api/evo/status', (req, res) => {
  res.json({
    generation: 15,
    population: 200,
    best: {
      config_id: 'cfg_abc123',
      metrics: {
        pf_after_costs: 1.18,
        sharpe: 0.42,
        trades: 640
      }
    },
    running: true,
    timestamp: new Date().toISOString()
  });
});

// Decisions recent endpoint
app.get('/api/decisions/recent', (req, res) => {
  const stage = req.query.stage || 'proposed';
  const limit = parseInt(req.query.limit) || 50;

  let decisions = [];

  if (stage === 'proposed') {
    decisions = [
      {
        id: `dec_${Date.now()}`,
        ts: new Date().toISOString(),
        symbol: 'SPY',
        strategy_id: 'news_momo_v2',
        confidence: 0.71,
        reason: 'news impulse + momo filter'
      },
      {
        id: `dec_${Date.now() + 1}`,
        ts: new Date().toISOString(),
        symbol: 'QQQ',
        strategy_id: 'mean_rev',
        confidence: 0.65,
        reason: 'mean reversion signal'
      }
    ];
  } else if (stage === 'intent') {
    decisions = [
      {
        id: `ti_${Date.now()}`,
        ts: new Date().toISOString(),
        symbol: 'SPY',
        side: 'BUY',
        qty: 5,
        limit: 443.12,
        strategy_id: 'news_momo_v2',
        ev_after_costs: 0.0032
      }
    ];
  } else if (stage === 'executed') {
    decisions = [
      {
        id: `exec_${Date.now()}`,
        ts: new Date().toISOString(),
        symbol: 'SPY',
        side: 'BUY',
        qty: 5,
        price: 443.12,
        status: 'FILLED',
        strategy_id: 'news_momo_v2'
      }
    ];
  }

  res.json(decisions.slice(0, limit));
});

// Brain flow recent endpoint
app.get('/api/brain/flow/recent', (req, res) => {
  const limit = parseInt(req.query.limit) || 100;
  const symbol = req.query.symbol;

  const flowData = [
    {
      symbol: 'SPY',
      ts: new Date().toISOString(),
      stages: {
        ingest: { ok: true, quote_age_s: 1.2 },
        context: { ok: true, vol_rank: 0.42, atr: 3.1 },
        candidates: { ok: true, count: 4, winner: { strategy_id: 'news_momo_v2', confidence: 0.71 } },
        gates: { ok: true, passed: ['hours_ok', 'spread_ok'], rejected: [] },
        plan: { ok: false, reason: 'discovery_mode' },
        route: { ok: false, skipped: true },
        manage: { ok: false, skipped: true },
        learn: { ok: false, skipped: true }
      },
      mode: 'discovery',
      trace_id: `trace_${Date.now()}`
    },
    {
      symbol: 'QQQ',
      ts: new Date().toISOString(),
      stages: {
        ingest: { ok: true, quote_age_s: 0.8 },
        context: { ok: true, vol_rank: 0.35, atr: 2.8 },
        candidates: { ok: true, count: 3, winner: { strategy_id: 'mean_rev', confidence: 0.68 } },
        gates: { ok: true, passed: ['hours_ok', 'spread_ok'], rejected: [] },
        plan: { ok: false, reason: 'discovery_mode' },
        route: { ok: false, skipped: true },
        manage: { ok: false, skipped: true },
        learn: { ok: false, skipped: true }
      },
      mode: 'discovery',
      trace_id: `trace_${Date.now() + 1}`
    }
  ];

  const filteredData = symbol ? flowData.filter(item => item.symbol === symbol) : flowData;
  res.json(filteredData.slice(0, limit));
});

// Brain scoring activity endpoint
app.get('/api/brain/scoring/activity', (req, res) => {
  const symbol = req.query.symbol || 'SPY';

  res.json({
    symbol,
    ts: new Date().toISOString(),
    candidates: [
      {
        strategy_id: 'news_momo_v2',
        raw_score: 0.82,
        ev_after_costs: 0.0032,
        reliability: 0.78,
        liquidity: 0.93,
        total: 0.82 * 0.78 * 0.93,
        selected: true,
        reason: 'momo+news; spread 6bps; PF 1.14 (90d); COST_OK'
      },
      {
        strategy_id: 'mean_rev',
        raw_score: 0.49,
        ev_after_costs: -0.0004,
        reliability: 0.55,
        liquidity: 0.98,
        total: -0.00021,
        selected: false,
        reason: 'ev<=0 after costs (blocked)'
      }
    ],
    weights: { ev: 1.0, reliability: 1.0, liquidity: 1.0 },
    trace_id: `trace_${Date.now()}`
  });
});

// Evo candidates endpoint
app.get('/api/evo/candidates', (req, res) => {
  const limit = parseInt(req.query.limit) || 20;

  const candidates = [
    {
      config_id: 'cfg_abc123',
      strategy_id: 'news_momo_v2',
      params: { lookback: 25, z_entry: 1.8 },
      backtest: {
        pf_after_costs: 1.18,
        sharpe: 0.42,
        trades: 640,
        regimes: { volatile: 1.14, quiet: 1.07 }
      },
      ready_for_paper: true
    },
    {
      config_id: 'cfg_def456',
      strategy_id: 'mean_rev',
      params: { threshold: 2.1, hold_period: 5 },
      backtest: {
        pf_after_costs: 1.05,
        sharpe: 0.28,
        trades: 320,
        regimes: { volatile: 0.95, quiet: 1.12 }
      },
      ready_for_paper: true
    }
  ];

  res.json(candidates.slice(0, limit));
});

// Evo schedule paper validate endpoint
app.post('/api/evo/schedule-paper-validate', (req, res) => {
  const { config_id, days = 14 } = req.body;

  if (!config_id) {
    return res.status(400).json({ error: 'config_id is required' });
  }

  const trackingId = `val_${Date.now()}`;

  res.json({
    success: true,
    tracking_id: trackingId,
    config_id,
    days,
    message: `Paper validation scheduled for ${days} days`,
    timestamp: new Date().toISOString()
  });
});

// Paper positions endpoint
app.get('/api/paper/positions', (req, res) => {
  const positions = [
    {
      symbol: 'SPY',
      qty: 10,
      avg_price: 442.50,
      current_price: 445.20,
      unrealized_pnl: 27.00,
      unrealized_pnl_pct: 0.61
    },
    {
      symbol: 'QQQ',
      qty: -5,
      avg_price: 398.00,
      current_price: 396.80,
      unrealized_pnl: 6.00,
      unrealized_pnl_pct: 0.30
    }
  ];

  res.json(positions);
});

app.listen(PORT, () => {
  console.log(`Simple dashboard API server listening on http://localhost:${PORT}`);
});
