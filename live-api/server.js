const express = require('express');
const cors = require('cors');
const dayjs = require('dayjs');
const fs = require('fs');
const path = require('path');
const { nanoid } = require('nanoid');
const { CONFIG } = require('./lib/config');
const { noteRequest, setQuoteTouch, setBrokerTouch, currentHealth } = require('./lib/health');
const { wrapResponse, errorResponse } = require('./lib/watermark');
const { loadOrders, saveOrders, loadPositions, savePositions } = require('./lib/persistence');
const { placeOrderAdapter } = require('./lib/brokerPaper');
const { preTradeGate } = require('./lib/gate');
const { saveBundle, getBundle, replayBundle } = require('./lib/trace');
const { AutoRefresh } = require('./lib/autoRefresh');
const { AutoLoop } = require('./lib/autoLoop');

const { getQuotesCache, onQuotes, startQuotesLoop, stopQuotesLoop } = require('./dist/src/services/quotesService');
const { roster } = require('./dist/src/services/symbolRoster');

const app = express();
app.use(cors());
app.use(express.json());

// Initialize auto-refresh for quotes to keep health GREEN
const quoteRefresher = new AutoRefresh({
  interval: 30_000, // 30 seconds
  symbols: ['SPY', 'AAPL', 'QQQ'], // Common symbols
  enabled: true
});
quoteRefresher.start();

// Initialize auto-loop for paper orders (disabled by default)
const autoLoop = new AutoLoop({
  interval: parseInt(process.env.AUTOLOOP_INTERVAL_MS || '30000', 10),
  symbols: (process.env.AUTOLOOP_SYMBOLS || 'SPY').split(','),
  quantity: parseFloat(process.env.AUTOLOOP_QTY || '1'),
  enabled: process.env.AUTOLOOP_ENABLED === '1'
});
autoLoop.start();

app.use((req, res, next) => {
  res.set('x-api-origin', 'live-api:4000');
  next();
});

// Response watermark: add meta on objects, headers for all
app.use((req, res, next) => {
  const originalJson = res.json.bind(res);
  res.json = (body) => {
    const trace = nanoid();
    const asOf = new Date().toISOString();
    res.set('x-meta-trace', trace);
    res.set('x-meta-asof', asOf);
    res.set('x-meta-source', 'paper');
    res.set('x-meta-schema', 'v1');
    if (body && typeof body === 'object' && !Array.isArray(body)) {
      if (!body.meta) body.meta = { asOf, source: 'paper', schema_version: 'v1', trace_id: trace };
    }
    return originalJson(body);
  };
  next();
});

// Request outcome tracking for health/error budget
app.use((req, res, next) => {
  res.on('finish', () => {
    noteRequest(res.statusCode < 400);
  });
  next();
});

// In-memory safety state
let tradingMode = 'paper';
let emergencyStopActive = false;
const asOf = () => new Date().toISOString();

// Health & Metrics
app.get('/api/health', (req, res) => {
  const h = currentHealth();
  res.json({
    env: process.env.NODE_ENV || 'dev',
    gitSha: process.env.GIT_SHA || 'local-dev',
    region: 'local',
    services: { api: { status: h.ok ? 'up' : 'degraded', lastUpdated: asOf() } },
    ok: h.ok,
    breaker: h.breaker,
    quote_age_s: h.quote_age_s,
    broker_age_s: h.broker_age_s,
    slo_error_budget: h.slo_error_budget,
    asOf: asOf(),
  });
});
app.get('/health', (req, res) => {
  res.json(currentHealth());
});

app.get('/metrics', (req, res) => {
  res.type('application/json').send(
    JSON.stringify({
      ok: true,
      ts: asOf(),
      uptime: process.uptime(),
      benbot_data_fresh_seconds: 12,
      benbot_ws_connected: 1,
      totalSymbolsTracked: 123,
      errorRate: 0.012,
      requestsLastHour: 4200,
      averageLatency: 42,
    })
  );
});

app.get('/metrics/prom', (req, res) => {
  res
    .type('text/plain')
    .send(
      '# HELP app_up 1=up\n# TYPE app_up gauge\napp_up 1\n' +
        '# HELP app_uptime_seconds Uptime in seconds\n# TYPE app_uptime_seconds gauge\napp_uptime_seconds ' +
        process.uptime() +
        '\n# HELP benbot_data_fresh_seconds Seconds since last data\n# TYPE benbot_data_fresh_seconds gauge\nbenbot_data_fresh_seconds 12\n' +
        '# HELP benbot_ws_connected Connected flag\n# TYPE benbot_ws_connected gauge\nbenbot_ws_connected 1\n'
    );
});

// --- Minimal auth for UI: return access_token ---
app.post('/auth/token', (req, res) => {
  try {
    const { username, password } = req.body || {};
    const u = process.env.ADMIN_USERNAME || 'admin';
    const p = process.env.ADMIN_PASSWORD || 'changeme';
    if (username && password && (username !== u || password !== p)) {
      return res.status(401).json({ error: 'invalid_credentials' });
    }
    return res.json({ access_token: 'dev-token', token_type: 'bearer', expires_in: 60 * 60 * 8 });
  } catch {
    return res.json({ access_token: 'dev-token', token_type: 'bearer', expires_in: 60 * 60 * 8 });
  }
});

// Market Context & News
app.get('/api/context', (req, res) => {
  res.json({
    timestamp: asOf(),
    regime: { type: 'Neutral', confidence: 0.58, description: 'Mixed breadth, range-bound.' },
    volatility: { value: 17.2, change: -0.3, classification: 'Medium' },
    sentiment: { score: 0.52, sources: ['news', 'social'], trending_words: ['AI', 'earnings', 'CPI'] },
    features: { vix: 17.2, put_call: 0.92, adv_dec: 1.1 },
  });
});

// Volatility tile (standalone endpoint)
app.get('/api/context/volatility', (req, res) => {
  res.json({ value: 17.2, delta: -0.3, asOf: asOf() });
});

// Context sub-resources expected by frontend
app.get('/api/context/regime', (req, res) => {
  res.json({
    regime: 'Neutral',
    confidence: 0.58,
    asOf: asOf(),
    // extra fields tolerated by some clients
    since: asOf(),
    description: 'Range-bound conditions',
  });
});
app.get('/api/context/sentiment', (req, res) => {
  res.json({
    // minimal keys
    score: 0.52,
    label: 'Neutral',
    delta24h: 0.003,
    asOf: asOf(),
    // extended keys used elsewhere
    overall_score: 0.52,
    market_sentiment: 'neutral',
    positive_factors: ['Earnings beats'],
    negative_factors: ['Geopolitical risk'],
    source: 'synthetic',
    timestamp: asOf(),
  });
});
app.get('/api/context/sentiment/history', (req, res) => {
  const days = Math.min(Number(req.query.days || 30), 180);
  const now = dayjs();
  const arr = Array.from({ length: days }).map((_, i) => ({
    timestamp: now.subtract(days - 1 - i, 'day').toISOString(),
    score: 0.4 + (i % 7) * 0.01,
    sentiment: (i % 3 === 0 ? 'positive' : i % 3 === 1 ? 'neutral' : 'negative'),
    volume: 100 + i,
  }));
  // return both common shapes: array and {points:[{t,score}]}
  res.json({ points: arr.map(p => ({ t: p.timestamp, score: p.score })), items: arr });
});
app.get('/api/context/sentiment/anomalies', (req, res) => {
  const limit = Math.min(Number(req.query.limit || 10), 50);
  const items = Array.from({ length: limit }).map((_, i) => ({
    t: asOf(),
    z: 2.1,
    score: 0.64,
    source: 'news',
  }));
  res.json({ items });
});
app.get('/api/context/news', (req, res) => {
  const limit = Math.min(Number(req.query.limit || 10), 50);
  const now = dayjs();
  const items = Array.from({ length: limit }).map((_, i) => ({
    id: nanoid(),
    headline: `Context news ${i + 1}`,
    summary: 'Short summary',
    url: 'https://example.com',
    source: 'Wire',
    published_at: now.subtract(i, 'minute').toISOString(),
    ts: now.subtract(i, 'minute').toISOString(),
    sentiment_score: 0.5,
    sentiment: { score: 0.5, label: 'Neutral' },
    impact: ['high','medium','low'][i % 3],
    categories: ['markets'],
    symbols: ['AAPL','SPY'],
  }));
  res.json(items);
});

// News sentiment aggregator used by news.ts
app.get('/api/news/sentiment', (req, res) => {
  const category = String(req.query.category || 'markets');
  res.json({ category, outlets: { Reuters: { score: 0.18, count: 5 }, Bloomberg: { score: -0.05, count: 5 } }, clusters: [], asOf: asOf() });
});

app.get('/api/news', (req, res) => {
  const limit = Math.min(Number(req.query.limit || 10), 50);
  const now = dayjs();
  const items = Array.from({ length: limit }).map((_, i) => ({
    id: nanoid(),
    title: `Market headline ${i + 1}`,
    source: 'Wire',
    url: 'https://example.com/news',
    published_at: now.subtract(i, 'minute').toISOString(),
    sentiment_score: 0.5,
    symbols: ['SPY', 'AAPL', 'QQQ'],
    summary: 'Summary...',
  }));
  res.json(items);
});

// Strategies
const strategies = [
  {
    id: 'news_momo_v2',
    name: 'News Momentum v2',
    status: 'active',
    asset_class: 'stocks',
    priority_score: 0.72,
    performance: { win_rate: 0.56, sharpe_ratio: 1.21, max_drawdown: 0.11, trades_count: 124 },
    asOf: asOf(),
  },
  {
    id: 'mean_rev',
    name: 'Mean Reversion',
    status: 'idle',
    asset_class: 'stocks',
    priority_score: 0.41,
    performance: { win_rate: 0.48, sharpe_ratio: 0.77, max_drawdown: 0.15, trades_count: 88 },
    asOf: asOf(),
  },
];

app.get('/api/strategies', (req, res) => res.json({ asOf: asOf(), items: strategies.map(s => ({ ...s, asOf: asOf() })) }));
app.get('/api/strategies/active', (req, res) => res.json(strategies.filter((s) => s.status === 'active').map(s => ({ ...s, asOf: asOf() }))));

// Decisions & Trades
const makeDecision = () => ({
  id: nanoid(),
  symbol: 'AAPL',
  strategy_id: 'news_momo_v2',
  strategy_name: 'News Momentum v2',
  direction: 'buy',
  action: 'buy',
  score: 0.74,
  entry_price: 195.12,
  target_price: 201.0,
  stop_loss: 192.5,
  potential_profit_pct: 3.0,
  risk_reward_ratio: 2.1,
  confidence: 0.66,
  time_validity: 'EOD',
  timeframe: '5m',
  created_at: asOf(),
  timestamp: asOf(),
  status: 'pending',
  executed: false,
  reason: 'News breakout + MA cross',
  reasons: ['News +2.1σ', '20/50 MA ↑', 'Regime: Risk-On'],
  plan: { sizePct: 1.2, slPct: -2.5, tpPct: 4.0 },
  nextCheck: 'EOD',
  tags: ['breakout', 'news'],
  entry_conditions: ['MA20 > MA50', 'Volume spike'],
  indicators: [{ name: 'RSI', value: 62, signal: 'bullish' }],
});

app.get('/api/decisions', (req, res) => res.json([makeDecision(), makeDecision()]));
app.get('/api/decisions/latest', (req, res) => res.json([makeDecision()]));
app.get('/api/decisions/recent', (req, res) => res.json([makeDecision(), makeDecision()]));

// Trace endpoints
app.get('/trace/:id', (req, res) => {
  const rec = getBundle(req.params.id);
  if (!rec) return res.status(404).json({ error: 'not found' });
  res.json(rec);
});
app.post('/trace/:id/replay', (req, res) => {
  const out = replayBundle(req.params.id);
  if (!out.ok) return res.status(404).json({ error: 'not found' });
  res.json(out);
});

app.get('/api/trades', (req, res) => {
  const limit = Math.min(Number(req.query.limit || 20), 200);
  const items = (paperOrders.length ? paperOrders : []).slice(0, limit).map(o => ({
    id: o.id,
    symbol: o.symbol,
    side: o.side,
    qty: o.qty,
    price: o.price,
    status: o.status,
    ts: o.submittedAt,
  }));
  if (!items.length && CONFIG.FAIL_CLOSE && !CONFIG.FALLBACKS_ENABLED) {
    const err = errorResponse('STALE_DATA', 503);
    return res.status(err.status).json(err.body);
  }
  // If no orders yet and fallbacks permitted, synthesize from decisions
  const fallback = items.length ? items : strategies.slice(0, 1).map((s, i) => ({
    id: nanoid(), symbol: i % 2 ? 'SPY' : 'AAPL', side: i % 2 ? 'sell' : 'buy', qty: 10, price: 100, status: 'filled', ts: asOf(),
  }));
  res.json({ items: items.length ? items : fallback });
});

// Portfolio
function buildPortfolio(mode) {
  const equity = mode === 'live' ? 125000 : 50000;
  const cash = mode === 'live' ? 25000 : 20000;
  const daily_pl = mode === 'live' ? 420 : 180;
  const daily_pl_percent = +((daily_pl / equity) * 100).toFixed(2);
  return {
    summary: {
      total_equity: equity,
      cash_balance: cash,
      buying_power: cash * 2,
      daily_pl,
      daily_pl_percent,
      total_pl: 6800,
      total_pl_percent: 5.7,
      positions_count: 2,
      account: mode,
      last_updated: asOf(),
    },
    positions: [
      {
        symbol: 'AAPL',
        quantity: 50,
        avg_cost: 190.1,
        last_price: 195.3,
        current_value: 9765,
        unrealized_pl: 260,
        unrealized_pl_percent: 2.74,
        realized_pl: 120,
        account: mode,
        strategy_id: 'news_momo_v2',
        entry_time: asOf(),
      },
      {
        symbol: 'SPY',
        quantity: 20,
        avg_cost: 520.5,
        last_price: 525.2,
        current_value: 10504,
        unrealized_pl: 94,
        unrealized_pl_percent: 0.9,
        realized_pl: 0,
        account: mode,
        strategy_id: 'mean_rev',
        entry_time: asOf(),
      },
    ],
  };
}

app.get('/api/portfolio', (req, res) => {
  const mode = (req.query.mode || tradingMode).toString();
  res.json(buildPortfolio(mode));
});
app.get('/api/portfolio/paper', (req, res) => res.json(buildPortfolio('paper')));
app.get('/api/portfolio/live', (req, res) => res.json(buildPortfolio('live')));

// Paper trading order/positions endpoints (aliases for quick smoke tests)
let paperPositions = loadPositions();
let paperTrades = loadOrders();
let paperOrders = loadOrders();
function persistAll() {
  savePositions(paperPositions);
  saveOrders(paperOrders);
}
app.post('/api/paper/orders/dry-run', async (req, res) => {
  const idempotencyKey = req.get('Idempotency-Key') || '';
  const { symbol = 'AAPL', side = 'buy', qty = 1, type = 'market' } = req.body || {};
  const h = currentHealth();
  const quote = await getQuote(symbol);
  const paperAccount = { cash: 20000 }; // mock account fetch

  const gate = preTradeGate({
    nav: 50_000,
    portfolio_heat: 0.05,
    strategy_heat: 0.03,
    dd_mult: Math.max(CONFIG.DD_MIN_MULT, 1 - CONFIG.DD_FLOOR),
    requested_qty: qty,
    price: quote.price,
    available_cash: paperAccount.cash,
    quote_age_s: h.quote_age_s,
    broker_age_s: h.broker_age_s,
    stale: !h.ok,
  });
  return res.json({
    ok: true,
    symbol, side, qty, type,
    gate,
    idempotencyKey: idempotencyKey || null,
  });
});
app.post('/api/paper/orders', async (req, res) => {
  const idempotencyKey = req.get('Idempotency-Key') || '';
  const { symbol = 'AAPL', side = 'buy', qty = 1, type = 'market' } = req.body || {};
  const health = currentHealth();
  if (!health.ok && CONFIG.FAIL_CLOSE) {
    const err = errorResponse('BREAKER_RED', 503, { details: health });
    return res.status(err.status).json(err.body);
  }
  const quote = await getQuote(symbol);
  const paperAccount = { cash: 20000 }; // mock account fetch

  const gate = preTradeGate({
    nav: 50_000,
    portfolio_heat: 0.05,
    strategy_heat: 0.03,
    dd_mult: Math.max(CONFIG.DD_MIN_MULT, 1 - CONFIG.DD_FLOOR),
    requested_qty: qty,
    price: quote.price,
    available_cash: paperAccount.cash,
    quote_age_s: health.quote_age_s,
    broker_age_s: health.broker_age_s,
    stale: !health.ok,
  });
  if (gate.decision !== 'ACCEPT') {
    const err = errorResponse(gate.reason || 'REJECTED', 400, { gate });
    return res.status(err.status).json(err.body);
  }
  try {
    const price = quote.price; // Use the fetched quote price
    const broker = await placeOrderAdapter({ request: { symbol, side, qty: gate.routed_qty, type, price }, idempotencyKey });
    setBrokerTouch();
    // If order with this broker id already exists, return existing without mutating state
    const existingOrder = paperOrders.find(o => o.id === broker.id);
    if (existingOrder) {
      const existingTrade = paperTrades.find(t => t.id === broker.id);
      return res.json({ success: true, trade: existingTrade || existingOrder, gate, traceId: broker.id, meta: { asOf: asOf(), source: 'paper', schema_version: 'v1', trace_id: broker.id } });
    }
    const trade = {
      id: broker.id,
      status: broker.status,
      symbol,
      side,
      qty: broker.filledQty,
      type,
      submittedAt: asOf(),
      filledAvgPrice: broker.avgPrice,
      action: side,
      quantity: broker.filledQty,
      price: broker.avgPrice,
      total_value: broker.avgPrice * broker.filledQty,
      timestamp: asOf(),
      strategy_id: 'manual',
      strategy_name: 'Manual',
      account: 'paper',
    };
    paperTrades.unshift(trade);
    paperOrders.unshift({
      id: trade.id,
      symbol: trade.symbol,
      side: trade.side,
      qty: trade.qty,
      type: trade.type,
      price: trade.filledAvgPrice,
      status: trade.status,
      submittedAt: trade.submittedAt,
      filledAt: trade.timestamp,
    });
    // position update with avg cost & realized PnL
    const priceFill = broker.avgPrice;
    const idx = paperPositions.findIndex(p => p.symbol === symbol);
    if (idx >= 0) {
      const p = paperPositions[idx];
      let quantity = p.quantity;
      let avg_cost = p.avg_cost;
      let realized_pl = p.realized_pl || 0;
      if (side === 'buy') {
        const newQty = quantity + broker.filledQty;
        avg_cost = (quantity * avg_cost + broker.filledQty * priceFill) / (newQty || 1);
        quantity = newQty;
      } else {
        // sell: realize PnL on sold qty
        realized_pl += (priceFill - avg_cost) * broker.filledQty;
        quantity = quantity - broker.filledQty;
      }
      const last_price = priceFill;
      paperPositions[idx] = {
        ...p,
        quantity,
        avg_cost: +avg_cost.toFixed(4),
        last_price,
        current_value: quantity * last_price,
        unrealized_pl: (last_price - avg_cost) * quantity,
        unrealized_pl_percent: quantity ? ((last_price - avg_cost) / avg_cost) * 100 : 0,
        realized_pl,
      };
    } else if (side === 'buy') {
      paperPositions.push({ symbol, quantity: broker.filledQty, avg_cost: priceFill, last_price: priceFill, current_value: broker.filledQty * priceFill, unrealized_pl: 0, unrealized_pl_percent: 0, realized_pl: 0, account: 'paper', strategy_id: 'manual', entry_time: asOf() });
    }
    persistAll();
    // Save trace bundle
    const traceId = saveBundle({
      id: trade.id,
      order: { symbol, side, qty: broker.filledQty, type, price: priceFill },
      risk_config: {
        PER_TRADE_CAP: CONFIG.PER_TRADE_CAP,
        MAX_PORTFOLIO_HEAT: CONFIG.MAX_PORTFOLIO_HEAT,
        MAX_STRATEGY_HEAT: CONFIG.MAX_STRATEGY_HEAT,
        DD_FLOOR: CONFIG.DD_FLOOR,
        DD_MIN_MULT: CONFIG.DD_MIN_MULT,
      },
      context: {},
      quotes_snapshot: [],
      venue_priors: {},
    });
    return res.json({ success: true, trade, gate, traceId, meta: { asOf: asOf(), source: 'paper', schema_version: 'v1', trace_id: traceId } });
  } catch (e) {
    if (e && (e.code === 'RATE_LIMIT')) {
      const err = errorResponse('RATE_LIMIT', 429);
      return res.status(err.status).json(err.body);
    }
    const err = errorResponse('BROKER_DOWN', 502);
    return res.status(err.status).json(err.body);
  }
});
app.get('/api/paper/orders/:id', (req, res) => {
  const ord = paperOrders.find(o => o.id === req.params.id);
  if (!ord) return res.status(404).json({ error: 'not found' });
  res.json(ord);
});

// All orders
app.get('/api/paper/orders', (req, res) => {
  res.json({ items: paperOrders });
});

// Open orders (simple: those not filled/canceled)
app.get('/api/paper/orders/open', (req, res) => {
  const open = paperOrders.filter(o => !['filled', 'canceled', 'rejected'].includes(String(o.status || '').toLowerCase()));
  res.json(open.map(o => ({
    order_id: o.id,
    symbol: o.symbol,
    side: String(o.side || '').toUpperCase(),
    qty: o.qty,
    status: o.status,
    created_ts: Math.floor(new Date(o.submittedAt || asOf()).getTime() / 1000),
    limit_price: o.price,
  })));
});

// Decision explain stub
app.get('/api/decisions/:id/explain', (req, res) => {
  const id = req.params.id;
  res.json({
    id,
    summary: 'Signal driven by news momentum and trend confirmation.',
    factors: [
      { name: 'News z-score', value: 2.1, weight: 0.5 },
      { name: 'MA20 > MA50', value: 1, weight: 0.3 },
      { name: 'Regime bias', value: 0.2, weight: 0.2 },
    ],
    dataPoints: [
      { t: asOf(), feature: 'news_z', value: 2.1 },
      { t: asOf(), feature: 'ma_trend', value: 1 },
    ],
  });
});

// Logs endpoints (simple list)
app.get('/api/logs', (req, res) => {
  const level = String(req.query.level || 'INFO');
  const limit = Math.min(Number(req.query.limit || 100), 500);
  const items = Array.from({ length: limit }).map((_, i) => ({
    id: nanoid(),
    timestamp: asOf(),
    level,
    message: `Log ${i + 1}`,
    source: 'system',
  }));
  res.json({ items });
});
app.get('/api/events/logs', (req, res) => {
  const level = String(req.query.level || 'INFO');
  const limit = Math.min(Number(req.query.limit || 100), 500);
  const items = Array.from({ length: limit }).map((_, i) => ({
    id: nanoid(),
    timestamp: asOf(),
    level,
    message: `Event log ${i + 1}`,
    source: 'system',
  }));
  res.json({ items });
});

// Ingestion activity events (for UI ticker and data timeline)
app.get('/api/ingestion/events', (req, res) => {
  const limit = Math.min(Number(req.query.limit || 50), 500);
  const now = dayjs();
  const stages = ['INGEST', 'CONTEXT', 'CANDIDATES', 'GATES', 'PLAN', 'ROUTE', 'MANAGE', 'LEARN'];
  const symbols = ['AAPL', 'SPY', 'MSFT', 'NVDA', 'AMD', 'TSLA'];

  const items = Array.from({ length: limit }).map((_, i) => {
    const t = now.subtract(i * 3, 'second');
    const ts = t.valueOf();
    const stage = stages[i % stages.length];
    const symbol = symbols[i % symbols.length];
    const latency = 20 + Math.floor(Math.random() * 400);
    const ok = Math.random() > 0.08; // ~8% errors
    return {
      id: nanoid(),
      // fields used by ActivityTicker
      timestamp: t.toISOString(),
      stage,
      symbol,
      note: ok ? 'processed' : 'retrying',
      latency_ms: latency,
      status: ok ? 'ok' : 'err',
      trace_id: nanoid(),
      // fields used by DataIngestionCard
      ts,
      strategy_id: 'news_momo_v2',
      step: ok ? 'ok' : 'error',
      message: ok ? `Stage ${stage} completed` : `Stage ${stage} failed`,
      ms: latency,
      level: ok ? 'ok' : 'err',
    };
  });

  res.json(items);
});

// Backtests stubs
const backtests = {};
app.post('/api/backtests', (req, res) => {
  const id = nanoid();
  backtests[id] = {
    id,
    status: 'queued',
    submittedAt: asOf(),
    progressPct: 0,
  };
  res.json({ id, status: 'queued', submittedAt: backtests[id].submittedAt });
});
app.get('/api/backtests/:id', (req, res) => {
  const bt = backtests[req.params.id];
  if (!bt) return res.status(404).json({ error: 'not found' });
  // advance progress a bit
  bt.progressPct = Math.min(100, bt.progressPct + 10);
  bt.status = bt.progressPct >= 100 ? 'done' : 'running';
  if (bt.status === 'done' && !bt.finishedAt) bt.finishedAt = asOf();
  res.json(bt);
});
app.get('/api/backtests/:id/results', (req, res) => {
  const bt = backtests[req.params.id];
  if (!bt) return res.status(404).json({ error: 'not found' });
  res.json({
    summary: { cagr: 0.18, sharpe: 1.2, maxDD: 0.11, winRate: 0.54 },
    equityCurve: Array.from({ length: 50 }).map((_, i) => ({ t: dayjs().subtract(49 - i, 'day').toISOString(), eq: 100000 + i * 500 })),
    trades: Array.from({ length: 10 }).map((_, i) => ({ id: nanoid(), symbol: i % 2 ? 'AAPL' : 'SPY', pnl: (i - 5) * 25 })),
  });
});
app.get('/api/paper/positions', (req, res) => {
  // map to minimal UI shape while keeping fields
  const items = paperPositions.map(p => ({
    symbol: p.symbol,
    qty: p.quantity,
    avgPrice: p.avg_cost,
    marketPrice: p.last_price,
    unrealizedPnl: (p.last_price - p.avg_cost) * p.quantity,
    // preserve original keys as passthrough
    ...p,
  }));
  res.json(items);
});

// Portfolio history for paper (basic time series)
app.get('/api/portfolio/paper/history', (req, res) => {
  const now = dayjs();
  const points = Array.from({ length: 30 }).map((_, i) => ({
    t: now.subtract(29 - i, 'minute').toISOString(),
    equity: 50000 + i * 10,
    cash: 20000 - i * 2,
  }));
  res.json(points);
});

// Paper account summary
app.get('/api/paper/account', (req, res) => {
  const equity = 50000 + paperPositions.reduce((s, p) => s + (p.last_price * p.quantity), 0);
  const cash = 20000;
  res.json({ equity, cash, buyingPower: cash * 5, asOf: asOf() });
});

// --- Portfolio allocations (equity/options + top symbols)
app.get('/api/portfolio/allocations', (req, res) => {
  try {
    const positions = paperPositions || [];
    const totalMV = positions.reduce((s, p) => s + (Number(p.last_price) * Number(p.quantity)), 0);
    const cash = 20000;

    const byType = new Map();
    const bySymbol = new Map();

    for (const p of positions) {
      const mv = (Number(p.last_price) || 0) * (Number(p.quantity) || 0);
      const assetClass = 'equity'; // paper stub; extend for options later
      byType.set(assetClass, (byType.get(assetClass) || 0) + mv);
      bySymbol.set(p.symbol, (bySymbol.get(p.symbol) || 0) + mv);
    }

    const toList = (m) => Array.from(m.entries()).map(([name, value]) => ({
      name,
      value,
      pct: totalMV > 0 ? +(100 * (value / totalMV)).toFixed(2) : 0,
    }));

    const typeAlloc = toList(byType);
    const symbolAlloc = toList(bySymbol).sort((a,b)=>b.value-a.value).slice(0, 8);

    res.json({ data: {
      equity: 50000 + totalMV,
      cash,
      buying_power: cash * 5,
      pl_day: 180,
      totalMV,
      typeAlloc,
      symbolAlloc,
    }});
  } catch (e) {
    res.status(500).json({ error: 'allocations_failed', message: e?.message || 'unknown' });
  }
});

app.get('/api/autoloop/status', (req, res) => {
  res.json({
    enabled: autoLoop.enabled,
    isRunning: autoLoop.isRunning,
    status: autoLoop.status,
    lastRun: autoLoop.lastRun,
    interval: autoLoop.interval,
  });
});

// Safety
app.get('/api/safety/status', (req, res) => {
  const payload = {
    // legacy keys used by existing UI
    tradingMode,
    emergencyStopActive,
    circuitBreakers: { active: false },
    cooldowns: { active: false },
    // new compact keys for cards
    mode: tradingMode === 'live' ? 'LIVE' : 'PAPER',
    killSwitch: { status: emergencyStopActive ? 'ACTIVE' : 'READY', lastTriggeredAt: null },
    circuitBreaker: { status: 'NORMAL', thresholdPct: 5, windowMin: 60 },
    cooldown: { status: 'READY', activeUntil: null },
    asOf: asOf(),
  };
  res.json(payload);
});
app.post('/api/admin/reconcile', (req, res) => {
  // For this stub, reconciliation just reloads persisted data
  paperPositions = loadPositions();
  paperOrders = loadOrders();
  paperTrades = loadOrders();
  res.json({ ok: true, positions: paperPositions.length, orders: paperOrders.length });
});
app.post('/api/safety/emergency-stop', (req, res) => {
  emergencyStopActive = !!(req.body && req.body.active);
  res.json({
    success: true,
    message: emergencyStopActive ? 'Emergency stop activated' : 'Emergency stop deactivated',
  });
});
app.post('/api/safety/trading-mode', (req, res) => {
  const mode = req.body && req.body.mode ? String(req.body.mode) : '';
  if (mode === 'live' || mode === 'paper') {
    tradingMode = mode;
    return res.json({ success: true, message: 'Trading mode set to ' + mode });
  }
  res.status(400).json({ success: false, message: 'Invalid mode' });
});

// Data status
app.get('/api/data/status', (req, res) => {
  res.json({
    timestamp: asOf(),
    sources: [{ id: 'quotes', name: 'Quotes API', type: 'http', status: 'ok', lastUpdate: asOf(), healthScore: 0.98 }],
    metrics: {
      totalSymbolsTracked: 1200,
      activeSymbols: ['AAPL', 'SPY', 'QQQ'],
      symbolsWithErrors: [],
      requestsLastHour: 4500,
      dataPointsIngested: 200000,
      lastFullSyncCompleted: asOf(),
      averageLatency: 120,
      errorRate: 0.01,
    },
  });
});
app.get('/data/status', (req, res) => {
  res.json({
    timestamp: asOf(),
    sources: [{ id: 'quotes', name: 'Quotes API', type: 'http', status: 'ok', lastUpdate: asOf(), healthScore: 0.98 }],
    metrics: {
      totalSymbolsTracked: 1200,
      activeSymbols: ['AAPL', 'SPY', 'QQQ'],
      symbolsWithErrors: [],
      requestsLastHour: 4500,
      dataPointsIngested: 200000,
      lastFullSyncCompleted: asOf(),
      averageLatency: 120,
      errorRate: 0.01,
    },
  });
});

// --- WATCHLISTS & UNIVERSE ---
const DEFAULT_UNIVERSE = [
  "SPY","QQQ","IWM","DIA","AAPL","MSFT","AMZN","NVDA","META","GOOGL","TSLA","AMD","NFLX",
  "AVGO","CRM","COST","ORCL","INTC","ADBE","PEP","KO","MCD","JPM","BAC","WFC","GS",
  "XOM","CVX","COP","BP","PFE","MRNA","JNJ","UNH","LLY","ABBV","BA","CAT","GE","NKE","HD",
  "LOW","WMT","TGT","DIS","CMCSA","CSCO","QCOM","TXN","SHOP","SQ","PLTR","UBER","LYFT"
];

const WATCHLISTS = {
  default: ['SPY','QQQ','IWM','AAPL','NVDA'],
  small_caps_liquid: ['PLTR','SOFI','RIOT','MARA','HOOD','IONQ','U','PATH','FUBO','RBLX'],
  etfs_top: ['SPY','QQQ','IWM','XLF','SMH'],
  news_movers_today: [] // you can fill dynamically later
};

app.get('/api/watchlists', (req, res) => res.json(WATCHLISTS));

app.get('/api/universe', (req, res) => {
  const list = String(req.query.list || 'default');
  const symbols = WATCHLISTS[list] || WATCHLISTS.default;
  res.json({ symbols, asOf: new Date().toISOString() });
});

// Legacy support for individual watchlist lookup
app.get('/api/watchlists/:id', (req, res) => {
  const symbols = WATCHLISTS[req.params.id];
  if (symbols) {
    res.json({ id: req.params.id, symbols, asOf: new Date().toISOString() });
  } else {
    res.status(404).json({ error: 'Watchlist not found' });
  }
});

// --- PER-TICKER NEWS SENTIMENT (lightweight stub) ---
app.get('/api/news/ticker-sentiment', (req, res) => {
  const syms = String(req.query.symbols || '').split(',').filter(Boolean).slice(0, 100);
  const now = Date.now();

  const mk = (s) => {
    // TODO: replace with your real aggregation if available
    const base = (s.charCodeAt(0) % 7) / 10 - 0.3; // deterministic-ish
    const impact1h = +(base + (Math.random() - 0.5) * 0.2).toFixed(2);
    const impact24h = +(impact1h + (Math.random() - 0.5) * 0.15).toFixed(2);
    const count24h = Math.floor(3 + Math.random() * 12);
    const topOutlets = ['Reuters','Bloomberg','CNBC','WSJ'].slice(0, Math.max(1, Math.floor(Math.random()*4)));
    return { symbol: s, impact1h, impact24h, count24h, topOutlets, asOf: new Date(now).toISOString() };
  };

  res.json(syms.map(mk));
});

// Quotes & Bars
app.get('/api/quotes', (req, res) => {
  const symbols = String(req.query.symbols || '').split(',').filter(Boolean);
  const now = asOf();
  const out = symbols.length ? symbols : DEFAULT_UNIVERSE.slice(0, 10); // Use first 10 from universe
  const items = out.map((s) => {
    const basePrice = 50 + Math.random() * 200; // Random price between 50-250
    const change = (Math.random() * 6) - 3; // Random change between -3 and +3
    const prevClose = basePrice - change;
    return {
      symbol: s.toUpperCase(),
      last: +(basePrice).toFixed(2),
      bid: +(basePrice - 0.05).toFixed(2),
      ask: +(basePrice + 0.05).toFixed(2),
      prevClose: +prevClose.toFixed(2),
      change: +change.toFixed(2),
      pct: +((change / prevClose) * 100).toFixed(2),
      ts: now,
    };
  });
  setQuoteTouch();
  res.json(items);
});

app.get('/api/bars', (req, res) => {
  const symbol = String(req.query.symbol || 'AAPL');
  const timeframe = String(req.query.timeframe || '1Day');
  const limit = Math.min(Number(req.query.limit || 30), 500);
  const now = dayjs();
  const bars = Array.from({ length: limit }).map((_, i) => {
    const t = now.subtract(limit - 1 - i, 'day');
    const o = 100 + i,
      c = o + 0.6,
      h = c + 0.5,
      l = o - 0.5,
      v = 1_000_000 + i * 5000;
    return { t: t.toISOString(), o, h, l, c, v };
  });
  res.json({ symbol, timeframe, bars });
});

// --- SCANNER: candidates ranked by score ---
app.get('/api/scanner/candidates', async (req, res) => {
  const list = String(req.query.list || 'small_caps_liquid');
  const limit = Math.min(+(req.query.limit || 50), 100);
  const universe = WATCHLISTS[list] || WATCHLISTS.default;
  const symbols = universe.slice(0, limit);

  // Helpers: try real endpoints; fall back if missing
  async function getQuotes(syms) {
    try {
      const ax = await fetch(`http://localhost:4000/api/quotes?symbols=${encodeURIComponent(syms.join(','))}`);
      if (!ax.ok) throw new Error('quotes not ok');
      return await ax.json(); // expect [{symbol,last,prevClose,spreadPct,volume}]
    } catch {
      if (CONFIG.FAIL_CLOSE && !CONFIG.FALLBACKS_ENABLED) throw new Error('STALE_DATA');
      return syms.map(s => ({
        symbol: s,
        last: +(5 + Math.random()*30).toFixed(2),
        prevClose: +(5 + Math.random()*30).toFixed(2),
        spreadPct: +(0.2 + Math.random()*0.8).toFixed(2),
        volume: Math.floor(500_000 + Math.random()*5_000_000),
      }));
    }
  }

  async function getATR(symbol) {
    try {
      const r = await fetch(`http://localhost:4000/api/bars?symbol=${symbol}&timeframe=1Day&limit=20`);
      if (!r.ok) throw new Error('bars not ok');
      const { bars = [] } = await r.json();
      if (bars.length < 5) throw new Error('not enough bars');
      // simple ATR-ish
      const tr = bars.map(b => Math.abs(b.h - b.l));
      const atr = tr.reduce((a,b)=>a+b,0) / tr.length;
      return +atr.toFixed(2);
    } catch {
      if (CONFIG.FAIL_CLOSE && !CONFIG.FALLBACKS_ENABLED) throw new Error('STALE_DATA');
      return +(0.2 + Math.random()*1.0).toFixed(2);
    }
  }

  async function getTickerSentiment(syms) {
    try {
      const r = await fetch(`http://localhost:4000/api/news/ticker-sentiment?symbols=${encodeURIComponent(syms.join(','))}`);
      if (!r.ok) throw new Error('sent not ok');
      const arr = await r.json();
      const map = new Map(arr.map(x => [x.symbol, x]));
      return (s) => map.get(s) || { impact1h: 0, impact24h: 0, count24h: 0, topOutlets: [] };
    } catch {
      if (CONFIG.FAIL_CLOSE && !CONFIG.FALLBACKS_ENABLED) throw new Error('STALE_DATA');
      return (_s) => ({ impact1h: 0, impact24h: 0, count24h: 0, topOutlets: [] });
    }
  }

  try {
    const quotes = await getQuotes(symbols);
    const sentOf = await getTickerSentiment(symbols);

  // Compute features & score
  const rows = await Promise.all(quotes.map(async q => {
    const last = Number(q.last) || 0;
    const prev = Number(q.prevClose) || last || 1;
    const gapPct = (last - prev) / (prev || 1);
    const spreadPct = Number(q.spreadPct) || 0.5;
    const rvol = +(1 + Math.random()*2.5).toFixed(2); // placeholder unless you have real RVOL
    const atr = await getATR(q.symbol);
    const s = sentOf(q.symbol);

    // Scoring (you can tune)
    const z = (x, m=0, sd=1) => (x - m) / (sd || 1);
    const score =
      0.40 * z(s.impact1h, 0, 0.25) +
      0.20 * z(rvol, 1.5, 0.6) +
      0.15 * z(gapPct, 0, 0.04) * Math.sign(s.impact1h || 0) +
      0.15 * z(Math.random()*0.05, 0.02, 0.02) - // momentum proxy if you lack intraday
      0.05 * z(spreadPct, 0.5, 0.3) -
      0.10 * (last < 1.5 ? 1 : 0); // microcap penalty

    const conf = 1 / (1 + Math.exp(-(0.7*z(s.impact1h,0,0.25) + 0.5*z(rvol,1.5,0.6) + 0.3*z(Math.abs(gapPct),0,0.04) - 0.3*z(spreadPct,0.5,0.3))));
    const side = score >= 0 ? "buy" : "sell";

    const plan = score >= 0
      ? { entry: last, stop: +(last - 2*atr).toFixed(2), take: +(last * 1.025).toFixed(2), type: 'long_catalyst' }
      : { entry: last, stop: +(last + 2*atr).toFixed(2), take: +(last * 0.975).toFixed(2), type: 'short_fade' };

    const explain = {
      impact1h: s.impact1h, impact24h: s.impact24h, count24h: s.count24h,
      rvol, gapPct: +gapPct.toFixed(4), spreadPct, atr,
      outlets: s.topOutlets
    };

    const risk = {
      suggestedQty: Math.max(1, Math.floor((0.01 * 50_000) / (atr || 0.2))), // ~1% of 50k by ATR risk
      spreadOK: spreadPct <= 1.0, liquidityOK: q.volume >= 2_000_000
    };

    return {
      symbol: q.symbol, last, score: +score.toFixed(3), confidence: +conf.toFixed(2), side,
      plan, risk, explain, asOf: new Date().toISOString()
    };
  }));

    rows.sort((a,b) => b.score - a.score);
    res.json(rows.slice(0, limit));
  } catch (e) {
    const err = errorResponse('STALE_DATA', 503);
    return res.status(err.status).json(err.body);
  }
});

app.get('/api/evotester/history', (req, res) => {
  console.log('[API] GET /api/evotester/history - returning empty array');
  res.json([]);
});

// Alerts & Logs
app.get('/api/alerts', (req, res) => {
  const limit = Math.min(Number(req.query.limit || 20), 100);
  const sevParam = String(req.query.severity || '').toLowerCase();
  const sevLabels = ['info', 'warning', 'critical'];
  const items = Array.from({ length: limit }).map((_, i) => ({
    id: nanoid(),
    timestamp: asOf(),
    severity: sevLabels[i % sevLabels.length],
    source: 'system',
    message: `Alert ${i + 1}`,
    acknowledged: false,
  }));
  const filtered = sevLabels.includes(sevParam)
    ? items.filter((a) => a.severity === sevParam)
    : items; // ignore numeric/unknown values
  res.json(filtered);
});

app.post('/api/alerts/:id/acknowledge', (req, res) => {
  res.json({ status: 'ok', message: `Alert ${req.params.id} acknowledged` });
});

const PORT = Number(process.env.PORT) || 4000;
// Migration guard: refuse to start if pending migrations are detected
try {
  const migFlag = path.resolve(__dirname, 'migrations/pending.flag');
  if (fs.existsSync(migFlag) && !CONFIG.ALLOW_UNSAFE_MIGRATIONS) {
    console.error('Pending migrations detected. Refusing to start. Set ALLOW_UNSAFE_MIGRATIONS=true to override.');
    process.exit(1);
  }
} catch {}
const server = app.listen(PORT, () => {
  console.log(`live-api listening on http://localhost:${PORT}`);
  // Optional GREEN-gated paper autoloop (env toggle)
  if (process.env.AUTOLOOP_ENABLED === '1' || process.env.AUTOLOOP_ENABLED === 'true') {
    try {
      require('./lib/autoLoop').start();
    } catch (e) {
      console.error('[autoLoop] failed to start', e?.message || e);
    }
  }
});

// WebSocket Support using ws library
const WebSocket = require('ws');
const { attachHeartbeat } = require('./lib/heartbeat');

// Create ws servers in noServer mode; we will route in a single upgrade handler
const wss = new WebSocket.Server({ noServer: true });
const wssDecisions = new WebSocket.Server({ noServer: true });
const wssPrices = new WebSocket.Server({ noServer: true });

// Handle WebSocket connections
wss.on('connection', (ws, request) => {
  console.log('WebSocket connection established');
});

// WebSocket Server for decisions endpoint
wssDecisions.on('connection', (ws) => {
  console.log('WebSocket connection established for decisions');
});

// WS for prices
wssPrices.on('connection', (ws) => {
  try {
    const { quotes, asOf } = getQuotesCache();
    ws.send(JSON.stringify({ type: 'prices', data: quotes, time: asOf }));
    
    ws.on('message', (raw) => {
      try {
        const msg = JSON.parse(raw.toString());
        if (msg?.type === 'subscribe' && Array.isArray(msg.symbols)) {
          const ttl = Number(msg.ttlSec || process.env.SUBSCRIPTION_TTL_SEC || 120);
          roster.subscribe(msg.symbols, ttl);
          ws.send(JSON.stringify({ type: 'subscribed', symbols: msg.symbols, ttlSec: ttl }));
        }
      } catch (e) {
        console.error('Error handling price subscription:', e);
      }
    });
  } catch (e) {
    console.error('Error in price WS connection:', e);
  }
});

if (CONFIG.PRICES_WS_ENABLED) {
  // Forward quote updates to connected clients
  onQuotes(({ quotes, time }) => {
    const payload = JSON.stringify({ type: 'prices', data: quotes, time });
    wssPrices.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        try { client.send(payload); } catch {}
      }
    });
  });
  
  // Start the quotes loop
  startQuotesLoop();
}

// Broadcast helper for decisions stream
function broadcastDecision(obj) {
  wssDecisions.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      try { client.send(JSON.stringify(obj)); } catch {}
    }
  });
}

module.exports.broadcastDecision = broadcastDecision;

// Add heartbeat to all WebSocket servers to detect and clean up dead connections
attachHeartbeat(wss);
attachHeartbeat(wssDecisions);
attachHeartbeat(wssPrices);

// Store WebSocket servers in app.locals for status endpoint
app.locals.wss = wss;
app.locals.wssDecisions = wssDecisions;
app.locals.wssPrices = wssPrices;

// Live status endpoint
app.use('/api/live', require('./routes/live'));

// Quotes API routes
app.use('/api/quotes', require('./dist/routes/quotes'));

// Handle WebSocket upgrades
server.on('upgrade', (request, socket, head) => {
  try {
    const { pathname } = new URL(request.url, `http://${request.headers.host}`);

    if (pathname === '/ws/decisions') {
      wssDecisions.handleUpgrade(request, socket, head, (ws) => {
        wssDecisions.emit('connection', ws, request);
      });
    } else if (pathname === '/ws/prices') {
      wssPrices.handleUpgrade(request, socket, head, (ws) => {
        wssPrices.emit('connection', ws, request);
      });
    } else if (pathname === '/ws') {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
      });
    } else {
      socket.destroy();
    }
  } catch (err) {
    console.error('WebSocket upgrade error:', err);
    socket.destroy();
  }
});
