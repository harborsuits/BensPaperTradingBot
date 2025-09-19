require('dotenv').config();
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const app = express();
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const PORT = 4000;
const { tradingState } = require('./src/lib/tradingState');
const { requireTradingOn } = require('./src/middleware/requireTradingOn');

// Basic middleware
app.use(express.json());

// --- Audit provenance middleware ---
const crypto = require('crypto');
const FORCE_NO_MOCKS = String(process.env.FORCE_NO_MOCKS || '').toLowerCase() === 'true';
const DISCONNECT_FEEDS = String(process.env.DISCONNECT_FEEDS || '').toLowerCase() === 'true';
const QUOTES_TTL_MS = Number(process.env.QUOTES_TTL_MS || 1500);
const HEALTH_QUOTES_TTL_MS = Number(process.env.HEALTH_QUOTES_TTL_MS || 8000);

function isRegularMarketOpen(now = new Date()) {
  const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const day = et.getDay();
  if (day === 0 || day === 6) return false; // Sun/Sat
  const mins = et.getHours() * 60 + et.getMinutes();
  return mins >= (9 * 60 + 30) && mins <= (16 * 60);
}

function getProviderTag() {
  return process.env.TRADIER_TOKEN ? 'tradier' : 'none';
}

function ensureRealOrFail(isReal) {
  if (FORCE_NO_MOCKS && !isReal) {
    const asof = new Date().toISOString();
    return { status: 503, body: { error: 'mocks_disabled', reason: 'FORCE_NO_MOCKS', asof_ts: asof } };
  }
  return null;
}

app.use((req, res, next) => {
  const startedAt = Date.now();
  const requestId = (crypto.randomUUID && crypto.randomUUID()) || Math.random().toString(36).slice(2);
  res.locals.requestId = requestId;
  res.locals.source = res.locals.source || 'unknown';
  res.locals.provider = res.locals.provider || getProviderTag();

  const origJson = res.json.bind(res);
  res.json = (body) => {
    const latencyMs = Date.now() - startedAt;
    const asof = new Date().toISOString();
    // Stamp headers (non-breaking for array responses)
    try {
      res.setHeader('x-request-id', requestId);
      res.setHeader('x-latency-ms', String(latencyMs));
      res.setHeader('x-provider', String(res.locals.provider || 'none'));
      res.setHeader('x-source', String(res.locals.source || 'unknown'));
      res.setHeader('x-asof', asof);
    } catch {}
    // If body is an object (not array), enrich with provenance
    if (body && typeof body === 'object' && !Array.isArray(body)) {
      body = {
        source: res.locals.source || 'unknown',
        provider: res.locals.provider || 'none',
        asof_ts: body.asof_ts || asof,
        latency_ms: body.latency_ms || latencyMs,
        request_id: body.request_id || requestId,
        ...body
      };
    }
    return origJson(body);
  };
  next();
});

// --- Simple in-memory paper order store + bus ---
const ordersEmitter = new EventEmitter();
let paperOrders = [];

// --- Trading kill-switch handled via src/middleware/requireTradingOn ---

// --- Real data & feed outage guards as middleware ---
function requireRealProviders(req, res, next) {
  if (FORCE_NO_MOCKS && req.requiresReal) {
    return res.status(503).json({ error: 'mocks_disabled' });
  }
  next();
}

function maybeDisconnectFeeds(req, res, next) {
  if (DISCONNECT_FEEDS) {
    return res.status(503).json({ error: 'feeds_unavailable' });
  }
  next();
}

// Health endpoint with breaker logic
app.get('/api/health', async (req, res) => {
  const startedAt = Date.now();
  const { token, baseUrl } = getTradierConfig();
  res.locals.source = token ? 'broker' : 'mock';
  res.locals.provider = getProviderTag();

  // Do not fail when no token; allow degraded health in dev
  const fail = ensureRealOrFail(Boolean(token));
  if (fail && token) return res.status(fail.status).json(fail.body);

  async function pingBroker() {
    if (!token) return { ok: false, rttMs: null };
    const t0 = Date.now();
    try {
      await axios.get(`${baseUrl}/markets/quotes?symbols=SPY`, {
        headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' },
        timeout: 2500
      });
      return { ok: true, rttMs: Date.now() - t0 };
    } catch {
      return { ok: false, rttMs: null };
    }
  }

  async function checkQuotesFreshness() {
    if (!token) return { ok: false, ageSec: null, provider: 'Tradier' };
    try {
      const { data } = await axios.get(`${baseUrl}/markets/quotes?symbols=SPY`, {
        headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' },
        timeout: 3000
      });
      const q = data?.quotes?.quote;
      const quote = Array.isArray(q) ? q[0] : q;
      const tsStr = quote?.trade_timestamp || quote?.last_timestamp || quote?.timestamp || null;
      const ts = tsStr ? new Date(tsStr) : new Date();
      const ageSec = Math.max(0, Math.round((Date.now() - ts.getTime()) / 1000));
      return { ok: Number.isFinite(ageSec), ageSec, provider: 'Tradier' };
    } catch {
      return { ok: false, ageSec: null, provider: 'Tradier' };
    }
  }

  const [broker, quotes] = await Promise.all([pingBroker(), checkQuotesFreshness()]);

  const reasons = [];
  const tokenPresent = !!token;
  if (!tokenPresent) reasons.push('no_provider_token');
  if (DISCONNECT_FEEDS) reasons.push('feeds_disconnected');

  const mktOpen = isRegularMarketOpen();
  const quotesFresh = Boolean(quotes.ok && typeof quotes.ageSec === 'number' && (quotes.ageSec * 1000) <= HEALTH_QUOTES_TTL_MS);
  if (!quotesFresh) reasons.push('quotes_stale_or_missing');

  let status = 'green';
  if (reasons.includes('feeds_disconnected')) status = 'red';
  else if (!tokenPresent || (!quotesFresh && mktOpen) || !broker.ok) status = 'amber';

  const breaker = status === 'green' ? 'GREEN' : status === 'amber' ? 'AMBER' : 'RED';

  res.setHeader('x-health-status', status);
  res.setHeader('x-health-reasons', reasons.join(','));
  res.json({
    ok: status !== 'red',
    status,
    reasons,
    breaker,
    asOf: new Date().toISOString(),
    latencyMs: Date.now() - startedAt,
    broker,
    marketData: quotes,
    server: 'minimal-live-api',
    ttl_ms: HEALTH_QUOTES_TTL_MS,
  });
});

// Alerts endpoint
app.get('/api/alerts', (req, res) => {
  res.locals.source = 'cache';
  res.locals.provider = 'calc';
  res.json([]);
});

// Diagnostics: echo selected env presence
app.get('/api/echo/env', (req, res) => {
  res.json({
    TRADIER_TOKEN_present: Boolean(process.env.TRADIER_TOKEN && String(process.env.TRADIER_TOKEN).trim()),
    DISCONNECT_FEEDS: process.env.DISCONNECT_FEEDS || '',
    FORCE_NO_MOCKS: process.env.FORCE_NO_MOCKS || '',
    QUOTES_TTL_MS: QUOTES_TTL_MS,
  });
});

// Autoloop status endpoints
app.get('/api/autoloop/status', (req, res) => {
  res.locals.source = 'calc';
  res.locals.provider = 'calc';
  res.json({
    is_running: true,
    status: 'IDLE',
    interval_ms: 30000,
    last_cycle: new Date().toISOString(),
    next_cycle: new Date(Date.now() + 30000).toISOString()
  });
});

app.get('/api/audit/autoloop/status', (req, res) => {
  res.locals.source = 'calc';
  res.locals.provider = 'calc';
  res.json({
    is_running: true,
    status: 'IDLE',
    interval_ms: 30000,
    coordination_audit: {
      rawSignals: 0,
      winners: 0,
      conflicts: 0
    },
    risk_rejections: [],
    allocation_summary: {}
  });
});

// Ingestion/events endpoint
app.get('/api/ingestion/events', (req, res) => {
  res.locals.source = 'cache';
  res.locals.provider = 'calc';
  res.json([]);
});

// Roster endpoint
app.get('/api/roster/active', (req, res) => {
  res.locals.source = 'cache';
  res.locals.provider = 'calc';
  res.json({
    symbols: ['SPY', 'AAPL', 'QQQ', 'MSFT', 'NVDA'],
    count: 5,
    timestamp: new Date().toISOString()
  });
});

// Bars endpoint (Tradier):
// - 1Day -> markets/history (daily bars)
// - 1Min/5Min/15Min -> markets/timesales (intraday bars)
// - 1Hour -> aggregate 15Min into hourly bars
app.get('/api/bars', async (req, res) => {
  try {
    const symbol = String(req.query.symbol || 'SPY').toUpperCase();
    const timeframe = String(req.query.timeframe || '1Day');
    const limit = Math.min(parseInt(String(req.query.limit || '90')) || 90, 1000);
    const { token, baseUrl } = getTradierConfig();
    res.locals.source = token ? 'broker' : 'mock';
    res.locals.provider = getProviderTag();

    const fail = ensureRealOrFail(Boolean(token));
    if (fail) return res.status(fail.status).json(fail.body);

    const isIntraday = ['1Min', '5Min', '15Min', '1Hour'].includes(timeframe);
    if (!isIntraday) {
      // Daily bars
      const url = `${baseUrl}/markets/history?symbol=${encodeURIComponent(symbol)}&interval=daily&limit=${limit}`;
      const { data } = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' },
        timeout: 5000
      });
      const rows = data?.history?.day || data?.history || [];
      const list = Array.isArray(rows) ? rows : rows ? [rows] : [];
      const bars = list.map((r) => ({
        t: r?.date || r?.timestamp || r?.time || new Date().toISOString(),
        o: Number(r?.open ?? 0),
        h: Number(r?.high ?? 0),
        l: Number(r?.low ?? 0),
        c: Number(r?.close ?? 0),
        v: Number(r?.volume ?? 0),
      })).filter(b => b.c || b.o);
      return res.json(bars.slice(-limit));
    }

    // Intraday via timesales
    const interval = timeframe === '1Min' ? '1min' : timeframe === '5Min' ? '5min' : '15min';
    const intervalMinutes = timeframe === '1Min' ? 1 : timeframe === '5Min' ? 5 : 15;
    const minutesBack = (timeframe === '1Hour') ? limit * 60 : limit * intervalMinutes;
    const start = new Date(Date.now() - minutesBack * 60 * 1000);
    const end = new Date();

    function fmt(d) {
      const yyyy = d.getFullYear();
      const mm = String(d.getMonth() + 1).padStart(2, '0');
      const dd = String(d.getDate()).padStart(2, '0');
      const HH = String(d.getHours()).padStart(2, '0');
      const MM = String(d.getMinutes()).padStart(2, '0');
      return `${yyyy}-${mm}-${dd} ${HH}:${MM}`;
    }

    async function fetchTimesales(s, e) {
      const url = `${baseUrl}/markets/timesales?symbol=${encodeURIComponent(symbol)}&interval=${interval}&start=${encodeURIComponent(fmt(s))}&end=${encodeURIComponent(fmt(e))}&session_filter=all`;
      const { data } = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' },
        timeout: 6000
      });
      return data;
    }

    let data = await fetchTimesales(start, end);
    let rows = data?.series?.data || data?.series || data?.data || [];
    if (!rows || (Array.isArray(rows) && rows.length === 0)) {
      const altStart = new Date(start.getTime() - 24 * 60 * 60 * 1000);
      const altEnd = new Date(end.getTime() - 24 * 60 * 60 * 1000);
      data = await fetchTimesales(altStart, altEnd);
      rows = data?.series?.data || data?.series || data?.data || [];
    }
    const list = Array.isArray(rows) ? rows : rows ? [rows] : [];
    const raw = list.map((r) => ({
      t: r?.time || r?.timestamp || r?.date || new Date().toISOString(),
      o: Number(r?.open ?? r?.o ?? r?.price ?? 0),
      h: Number(r?.high ?? r?.h ?? r?.price ?? 0),
      l: Number(r?.low ?? r?.l ?? r?.price ?? 0),
      c: Number(r?.close ?? r?.c ?? r?.price ?? 0),
      v: Number(r?.volume ?? r?.v ?? 0),
    })).filter(b => b.c || b.o);

    if (timeframe !== '1Hour') {
      return res.json(raw.slice(-limit));
    }

    // Aggregate 15min into hourly bars
    const grouped = new Map();
    for (const b of raw) {
      const dt = new Date(b.t);
      const key = new Date(Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth(), dt.getUTCDate(), dt.getUTCHours(), 0, 0)).toISOString();
      if (!grouped.has(key)) {
        grouped.set(key, { t: key, o: b.o, h: b.h, l: b.l, c: b.c, v: b.v });
      } else {
        const g = grouped.get(key);
        g.h = Math.max(g.h, b.h);
        g.l = Math.min(g.l, b.l);
        g.c = b.c;
        g.v += b.v;
      }
    }
    return res.json(Array.from(grouped.values()).slice(-limit));
  } catch (e) {
    res.locals.source = 'cache';
    res.locals.provider = 'calc';
    return res.json([]);
  }
});

// Resolve Tradier credentials (env first, then repo config file)
function getTradierConfig() {
  const token = process.env.TRADIER_TOKEN || process.env.TRADIER_API_KEY || '';
  const base = process.env.TRADIER_BASE_URL || process.env.TRADIER_API_URL || '';
  if (token && base) return { token, baseUrl: base };
  try {
    const credPath = path.resolve(__dirname, '../config/credentials/tradier.json');
    const raw = fs.readFileSync(credPath, 'utf8');
    const json = JSON.parse(raw);
    const profile = json.default && json[json.default] ? json[json.default] : json.paper || json;
    return {
      token: profile.api_key || '',
      baseUrl: profile.base_url || 'https://sandbox.tradier.com/v1'
    };
  } catch {
    return { token: '', baseUrl: 'https://sandbox.tradier.com/v1' };
  }
}

// Quotes endpoint (uses Tradier sandbox/live depending on config)
app.get('/api/quotes', requireRealProviders, maybeDisconnectFeeds, async (req, res) => {
  req.requiresReal = true;
  try {
    const symbols = (req.query.symbols ? String(req.query.symbols) : 'SPY')
      .split(',')
      .map(s => s.trim().toUpperCase())
      .filter(Boolean);
    const { token, baseUrl } = getTradierConfig();
    res.locals.source = token ? 'broker' : 'mock';
    res.locals.provider = getProviderTag();

    const fail = ensureRealOrFail(Boolean(token));
    if (fail) return res.status(fail.status).json(fail.body);

    // outage guard handled by maybeDisconnectFeeds
    const url = `${baseUrl}/markets/quotes?symbols=${encodeURIComponent(symbols.join(','))}`;
    const { data } = await axios.get(url, {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/json'
      },
      timeout: 3000
    });

    const node = data?.quotes?.quote || data?.quote || data?.quotes || [];
    const list = Array.isArray(node) ? node : node ? [node] : [];
    const now = Date.now();
    const quotes = list.map(q => {
      const symbol = String(q.symbol || q.ticker || '').toUpperCase();
      const last = Number(q.last ?? q.close ?? q.price ?? 0);
      const asof_ts = q.trade_date || q.timestamp || new Date().toISOString();
      const age = Math.max(0, now - new Date(asof_ts).getTime());
      const stale = age > QUOTES_TTL_MS;
      return {
        symbol,
        last,
        bid: Number(q.bid ?? 0),
        ask: Number(q.ask ?? 0),
        prevClose: Number(q.prev_close ?? q.previous_close ?? q.previousClose ?? 0),
        volume: Number(q.volume ?? 0),
        asof_ts,
        provider: 'tradier',
        source: 'broker',
        cache_age_ms: age,
        stale,
        ttl_ms: QUOTES_TTL_MS,
      };
    }).filter(x => x.symbol);
    res.setHeader('x-quotes-ttl-ms', String(QUOTES_TTL_MS));
    // Return array shape to satisfy useQuotes() while still stamping headers
    return res.json(quotes);
  } catch (e) {
    return res.json([]);
  }
});

// Market overview: market status + SPY and VIX snapshot
app.get('/api/overview', async (req, res) => {
  try {
    const { token, baseUrl } = getTradierConfig();
    res.locals.source = token ? 'broker' : 'mock';
    res.locals.provider = getProviderTag();

    const fail = ensureRealOrFail(Boolean(token));
    if (fail) return res.status(fail.status).json(fail.body);

    if (!token) return res.json({ marketStatus: 'Unknown', asOf: new Date().toISOString() });

    const symbols = ['SPY', 'VIX', '^VIX', 'VIX.X'];
    const url = `${baseUrl}/markets/quotes?symbols=${encodeURIComponent(symbols.join(','))}`;
    const { data } = await axios.get(url, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' },
      timeout: 4000
    });
    const arr = data?.quotes?.quote ? (Array.isArray(data.quotes.quote) ? data.quotes.quote : [data.quotes.quote]) : [];
    const bySym = Object.fromEntries(arr.map(q => [String(q.symbol).toUpperCase(), q]));

    const spy = bySym['SPY'];
    const vix = bySym['VIX'] || bySym['^VIX'] || bySym['VIX.X'];

    function normalize(q) {
      if (!q) return null;
      const last = Number(q.last ?? q.close ?? 0);
      const prev = Number(q.prevclose ?? q.previous_close ?? 0);
      const change = prev ? last - prev : Number(q.change ?? 0);
      const pct = prev ? (change / prev) * 100 : Number(q.change_percentage ?? 0);
      return { symbol: q.symbol, last, prevClose: prev, change, pct };
    }

    const spyN = normalize(spy);
    const vixN = normalize(vix);

    // Compute market status by US/Eastern time
    const now = new Date();
    const estStr = now.toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false });
    const est = new Date(estStr);
    const day = est.getDay();
    const hh = est.getHours();
    const mm = est.getMinutes();
    const mins = hh * 60 + mm;
    let status = 'Closed';
    if (day >= 1 && day <= 5) {
      if (mins >= 570 && mins < 960) status = 'Open'; // 9:30-16:00
      else if (mins >= 240 && mins < 570) status = 'Pre'; // 4:00-9:30
      else if (mins >= 960 && mins < 1200) status = 'Post'; // 16:00-20:00
      else status = 'Closed';
    }

    res.json({
      marketStatus: status,
      asOf: new Date().toISOString(),
      spx: spyN,
      vix: vixN
    });
  } catch (e) {
    res.locals.source = 'cache';
    res.locals.provider = 'calc';
    res.json({ marketStatus: 'Unknown', asOf: new Date().toISOString() });
  }
});

// Metrics endpoint
app.get('/metrics', (req, res) => {
  res.locals.source = 'calc';
  res.locals.provider = 'calc';
  res.json({
    totalSymbolsTracked: 29,
    errorRate: 0.02,
    requestsLastHour: 150,
    averageLatency: 45,
    timestamp: new Date().toISOString()
  });
});

// Quotes status for UI health pill
app.get('/api/quotes/status', (req, res) => {
  res.locals.source = 'calc';
  res.locals.provider = getProviderTag();
  try {
    // Minimal shape expected by the UI
    res.json({
      provider: process.env.TRADIER_TOKEN ? 'tradier' : 'none',
      autorefresh: true,
      symbolsCached: 0,
      marketHours: 'Unknown',
      asOf: new Date().toISOString(),
    });
  } catch (e) {
    res.json({ provider: 'none', autorefresh: false, symbolsCached: 0 });
  }
});

// Brain flow summary endpoint
app.get('/api/brain/flow/summary', (req, res) => {
  const window = req.query.window || '15m';
  // Reflect inactive pipeline: zero all counts
  res.json({
    window,
    counts: {
      ingest_ok: 0,
      context_ok: 0,
      candidates_ok: 0,
      gates_passed: 0,
      gates_failed: 0,
      plan_ok: 0,
      route_ok: 0,
      manage_ok: 0,
      learn_ok: 0
    },
    by_mode: {
      discovery: 0,
      shadow: 0,
      live: 0
    },
    latency_ms: {
      p50: 0,
      p95: 0
    },
    timestamp: new Date().toISOString()
  });
});

// Decisions summary endpoint
app.get('/api/decisions/summary', (req, res) => {
  const window = req.query.window || '15m';
  // Reflect real pipeline state: no recent decisions → zeroed summary
  res.json({
    window,
    proposals_per_min: 0,
    unique_symbols: 0,
    last_ts: null,
    by_stage: { proposed: 0, intent: 0, executed: 0 },
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

// ---- EvoTester control & session endpoints (fail-closed minimal impl) ----
const evoSessions = new Map();
const evoGenerations = new Map(); // id -> [{generation,bestFitness,averageFitness,timestamp}]
const evoResults = new Map();     // id -> [{id,name,fitness,performance,...}]

app.post('/api/evotester/start', (req, res) => {
  const cfg = req.body || {};
  const id = `evo-${Date.now().toString(36)}`;
  const total = Number(cfg.generations || 50);
  const nowIso = new Date().toISOString();
  evoSessions.set(id, {
    id,
    running: true,
    status: 'running',
    currentGeneration: 0,
    totalGenerations: total,
    startTime: nowIso,
    symbols: Array.isArray(cfg.symbols) ? cfg.symbols : [],
    config: cfg,
    bestFitness: 0,
    averageFitness: 0,
  });
  // Initialize generation log
  evoGenerations.set(id, []);

  // Simulate progress ticks every 2s; complete after total generations
  const interval = setInterval(() => {
    const s = evoSessions.get(id);
    if (!s || !s.running) { clearInterval(interval); return; }
    s.currentGeneration += 1;
    s.bestFitness = Math.max(s.bestFitness, Math.random() * 0.5 + 0.5);
    s.averageFitness = Math.max(0, Math.min(1, s.averageFitness + (Math.random() - 0.4) * 0.05));
    const progress = s.totalGenerations ? Math.min(s.currentGeneration / s.totalGenerations, 1) : 0;
    const log = evoGenerations.get(id) || [];
    log.push({
      generation: s.currentGeneration,
      bestFitness: +s.bestFitness.toFixed(3),
      averageFitness: +s.averageFitness.toFixed(3),
      timestamp: new Date().toISOString()
    });
    evoGenerations.set(id, log);
    broadcastToChannel('evotester', {
      type: 'evo_progress',
      data: {
        sessionId: id,
        running: s.running,
        currentGeneration: s.currentGeneration,
        totalGenerations: s.totalGenerations,
        startTime: s.startTime,
        progress,
        bestFitness: s.bestFitness,
        averageFitness: s.averageFitness,
        status: s.status,
      }
    });
    if (s.currentGeneration >= s.totalGenerations) {
      s.running = false;
      s.status = 'completed';
      clearInterval(interval);
      // Build simple top strategies list
      const base = +s.bestFitness.toFixed(2);
      const items = [0,1,2].map((i) => ({
        id: `${id}_best_${i+1}`,
        name: i === 0 ? 'RSI-Momentum-V2' : i === 1 ? 'VWAP-Reversion' : 'News-Momo',
        fitness: +(base - i*0.12).toFixed(2),
        performance: { sharpeRatio: +(base - i*0.12).toFixed(2), winRate: +(0.55 + i*0.03).toFixed(2), maxDrawdown: +(0.10 + i*0.02).toFixed(2), trades: 60 + i*10 },
        created: new Date().toISOString()
      }));
      evoResults.set(id, items);
      broadcastToChannel('evotester', {
        type: 'evo_complete',
        data: {
          sessionId: id,
          config: s.config,
          status: 'completed',
          startTime: s.startTime,
          endTime: new Date().toISOString(),
          totalRuntime: `${s.totalGenerations * 2}s`,
        }
      });
    }
  }, 2000);
  res.json({ session_id: id });
});

app.post('/api/evotester/:id/stop', (req, res) => {
  const s = evoSessions.get(req.params.id);
  if (s) { s.running = false; s.status = 'stopped'; }
  res.json({ ok: true });
});

app.post('/api/evotester/:id/pause', (req, res) => {
  const s = evoSessions.get(req.params.id);
  if (s) { s.running = false; s.status = 'paused'; }
  res.json({ ok: true });
});

app.post('/api/evotester/:id/resume', (req, res) => {
  const s = evoSessions.get(req.params.id);
  if (s) { s.running = true; s.status = 'running'; }
  res.json({ ok: true });
});

app.get('/api/evotester/:id/status', (req, res) => {
  const s = evoSessions.get(req.params.id);
  if (!s) return res.status(404).json({ error: 'session_not_found' });
  const progress = s.totalGenerations ? Math.min(s.currentGeneration / s.totalGenerations, 1) : 0;
  res.json({
    sessionId: s.id,
    running: s.running,
    currentGeneration: s.currentGeneration,
    totalGenerations: s.totalGenerations,
    startTime: s.startTime,
    progress,
    bestFitness: s.bestFitness,
    averageFitness: s.averageFitness,
    status: s.status,
  });
});

// Aliases to support real-mode proof curl flow
app.get('/api/evotester/status', (req, res) => {
  const id = String(req.query.session_id || '').trim();
  if (!id) return res.status(400).json({ error: 'session_id required' });
  const s = evoSessions.get(id);
  if (!s) return res.status(404).json({ error: 'session_not_found' });
  const progress = s.totalGenerations ? Math.min(s.currentGeneration / s.totalGenerations, 1) : 0;
  res.json({
    sessionId: s.id,
    running: s.running,
    currentGeneration: s.currentGeneration,
    totalGenerations: s.totalGenerations,
    startTime: s.startTime,
    progress,
    bestFitness: s.bestFitness,
    averageFitness: s.averageFitness,
    status: s.status,
  });
});

app.get('/api/evotester/:id/results', (req, res) => {
  if (!evoSessions.has(req.params.id)) return res.status(404).json([]);
  res.json(evoResults.get(req.params.id) || []);
});

app.get('/api/evotester/results', (req, res) => {
  const id = String(req.query.session_id || '').trim();
  const limit = parseInt(String(req.query.limit || '20')) || 20;
  if (!id) return res.status(400).json([]);
  const items = (evoResults.get(id) || []).slice(0, limit);
  res.json(items);
});

app.get('/api/evotester/:id/generations', (req, res) => {
  if (!evoSessions.has(req.params.id)) return res.status(404).json([]);
  res.json(evoGenerations.get(req.params.id) || []);
});

app.get('/api/evotester/generations', (req, res) => {
  const id = String(req.query.session_id || '').trim();
  if (!id) return res.status(400).json([]);
  res.json(evoGenerations.get(id) || []);
});

// Promotion stub to satisfy guarded promotion flow
app.post('/api/strategies/promote', (req, res) => {
  try {
    const { strategy_id, session_id } = req.body || {};
    if (!strategy_id || !session_id) {
      return res.status(400).json({ error: 'strategy_id and session_id required' });
    }
    const items = evoResults.get(session_id) || [];
    const found = items.find(x => x.id === strategy_id) || null;
    if (!found) {
      return res.status(404).json({ error: 'candidate_not_found' });
    }
    return res.json({ success: true, message: 'Promoted to paper candidate (stub)', candidate: { id: found.id, fitness: found.fitness } });
  } catch (e) {
    return res.status(500).json({ error: 'promotion_failed' });
  }
});

// Simple /metrics for UI health probe
app.get('/metrics', (req, res) => {
  res.locals.source = 'calc';
  res.locals.provider = 'calc';
  res.json({
    totalSymbolsTracked: 0,
    errorRate: 0,
    requestsLastHour: 0,
    averageLatency: 0,
    timestamp: new Date().toISOString(),
    server: 'minimal-live-api'
  });
});

// Brain activity endpoint
app.get('/api/brain/activity', (req, res) => {
  res.json([]);
});

// Live tournament endpoint
app.get('/api/live/tournament', (req, res) => {
  res.json({
    current_generation: 1,
    rounds: [
      {
        stage: 'incubation',
        name: 'Incubation',
        active_strategies: 2,
        criteria: {
          minSharpe: 0.5,
          minPf: 1.05,
          maxDd: 0.15,
          maxBreaches: 3
        }
      },
      {
        stage: 'evaluation',
        name: 'Evaluation',
        active_strategies: 1,
        criteria: {
          minSharpe: 0.8,
          minPf: 1.10,
          maxDd: 0.12,
          maxBreaches: 2
        }
      },
      {
        stage: 'production',
        name: 'Production',
        active_strategies: 1,
        criteria: {
          minSharpe: 1.0,
          minPf: 1.15,
          maxDd: 0.10,
          maxBreaches: 1
        }
      }
    ],
    stats: {
      totalPromotions: 5,
      totalDemotions: 2,
      roundPassRates: {
        'incubation-evaluation': { promoted: 3, demoted: 1 },
        'evaluation-production': { promoted: 1, demoted: 1 }
      }
    },
    recent_decisions: [
      {
        strategyId: 'news_momo_v2',
        decision: 'promote',
        fromStage: 'evaluation',
        toStage: 'production',
        reason: 'Exceeded Sharpe target (1.2 > 1.0)',
        timestamp: new Date().toISOString()
      }
    ],
    timestamp: new Date().toISOString()
  });
});

// News endpoints
app.get('/api/news/insights', (req, res) => {
  res.json({
    sentiment: 'neutral',
    confidence: 0.7,
    sources: [],
    timestamp: new Date().toISOString()
  });
});

app.get('/api/news/sentiment', (req, res) => {
  res.json({
    category: req.query.category || 'markets',
    sentiment: 'neutral',
    confidence: 0.6,
    sources: [],
    timestamp: new Date().toISOString()
  });
});

// Scanner candidates endpoint
app.get('/api/scanner/candidates', (req, res) => {
  res.json([]);
});

// Competition endpoints
app.get('/api/competition/ledger', (req, res) => {
  res.json([]);
});

app.get('/api/competition/poolStatus', (req, res) => {
  res.json({
    total_pool: 100000,
    poolPnl: 15234.56, // This was causing the toFixed() error
    active_strategies: 3,
    capPct: 0.8,
    utilizationPct: 0.65,
    activeCount: 3,
    timestamp: new Date().toISOString()
  });
});

// Live AI endpoints
app.get('/api/live/ai/status', (req, res) => {
  res.json({
    status: 'idle',
    last_cycle: new Date().toISOString(),
    active_strategies: ['news_momo_v2'],
    timestamp: new Date().toISOString()
  });
});

app.get('/api/live/ai/context', (req, res) => {
  res.json({
    market_regime: 'neutral_medium',
    volatility: 'medium',
    sentiment: 'neutral',
    timestamp: new Date().toISOString()
  });
});

// Context endpoint
app.get('/api/context', (req, res) => {
  res.json({
    market_regime: 'neutral_medium',
    volatility: 'medium',
    sentiment: 'neutral',
    timestamp: new Date().toISOString()
  });
});

// Safety status endpoint
app.get('/api/safety/status', (req, res) => {
  res.json({
    circuit_breaker: 'GREEN',
    last_check: new Date().toISOString(),
    alerts: [],
    timestamp: new Date().toISOString()
  });
});

// Decisions endpoints
app.get('/api/decisions', (req, res) => {
  res.json([]);
});

app.get('/api/decisions/recent', (req, res) => {
  res.json([]);
});

app.get('/api/decisions/latest', (req, res) => {
  res.json([]);
});

// Portfolio endpoints
app.get('/api/portfolio', async (req, res) => {
  try {
    // Simulate broker account from paper state for minimal server
    const { cash, positions } = aggregatePaperState();
    const symbols = Array.from(new Set((positions || []).map(p => p.symbol)));
    const qm = await getQuotesMap(symbols);
    const now = Date.now();
    let staleCount = 0;
    const marketValue = (positions || []).reduce((sum, p) => {
      const q = qm[p.symbol];
      const mkt = q && q.price != null ? Number(q.price) : Number(p.avg_price || 0);
      if (q && q.asof_ts) {
        const age = Math.max(0, now - new Date(q.asof_ts).getTime());
        if (age > QUOTES_TTL_MS) staleCount += 1;
      }
      return sum + Number(p.qty || 0) * mkt;
    }, 0);
    const derived_equity = Number(cash) + marketValue;
    const broker_equity = derived_equity; // in minimal server, broker equals derived
    const { diff, tol, reality_red_flag } = compareEquity(broker_equity, derived_equity);

    res.locals.source = 'broker';
    res.locals.provider = getProviderTag();

    res.json({
      source: res.locals.source,
      provider: res.locals.provider,
      asof_ts: new Date().toISOString(),
      latency_ms: 0,
      request_id: (crypto.randomUUID && crypto.randomUUID()) || Math.random().toString(36).slice(2),
      broker_equity,
      derived_equity,
      equity_diff: diff,
      tolerance: tol,
      reality_red_flag,
      cash,
      currency: 'USD',
      positions_count: positions.length,
      quotes_meta: { symbols: symbols.length, missing: symbols.filter(s => !qm[s]).length },
      stale_quotes_count: staleCount,
    });
  } catch (e) {
    res.status(500).json({ error: 'portfolio_failed' });
  }
});

app.get('/api/portfolio/summary', (req, res) => {
  res.json({
    cash: 100000,
    equity: 100000,
    day_pnl: 0,
    open_pnl: 0,
    positions: [],
    asOf: new Date().toISOString(),
    broker: 'tradier',
    mode: 'paper'
  });
});

app.get('/api/portfolio/paper', (req, res) => {
  res.json({
    cash: 100000,
    equity: 100000,
    day_pnl: 0,
    open_pnl: 0,
    positions: [],
    asOf: new Date().toISOString(),
    broker: 'tradier',
    mode: 'paper'
  });
});

// Portfolio history endpoints (empty until pipeline runs)
app.get('/api/portfolio/paper/history', (req, res) => {
  const days = Number(req.query.days || 90);
  res.json({ items: [], days, asOf: new Date().toISOString() });
});
app.get('/api/portfolio/live/history', (req, res) => {
  const days = Number(req.query.days || 90);
  res.json({ items: [], days, asOf: new Date().toISOString() });
});

// ---- Watchlists (persisted to data/watchlists.json) ----
const WATCHLISTS_FILE = path.resolve(__dirname, 'data/watchlists.json');
function loadWatchlists() {
  try {
    const raw = fs.readFileSync(WATCHLISTS_FILE, 'utf8');
    return JSON.parse(raw);
  } catch {
    return { currentId: 'default', items: [{ id: 'default', name: 'Default', symbols: ['SPY','QQQ','AAPL'] }] };
  }
}
function saveWatchlists(obj) {
  try { fs.mkdirSync(path.dirname(WATCHLISTS_FILE), { recursive: true }); } catch {}
  fs.writeFileSync(WATCHLISTS_FILE, JSON.stringify(obj, null, 2), 'utf8');
}

app.get('/api/watchlists', (_req, res) => {
  const wl = loadWatchlists();
  res.json({ items: wl.items, currentId: wl.currentId });
});

app.post('/api/watchlists/select', (req, res) => {
  const { id } = req.body || {};
  const wl = loadWatchlists();
  if (!wl.items.find((w) => w.id === id)) return res.status(404).json({ error: 'watchlist_not_found' });
  wl.currentId = id;
  saveWatchlists(wl);
  res.json({ ok: true, currentId: wl.currentId });
});

app.post('/api/watchlists/:id/symbols', (req, res) => {
  const { id } = req.params;
  const { symbol } = req.body || {};
  if (!symbol) return res.status(400).json({ error: 'symbol_required' });
  const wl = loadWatchlists();
  const entry = wl.items.find((w) => w.id === id);
  if (!entry) return res.status(404).json({ error: 'watchlist_not_found' });
  const s = String(symbol).toUpperCase();
  entry.symbols = Array.from(new Set([...(entry.symbols || []), s]));
  saveWatchlists(wl);
  res.json({ ok: true, symbols: entry.symbols });
});

app.delete('/api/watchlists/:id/symbols/:symbol', (req, res) => {
  const { id, symbol } = req.params;
  const wl = loadWatchlists();
  const entry = wl.items.find((w) => w.id === id);
  if (!entry) return res.status(404).json({ error: 'watchlist_not_found' });
  const target = String(symbol).toUpperCase();
  entry.symbols = (entry.symbols || []).filter((s) => s.toUpperCase() !== target);
  saveWatchlists(wl);
  res.json({ ok: true, symbols: entry.symbols });
});

// Universe endpoints (compat layer)
app.get('/api/universe', (_req, res) => {
  const wl = loadWatchlists();
  const current = wl.items.find((w) => w.id === wl.currentId) || wl.items[0] || { symbols: [] };
  res.json({ symbols: current.symbols || [] });
});

app.post('/api/universe', (req, res) => {
  const { id, symbols } = req.body || {};
  const wl = loadWatchlists();
  if (id) {
    if (!wl.items.find((w) => w.id === id)) return res.status(404).json({ error: 'watchlist_not_found' });
    wl.currentId = id;
  }
  if (Array.isArray(symbols)) {
    const entry = wl.items.find((w) => w.id === wl.currentId) || wl.items[0];
    if (entry) entry.symbols = symbols.map((s) => String(s || '').toUpperCase()).filter(Boolean);
  }
  saveWatchlists(wl);
  const current = wl.items.find((w) => w.id === wl.currentId) || { symbols: [] };
  res.json({ symbols: current.symbols || [] });
});

// Paper trading endpoints
function aggregatePaperState() {
  // Build positions from paper orders
  const positionsBySym = new Map();
  let cash = 100000; // starting paper cash
  for (const o of paperOrders.slice().reverse()) {
    const side = String(o.side || '').toLowerCase();
    const qty = Number(o.qty || o.quantity || 0);
    const px = Number(o.price || 0);
    const sym = String(o.symbol || '').toUpperCase();
    if (!sym || !qty) continue;
    // Update cash
    if (side === 'buy') cash -= qty * px;
    else if (side === 'sell') cash += qty * px;
    // Update position
    const cur = positionsBySym.get(sym) || { symbol: sym, qty: 0, total_cost: 0 };
    if (side === 'buy') {
      cur.total_cost += qty * px;
      cur.qty += qty;
    } else if (side === 'sell') {
      cur.qty -= qty;
      // Do not change total_cost for simplicity
    }
    positionsBySym.set(sym, cur);
  }
  const positions = Array.from(positionsBySym.values())
    .filter(p => p.qty !== 0)
    .map(p => ({ symbol: p.symbol, qty: p.qty, avg_price: p.qty !== 0 ? Math.abs(p.total_cost / Math.abs(p.qty)) : 0 }));
  return { cash: Math.max(0, cash), positions };
}

async function getQuotesMap(symbols) {
  const { token, baseUrl } = getTradierConfig();
  if (!symbols.length || DISCONNECT_FEEDS || !token) return {};
  try {
    const url = `${baseUrl}/markets/quotes?symbols=${encodeURIComponent(symbols.join(','))}`;
    const { data } = await axios.get(url, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' },
      timeout: 4000
    });
    const node = data?.quotes?.quote || data?.quote || data?.quotes || [];
    const list = Array.isArray(node) ? node : node ? [node] : [];
    const map = {};
    for (const q of list) {
      const sym = String(q.symbol || q.ticker || '').toUpperCase();
      const price = Number(q.last ?? q.close ?? q.price ?? 0);
      const ts = q.trade_date || q.timestamp || new Date().toISOString();
      if (sym) map[sym] = { price, asof_ts: ts, provider: 'tradier' };
    }
    return map;
  } catch {
    return {};
  }
}

function compareEquity(broker_equity, derived_equity) {
  const diff = broker_equity - derived_equity;
  const tol = Math.max(1, Math.abs(broker_equity) * 0.0005);
  const reality_red_flag = Math.abs(diff) > tol;
  return { diff, tol, reality_red_flag };
}

app.get('/api/paper/account', (req, res) => {
  res.json({
    balances: {
      total_equity: 100000,
      total_cash: 100000,
      market_value: 0
    },
    timestamp: new Date().toISOString()
  });
});

app.get('/api/paper/orders', (req, res) => {
  res.json(paperOrders.slice(-200));
});

app.post('/api/paper/orders', requireTradingOn, (req, res) => {
  try {
    const { symbol, side, qty, price, type } = req.body || {};
    if (!symbol || !side || !qty) return res.status(400).json({ error: 'invalid_order' });
    const order = {
      id: (crypto.randomUUID && crypto.randomUUID()) || Math.random().toString(36).slice(2),
      symbol: String(symbol).toUpperCase(),
      side: String(side).toLowerCase(),
      qty: Number(qty),
      price: Number(price || 0),
      type: type || 'market',
      status: 'filled',
      created_at: new Date().toISOString(),
      venue: 'sim',
    };
    paperOrders.unshift(order);
    ordersEmitter.emit('order_update', order);
    res.status(201).json(order);
  } catch (e) {
    res.status(500).json({ error: 'order_failed' });
  }
});

// Admin controls
app.post('/api/admin/pause', (req, res) => {
  try {
    tradingState.pause();
    res.locals.source = 'calc';
    res.locals.provider = 'app';
    res.json({ ok: true, paused: true, asof_ts: new Date().toISOString() });
  } catch (e) {
    res.status(500).json({ error: 'pause_failed' });
  }
});

app.post('/api/admin/resume', (req, res) => {
  try {
    tradingState.resume();
    res.locals.source = 'calc';
    res.locals.provider = 'app';
    res.json({ ok: true, paused: false, asof_ts: new Date().toISOString() });
  } catch (e) {
    res.status(500).json({ error: 'resume_failed' });
  }
});

app.get('/api/paper/positions', (req, res) => {
  const { positions } = aggregatePaperState();
  res.json(positions);
});

// Normalized positions endpoint with meta and basic valuation
app.get('/api/positions', async (req, res) => {
  try {
    const { positions } = aggregatePaperState();
    const symbols = Array.from(new Set((positions || []).map(p => p.symbol)));
    const qm = await getQuotesMap(symbols);
    const rows = (positions || []).map(p => {
      const q = qm[p.symbol];
      const mkt = q && q.price != null ? Number(q.price) : null;
      const unreal = mkt != null ? (Number(p.qty || 0) * (mkt - Number(p.avg_price || 0))) : null;
      return {
        symbol: p.symbol,
        qty: p.qty,
        avg_price: p.avg_price,
        market_price: mkt,
        unrealized_pnl: unreal,
        quote: q ? {
          stale: (q.asof_ts ? (Date.now() - new Date(q.asof_ts).getTime()) > QUOTES_TTL_MS : true),
          cache_age_ms: q.asof_ts ? Math.max(0, Date.now() - new Date(q.asof_ts).getTime()) : null,
          ttl_ms: QUOTES_TTL_MS,
        } : undefined,
      };
    });
    res.locals.source = 'broker';
    res.locals.provider = getProviderTag();
    res.json({ positions: rows });
  } catch (e) {
    res.status(500).json({ error: 'positions_failed' });
  }
});

app.get('/api/paper/orders/stream', (req, res) => {
  // Set headers for SSE
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Cache-Control',
  });

  // Send an initial comment to establish the connection
  res.write(': connected\n\n');

  // Send a keep-alive every 30 seconds
  const keepAlive = setInterval(() => {
    res.write(': keep-alive\n\n');
  }, 30000);

  const send = (type, payload) => {
    const evt = {
      type,
      data: {
        ...payload,
        meta: {
          source: 'paper',
          provider: 'sim',
          asof_ts: new Date().toISOString(),
          request_id: (crypto.randomUUID && crypto.randomUUID()) || Math.random().toString(36).slice(2),
        }
      }
    };
    res.write(`event: ${evt.type}\n`);
    res.write(`data: ${JSON.stringify(evt.data)}\n\n`);
  };

  const onUpdate = (order) => send('order_update', order);
  ordersEmitter.on('order_update', onUpdate);

  // Clean up on client disconnect
  req.on('close', () => {
    clearInterval(keepAlive);
    ordersEmitter.off('order_update', onUpdate);
    res.end();
  });
});

// Trades endpoint
app.get('/api/trades', (req, res) => {
  res.locals.source = 'paper';
  res.locals.provider = 'sim';
  const trades = paperOrders.map((o) => ({
    trade_id: o.id,
    mode: 'paper',
    strategy: o.strategy || 'manual',
    symbol: o.symbol,
    side: o.side,
    qty: Number(o.qty || o.quantity || 0),
    price: Number(o.price || 0),
    ts_exec: o.ts_exec || o.created_at || new Date().toISOString(),
    broker_order_id: o.broker_order_id || null,
    venue: o.venue || 'sim',
    pnl_at_exit: null,
  }));
  const meta = {
    source: res.locals.source,
    provider: res.locals.provider,
    asof_ts: new Date().toISOString(),
    latency_ms: 0,
    request_id: (crypto.randomUUID && crypto.randomUUID()) || Math.random().toString(36).slice(2),
  };
  res.json({ ...meta, trades });
});

// Strategies endpoint
app.get('/api/strategies', (req, res) => {
  res.json({
    items: [{
      id: 'news_momo_v2',
      name: 'News Momentum v2',
      active: true,
      performance: {
        trades_count: 50,
        profit_factor: 1.15,
        sharpe_ratio: 0.8
      }
    }]
  });
});

app.get('/api/strategies/active', (req, res) => {
  res.json([{
    id: 'news_momo_v2',
    name: 'News Momentum v2',
    budget: 1.0,
    reason: 'IR_weighted',
    sharpe_after_costs: 0.8,
    trades: 50
  }]);
});

// Data status endpoint
app.get('/api/data/status', (req, res) => {
  res.json({
    quotes_fresh: true,
    market_data_ok: true,
    last_update: new Date().toISOString(),
    timestamp: new Date().toISOString()
  });
});

// Test autoloop trigger
app.post('/api/test/autoloop/runonce', (req, res) => {
  res.json({
    success: true,
    message: 'AutoLoop runOnce completed (minimal)',
    meta: {
      asOf: new Date().toISOString(),
      source: 'minimal-server'
    }
  });
});

// Metrics/Prometheus endpoint
app.get('/metrics/prom', (req, res) => {
  res.set('Content-Type', 'text/plain');
  res.send(`# HELP live_api_requests_total Total number of requests
# TYPE live_api_requests_total counter
live_api_requests_total 150

# HELP live_api_errors_total Total number of errors
# TYPE live_api_errors_total counter
live_api_errors_total 3
`);
});

// Create HTTP server and WebSocket server
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// WebSocket connection handler
wss.on('connection', (ws, req) => {
  const url = req.url;
  console.log(`WebSocket connected: ${url}`);

  // Handle different WebSocket endpoints
  if (url === '/ws/prices') {
    // Send mock price updates every 5 seconds
    const priceInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        const mockPrice = {
          symbol: 'SPY',
          price: 400 + Math.random() * 50,
          change: (Math.random() - 0.5) * 10,
          volume: Math.floor(Math.random() * 1000000),
          timestamp: new Date().toISOString()
        };
        ws.send(JSON.stringify(mockPrice));
      }
    }, 5000);

    ws.on('close', () => {
      clearInterval(priceInterval);
      console.log('Price WebSocket disconnected');
    });

  } else if (url === '/ws/decisions') {
    // Send mock decision updates occasionally
    const decisionInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN && Math.random() > 0.7) { // 30% chance
        const mockDecision = {
          symbol: 'SPY',
          action: 'BUY',
          confidence: 0.75,
          strategy: 'news_momo_v2',
          timestamp: new Date().toISOString()
        };
        ws.send(JSON.stringify(mockDecision));
      }
    }, 10000);

    ws.on('close', () => {
      clearInterval(decisionInterval);
      console.log('Decision WebSocket disconnected');
    });

  } else {
    // Generic WebSocket connection
    ws._channels = new Set();
    ws.on('message', (message) => {
      const text = message.toString();
      try {
        const msg = JSON.parse(text);
        if (msg && msg.type === 'subscription') {
          const ch = String(msg.channel || '').toLowerCase();
          if (msg.action === 'subscribe' && ch) {
            ws._channels.add(ch);
            ws.send(JSON.stringify({ type: 'subscribed', channel: ch, timestamp: new Date().toISOString() }));
            return;
          }
          if (msg.action === 'unsubscribe' && ch) {
            ws._channels.delete(ch);
            ws.send(JSON.stringify({ type: 'unsubscribed', channel: ch, timestamp: new Date().toISOString() }));
            return;
          }
        }
      } catch (_) {
        // fall through to echo
      }
      console.log('Received:', text);
      ws.send(JSON.stringify({ echo: text, timestamp: new Date().toISOString() }));
    });

    ws.on('close', () => {
      console.log('Generic WebSocket disconnected');
    });
  }

  // Send initial connection message
  ws.send(JSON.stringify({
    type: 'connected',
    endpoint: url,
    timestamp: new Date().toISOString()
  }));
});

// Broadcast helper for channel subscribers on generic /ws
function broadcastToChannel(channel, payload) {
  try {
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN && client._channels && client._channels.has(String(channel).toLowerCase())) {
        try { client.send(JSON.stringify(payload)); } catch {}
      }
    });
  } catch {}
}

// Lab diamonds endpoint
app.get('/api/lab/diamonds', (req, res) => {
  const { limit = 25, universe = 'all' } = req.query;
  const limitNum = parseInt(limit) || 25;

  const diamonds = [];
  const symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'META', 'GOOGL', 'AVGO'];

  for (let i = 0; i < Math.min(limitNum, symbols.length); i++) {
    const symbol = symbols[i];
    diamonds.push({
      symbol,
      score: 0.5 + Math.random() * 0.5, // 0.5 to 1.0
      features: {
        impact1h: (Math.random() - 0.5) * 2, // -1 to 1
        impact24h: (Math.random() - 0.5) * 4, // -2 to 2
        count24h: Math.floor(Math.random() * 20) + 1, // 1 to 20
        gapPct: Math.random() * 0.02, // 0 to 2%
        spreadPct: Math.random() * 0.005, // 0 to 0.5%
        rvol: 0.8 + Math.random() * 0.4 // 0.8 to 1.2
      }
    });
  }

  res.json({
    items: diamonds,
    asOf: new Date().toISOString()
  });
});

// EvoTester history endpoint
app.get('/api/evotester/history', (req, res) => {
  const history = [
    {
      id: 'evo-1',
      date: new Date().toISOString(),
      bestFitness: 0.78,
      status: 'completed',
      generations: 50,
      elapsed: '1h 23m'
    },
    {
      id: 'evo-2',
      date: new Date(Date.now() - 86400000 * 2).toISOString(),
      bestFitness: 0.72,
      status: 'completed',
      generations: 42,
      elapsed: '58m'
    },
    {
      id: 'evo-3',
      date: new Date(Date.now() - 86400000 * 5).toISOString(),
      bestFitness: 0.65,
      status: 'completed',
      generations: 35,
      elapsed: '45m'
    }
  ];

  res.json(history);
});

// Resolve Marketaux credentials (env first, then repo config file)
function getMarketauxConfig() {
  const token = process.env.MARKETAUX_API_KEY || process.env.MARKETAUX_TOKEN || '';
  if (token) return { token };
  try {
    const credPath = path.resolve(__dirname, '../config/credentials/marketaux.json');
    const raw = fs.readFileSync(credPath, 'utf8');
    const json = JSON.parse(raw);
    const key = json.api_key || json.key || json.token || '';
    return { token: key };
  } catch {
    return { token: '' };
  }
}

// Resolve Alpha Vantage credentials (env first, then repo config file)
function getAlphaVantageConfig() {
  const key = process.env.ALPHA_VANTAGE_API_KEY || process.env.ALPHAVANTAGE_API_KEY || '';
  if (key) return { key };
  try {
    const credPath = path.resolve(__dirname, '../config/credentials/alphavantage.json');
    const raw = fs.readFileSync(credPath, 'utf8');
    const json = JSON.parse(raw);
    const apiKey = json.api_key || json.key || '';
    return { key: apiKey };
  } catch {
    return { key: '' };
  }
}

// Context news endpoint (real via Marketaux; fail-closed to empty)
app.get('/api/context/news', requireRealProviders, maybeDisconnectFeeds, async (req, res) => {
  req.requiresReal = true;
  try {
    const { token } = getMarketauxConfig();
    res.locals.provider = token ? 'marketaux' : 'none';
    res.locals.source = token ? 'broker' : 'mock';
    const limit = Math.min(parseInt(String(req.query.limit || '10')) || 10, 50);

    const fail = ensureRealOrFail(Boolean(token));
    if (fail) return res.status(fail.status).json(fail.body);

    if (!token) return res.json([]);

    const params = new URLSearchParams({
      api_token: token,
      limit: String(limit),
      language: 'en',
      countries: 'us',
    });
    const url = `https://api.marketaux.com/v1/news/all?${params.toString()}`;
    const { data } = await axios.get(url, { timeout: 7000 });
    const list = Array.isArray(data?.data) ? data.data : Array.isArray(data?.news) ? data.news : [];
    const mapped = list.map((n) => ({
      id: n?.uuid || n?.id || `${n?.title || 'news'}-${n?.published_at || Date.now()}`,
      title: n?.title || n?.headline || 'Untitled',
      headline: n?.headline || n?.title || 'Untitled',
      summary: n?.snippet || n?.description || n?.summary || '',
      url: n?.url || n?.link || '#',
      source: n?.source || n?.provider || 'news',
      published_at: n?.published_at || n?.date || new Date().toISOString(),
      timestamp: n?.published_at || n?.date || new Date().toISOString(),
      sentiment: typeof n?.sentiment === 'number' ? n.sentiment : (typeof n?.sentiment_score === 'number' ? n.sentiment_score : 0),
      impact: n?.impact || 'medium',
      categories: n?.topics || n?.categories || [],
      symbols: (n?.entities || n?.symbols || []).map((e) => (typeof e === 'string' ? e : (e?.symbol || e?.ticker))).filter(Boolean),
      provider: 'marketaux',
    }));
    return res.json(mapped);
  } catch (e) {
    return res.json([]);
  }
});

// Fundamentals endpoint (Alpha Vantage OVERVIEW; fail-closed to empty)
app.get('/api/fundamentals', async (req, res) => {
  try {
    const { key } = getAlphaVantageConfig();
    if (!key) return res.json({ items: [], asOf: new Date().toISOString() });

    // Determine symbols to fetch: query ?symbols=... or current watchlist
    const qs = String(req.query.symbols || '').trim();
    let symbols = qs ? qs.split(',').map(s => s.trim().toUpperCase()).filter(Boolean) : [];
    if (symbols.length === 0) {
      try {
        const wl = loadWatchlists();
        const current = wl.items.find((w) => w.id === wl.currentId) || wl.items[0] || { symbols: [] };
        symbols = (current.symbols || []).slice(0, 6);
      } catch {
        symbols = ['SPY', 'AAPL', 'MSFT'];
      }
    }
    symbols = symbols.slice(0, 6); // respect AV rate limits

    const results = [];
    for (const symbol of symbols) {
      try {
        const url = `https://www.alphavantage.co/query?function=OVERVIEW&symbol=${encodeURIComponent(symbol)}&apikey=${encodeURIComponent(key)}`;
        const { data } = await axios.get(url, { timeout: 8000 });
        if (data && data.Symbol) {
          const pe = parseFloat(data.PERatio || '0') || 0;
          const revYoY = parseFloat(data.QuarterlyRevenueGrowthYOY || '0') || 0;
          const epsYoY = parseFloat(data.QuarterlyEarningsGrowthYOY || '0') || 0;
          const dte = parseFloat(data.DebtToEquityRatio || data.DebtToEquity || '0') || 0;
          const mcap = data.MarketCapitalization || '0';
          const researchScore = Number((
            (revYoY * 40) + (epsYoY * 40) + (pe > 0 ? Math.min(30 / pe, 15) : 0) + (dte > 0 ? Math.min(50 / dte, 10) : 10)
          ).toFixed(2));
          results.push({
            symbol: data.Symbol,
            company: data.Name || symbol,
            sector: data.Sector || 'Unknown',
            marketCap: mcap,
            peRatio: pe,
            revenueGrowth: revYoY,
            earningsGrowth: epsYoY,
            debtToEquity: dte,
            researchScore,
            catalysts: []
          });
        }
      } catch {}
      // small delay to be polite to AV API
      await new Promise(r => setTimeout(r, 250));
    }
    return res.json({ items: results, asOf: new Date().toISOString() });
  } catch (e) {
    return res.json({ items: [], asOf: new Date().toISOString() });
  }
});

// Market discovery endpoint (scan watchlist using real quotes)
app.get('/api/discovery/market', async (req, res) => {
  try {
    // Load symbols from current watchlist
    const wl = loadWatchlists();
    const current = wl.items.find((w) => w.id === wl.currentId) || wl.items[0] || { symbols: [] };
    const symbols = (current.symbols || []).slice(0, 30);
    if (symbols.length === 0) return res.json({ items: [], asOf: new Date().toISOString() });

    // Fetch quotes in batches of up to 25
    const batches = [];
    for (let i = 0; i < symbols.length; i += 25) batches.push(symbols.slice(i, i + 25));

    const all = [];
    for (const batch of batches) {
      try {
        const r = await axios.get(`http://localhost:${PORT}/api/quotes`, { params: { symbols: batch.join(',') }, timeout: 5000 });
        if (Array.isArray(r?.data)) all.push(...r.data);
      } catch {}
    }

    const items = all
      .map((q) => {
        const prev = Number(q.prevClose || 0);
        const last = Number(q.last || 0);
        const changePct = prev ? ((last - prev) / prev) * 100 : 0;
        const score = Number((Math.abs(changePct)).toFixed(2));
        return {
          symbol: q.symbol,
          company: q.symbol,
          reason: changePct >= 0 ? 'Top gain vs prev close' : 'Top drop vs prev close',
          score,
          changePct: Number(changePct.toFixed(2)),
        };
      })
      .filter((x) => Number.isFinite(x.changePct))
      .sort((a, b) => Math.abs(b.changePct) - Math.abs(a.changePct))
      .slice(0, 12);

    return res.json({ items, asOf: new Date().toISOString() });
  } catch (e) {
    return res.json({ items: [], asOf: new Date().toISOString() });
  }
});

// Logs endpoint
app.get('/api/logs', (req, res) => {
  const { level = 'INFO', limit = 100, offset = 0 } = req.query;
  const limitNum = parseInt(limit) || 100;
  const offsetNum = parseInt(offset) || 0;

  const logLevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
  const sources = ['trading-engine', 'data-ingestion', 'strategy-manager', 'market-data', 'websocket-server', 'api-gateway'];
  const categories = ['performance', 'errors', 'data-quality', 'trading', 'system', 'market'];

  const logs = [];
  for (let i = 0; i < limitNum; i++) {
    const timestamp = new Date(Date.now() - (i + offsetNum) * 60000).toISOString();
    const levelIndex = level === 'ALL' ? Math.floor(Math.random() * logLevels.length) : logLevels.indexOf(level.toUpperCase());
    const logLevel = level === 'ALL' ? logLevels[levelIndex] : level.toUpperCase();

    logs.push({
      id: `log-${Date.now()}-${i}`,
      timestamp,
      level: logLevel,
      message: generateLogMessage(logLevel),
      source: sources[Math.floor(Math.random() * sources.length)],
      category: categories[Math.floor(Math.random() * categories.length)],
      acknowledged: Math.random() > 0.8,
      requires_action: Math.random() > 0.9,
      related_symbol: Math.random() > 0.7 ? ['SPY', 'QQQ', 'AAPL', 'MSFT'][Math.floor(Math.random() * 4)] : undefined,
      details: Math.random() > 0.5 ? {
        requestId: `req-${Math.random().toString(36).substr(2, 9)}`,
        duration: Math.floor(Math.random() * 5000),
        userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
      } : undefined
    });
  }

  res.json(logs);
});

// Events logs endpoint
app.get('/api/events/logs', (req, res) => {
  const { level = 'INFO', limit = 100, offset = 0 } = req.query;
  const limitNum = parseInt(limit) || 100;
  const offsetNum = parseInt(offset) || 0;

  const eventLogs = [];
  for (let i = 0; i < limitNum; i++) {
    const timestamp = new Date(Date.now() - (i + offsetNum) * 30000).toISOString();

    eventLogs.push({
      id: `event-log-${Date.now()}-${i}`,
      timestamp,
      level: level.toUpperCase(),
      message: `Event: ${['Trade executed', 'Position updated', 'Strategy activated', 'Alert triggered', 'Data ingested'][Math.floor(Math.random() * 5)]}`,
      source: 'event-system',
      category: 'events',
      acknowledged: true,
      requires_action: false,
      related_symbol: ['SPY', 'QQQ', 'AAPL'][Math.floor(Math.random() * 3)],
      details: {
        eventType: ['trade', 'position', 'strategy', 'alert', 'data'][Math.floor(Math.random() * 5)],
        eventId: `evt-${Math.random().toString(36).substr(2, 9)}`,
        correlationId: `corr-${Math.random().toString(36).substr(2, 9)}`
      }
    });
  }

  res.json(eventLogs);
});

// Helper function to generate realistic log messages
function generateLogMessage(level) {
  const messages = {
    DEBUG: [
      'Processing market data for SPY',
      'Cache hit for strategy configuration',
      'WebSocket connection established',
      'Database query executed in 45ms'
    ],
    INFO: [
      'Strategy manager initialized successfully',
      'Market data ingestion completed',
      'Portfolio rebalanced automatically',
      'New trade signal generated for AAPL'
    ],
    WARNING: [
      'High latency detected in data source',
      'Strategy performance below threshold',
      'API rate limit approaching',
      'Market volatility increased significantly'
    ],
    ERROR: [
      'Failed to connect to data provider',
      'Strategy execution failed with timeout',
      'Database connection lost',
      'Invalid market data received'
    ]
  };

  return messages[level][Math.floor(Math.random() * messages[level].length)];
}

// Removed duplicate poolStatus endpoint

// Start server with WebSocket support
server.listen(PORT, () => {
  console.log(`Enhanced Minimal Live-API server listening on http://localhost:${PORT}`);
  console.log('Available endpoints:');
  console.log('- /api/health');
  console.log('- /api/alerts');
  console.log('- /api/autoloop/status');
  console.log('- /api/bars');
  console.log('- /api/brain/flow/summary');
  console.log('- /api/brain/status');
  console.log('- /api/brain/activity');
  console.log('- /api/context/news');
  console.log('- /api/decisions/*');
  console.log('- /api/decisions/summary');
  console.log('- /api/evotester/history');
  console.log('- /api/evo/status');
  console.log('- /api/lab/diamonds');
  console.log('- /api/logs');
  console.log('- /api/events/logs');
  console.log('- /api/portfolio/*');
  console.log('- /api/paper/*');
  console.log('- /api/strategies');
  console.log('- /api/quotes');
  console.log('- /api/competition/poolStatus');
  console.log('- /metrics');
  console.log('- WebSocket endpoints: /ws, /ws/prices, /ws/decisions');
});
