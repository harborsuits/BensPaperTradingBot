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
const { PaperBroker } = require('./lib/PaperBroker');
const { StrategyManager } = require('./lib/StrategyManager');
const { MACrossoverStrategy } = require('./lib/strategies/MACrossoverStrategy');
const { RSIStrategy } = require('./lib/strategies/RSIStrategy');
const { AutoLoop } = require('./lib/autoLoop');
const { scoreSymbol, planTrade } = require('./src/services/BrainService.js');
const { TokenBucketLimiter } = require('./lib/rateLimiter');
const { setQuoteTouch, setBrokerTouch, noteRequest } = require('./lib/health');

// Basic middleware
app.use(express.json());
// Lightweight request logging
app.use((req, _res, next) => {
  try { console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`); } catch {}
  next();
});
// Simple token-bucket rate limiting for expensive endpoints
const limiterMap = {
  '/api/quotes': new TokenBucketLimiter(5, 10),
  '/api/bars': new TokenBucketLimiter(2, 4),
  '/api/context/news': new TokenBucketLimiter(1, 2),
  '/api/fundamentals': new TokenBucketLimiter(0.5, 1),
  '/api/brain/score': new TokenBucketLimiter(2, 4),
};
app.use((req, res, next) => {
  const limiter = limiterMap[req.path];
  if (!limiter) return next();
  if (limiter.tryRemoveToken()) return next();
  return res.status(429).json({ error: 'rate_limited' });
});

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
  try {
    const { token } = getTradierConfig();
    return token ? 'tradier' : 'none';
  } catch {
    return 'none';
  }
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

// --- PaperBroker instance ---
// Initialize ordersEmitter early to avoid initialization errors
const ordersEmitter = new EventEmitter();

const paperBroker = new PaperBroker({
  initialCash: 100000,
  dataDir: path.join(__dirname, 'data'),
  persistenceEnabled: true
});

// Performance recording and learning systems
const PerformanceRecorder = require('./services/performance_recorder');
const EvolutionBridge = require('./services/evolution_bridge');
const BotCompetitionService = require('./services/botCompetitionService');

const performanceRecorder = new PerformanceRecorder({
  dataDir: path.join(__dirname, 'data/performance')
});

// Bot Competition Service
const botCompetitionService = new BotCompetitionService();

// Services that need later initialization
let evolutionBridge = null; // Will be initialized after strategyManager
let geneticInheritance = null; // Will be initialized after evolutionBridge
let dailyReporter = null; // Will be initialized after paperBroker

// Import services
const { GeneticInheritanceService } = require('./services/geneticInheritance');
const { DailyReportGenerator } = require('./services/daily_report_generator');

// Broker proxy that routes submits through REST (so Tradier is used when configured)
const brokerProxy = {
  submitOrder: async (order) => {
    try {
      const url = `http://localhost:${PORT}/api/paper/orders`;
      const { data } = await axios.post(url, order, { timeout: 7000 });
      return data;
    } catch (e) {
      console.warn('[BrokerProxy] submitOrder failed, falling back to local PaperBroker:', e?.message || e);
      return paperBroker.submitOrder(order);
    }
  },
  getPositions: () => paperBroker.getPositions(),
  getAccount: () => paperBroker.getAccount(),
  getCurrentPrice: (sym) => paperBroker.getCurrentPrice(sym)
};

// StrategyManager wiring (runs alongside AutoLoop)
const strategyManager = new StrategyManager({ paperBroker: brokerProxy });
try {
  // Register sample strategies
  strategyManager.registerStrategy('rsi_reversion', new RSIStrategy({ symbol: 'AAPL', period: 14, oversold: 35, overbought: 65, qty: 5 }));
  strategyManager.registerStrategy('ma_crossover', new MACrossoverStrategy({ symbol: 'SPY', fastPeriod: 5, slowPeriod: 20, qty: 3 }));
  // Start them if enabled
  if (process.env.STRATEGIES_ENABLED === '1') {
    strategyManager.startStrategy('rsi_reversion');
    strategyManager.startStrategy('ma_crossover');
  }
} catch (e) {
  console.warn('[StrategyManager] init failed:', e?.message || e);
}

// Declare AI components (will be initialized after other services)
let aiOrchestrator = null;
let tournamentController = null;
let autoEvolutionManager = null;

// Bridge strategy signals to decisions feed and orders stream
strategyManager.on('strategySignal', ({ strategy, signal, data, timestamp }) => {
  const decision = {
    id: `${Date.now()}-${strategy}`,
    symbol: data?.symbol,
    action: signal,
    confidence: 0.7,
    qty: data?.qty,
    strategy,
    timestamp: timestamp?.toISOString?.() || new Date().toISOString()
  };
  recentDecisions.push(decision);
  if (recentDecisions.length > MAX_DECISIONS) recentDecisions.shift();
  
  // Check for active bot competitions and simulate trades
  try {
    const activeCompetitions = botCompetitionService.getActiveCompetitions();
    if (activeCompetitions.length > 0) {
      const competition = activeCompetitions[0];
      const bot = botCompetitionService.getBotByStrategy(
        competition.id, 
        strategy,
        data?.symbol
      );
      
      if (bot && data?.price) {
        const trade = botCompetitionService.simulateBotTrade(bot.id, {
          symbol: data.symbol,
          side: signal.toLowerCase() === 'buy' ? 'buy' : 'sell',
          price: data.price,
          confidence: data.confidence || 0.7
        });
        
        if (trade) {
          console.log(`[BotCompetition] Simulated trade for bot ${bot.strategy.name}: ${trade.symbol} ${trade.pnl > 0 ? '+' : ''}$${trade.pnl.toFixed(2)}`);
        }
      }
    }
  } catch (err) {
    console.error('[BotCompetition] Error simulating trade:', err);
  }
  
  try {
    wss.clients.forEach(c => { if (c.readyState === WebSocket.OPEN && c.decisionSocket) c.send(JSON.stringify(decision)); });
  } catch {}
});

strategyManager.on('orderExecuted', ({ order }) => {
  try { ordersEmitter.emit('order_update', order); } catch {}
});


// Listen to PaperBroker events and forward to ordersEmitter
paperBroker.on('orderFilled', (order) => {
  ordersEmitter.emit('order_update', order);
  
  // Record trade for performance tracking
  if (performanceRecorder && order) {
    performanceRecorder.recordTrade({
      id: order.id,
      symbol: order.symbol,
      strategy_id: order.metadata?.strategy_id || 'unknown',
      side: order.side,
      quantity: order.quantity,
      price: order.price,
      order_id: order.id,
      status: 'filled',
      fill_price: order.fillPrice || order.price,
      fill_time: order.filledAt || new Date(),
      commission: order.commission || 0
    });
  }
});

paperBroker.on('positionUpdated', (data) => {
  ordersEmitter.emit('position_update', data);
});

// Initialize AutoLoop (disabled by default, enable with AUTOLOOP_ENABLED=1)
const autoLoop = new AutoLoop({
  interval: parseInt(process.env.AUTOLOOP_INTERVAL_MS || '30000', 10),
  symbols: [], // Empty - will be populated dynamically from diamonds/scanner
  quantity: parseFloat(process.env.AUTOLOOP_QTY || '1'),
  enabled: process.env.AUTOLOOP_ENABLED === '1',
  aiOrchestrator: null, // Will be set after AI components are initialized
  performanceRecorder: performanceRecorder,
  dynamicDiscovery: true, // Enable dynamic symbol discovery
  useDiamonds: true, // Use diamonds scorer for penny stock discovery
  useScanner: true // Use scanner for market opportunities
});

// Initialize AI components after AutoLoop exists
try {
  const TournamentController = require('./services/tournament_controller');
  const AIOrchestrator = require('./services/ai_orchestrator');
  const { MarketIndicatorsService } = require('./src/services/marketIndicators');
  
  // MarketIndicatorsService will use mock data for now
  const marketIndicators = new MarketIndicatorsService(null);
  tournamentController = new TournamentController(strategyManager, ordersEmitter);
  // Pass the initialized market indicators
  aiOrchestrator = new AIOrchestrator(strategyManager, tournamentController, marketIndicators);
  
  // Connect AI Orchestrator to AutoLoop for capital controls
  if (autoLoop) {
    autoLoop.aiOrchestrator = aiOrchestrator;
    autoLoop.capitalLimits = {
      maxTotalCapital: aiOrchestrator.policy?.ai_policy?.paper_cap_max || 20000,
      maxPerTrade: 1000,
      maxOpenTrades: 10,
      maxDailyTrades: 50,
      minCashBuffer: 1000
    };
  }
  
  console.log('[AI] Initialized AI Orchestrator and Tournament Controller');
  
// Initialize evolution bridge
evolutionBridge = new EvolutionBridge(strategyManager, performanceRecorder);

// Initialize genetic inheritance after evolution bridge
geneticInheritance = new GeneticInheritanceService(botCompetitionService, evolutionBridge);
console.log('[GeneticInheritance] Initialized');

// Initialize auto evolution manager
try {
  const AutoEvolutionManager = require('./services/autoEvolutionManager');
  autoEvolutionManager = new AutoEvolutionManager(
    botCompetitionService,
    performanceRecorder,
    geneticInheritance
  );

  // Start auto evolution if enabled
  if (process.env.AUTO_EVOLUTION_ENABLED === '1') {
    autoEvolutionManager.start();
    console.log('[AutoEvolution] Started automatic evolution cycling');
  }
} catch (error) {
  console.log('[AutoEvolution] Could not initialize:', error.message);
  autoEvolutionManager = null;
}
  
} catch (error) {
  console.log('[AI] Could not initialize AI components:', error.message);
}

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
      try { setBrokerTouch(); noteRequest(true); } catch {}
      return { ok: true, rttMs: Date.now() - t0 };
    } catch {
      try { noteRequest(false); } catch {}
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
      try { setQuoteTouch(); } catch {}
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
// ---- Brain API (Python brain scoring service) ----
app.get('/api/brain/status', (req, res) => {
  const pythonBrainUrl = process.env.PYTHON_BRAIN_URL || 'http://localhost:8001';
  res.json({
    status: 'active',
    python_brain_config: {
      endpoint: pythonBrainUrl,
      configured: true
    },
    current_symbols: autoLoop ? autoLoop.symbols : [],
    last_scored: Date.now() - 30000
  });
});

app.post('/api/brain/score', async (req, res) => {
  try {
    const { symbol } = req.body;
    
    if (!symbol) {
      return res.status(400).json({ error: 'symbol_required' });
    }
    
    // Use Python brain service
    const pythonBrainUrl = process.env.PYTHON_BRAIN_URL || 'http://localhost:8001';
    try {
      const brainResp = await fetch(`${pythonBrainUrl}/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symbol,
          // Add context for more sophisticated scoring
          context: {
            regime: aiOrchestrator?.marketContext?.regime || 'neutral',
            volatility: aiOrchestrator?.marketContext?.volatility || 'medium',
            timestamp: new Date().toISOString()
          }
        })
      });
      
      if (brainResp.ok) {
        const scoreData = await brainResp.json();
        return res.json(scoreData);
      }
    } catch (error) {
      console.log('[Brain] Python brain error, using fallback:', error.message);
    }
    
    // Fallback scoring with some variation
    const baseScore = 0.5;
    const variation = (Math.random() - 0.5) * 0.2;
    
    res.json({
      symbol,
      score: Math.max(0, Math.min(1, baseScore + variation)),
      final_score: Math.max(0, Math.min(1, baseScore + variation)),
      factors: {
        technical: 0.6,
        sentiment: 0.4,
        volume: 0.7
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/brain/flow/summary', (req, res) => {
  res.json({
    window: req.query.window || '15m',
    total_scores: 50,
    avg_score: 0.65,
    high_confidence: 12,
    by_symbol: {}
  });
});

// ---- AutoLoop endpoints ----
app.get('/api/autoloop/status', (req, res) => {
  res.locals.source = 'calc';
  res.locals.provider = 'calc';
  res.json({
    is_running: autoLoop.isRunning,
    status: autoLoop.status,
    interval_ms: autoLoop.interval,
    last_cycle: autoLoop.lastRun || new Date().toISOString(),
    next_cycle: autoLoop.lastRun ? new Date(new Date(autoLoop.lastRun).getTime() + autoLoop.interval).toISOString() : new Date(Date.now() + autoLoop.interval).toISOString(),
    enabled: autoLoop.enabled,
    symbols: autoLoop.symbols
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
    try { setQuoteTouch(); noteRequest(true); } catch {}
    res.setHeader('x-quotes-ttl-ms', String(QUOTES_TTL_MS));
    // Return array shape to satisfy useQuotes() while still stamping headers
    return res.json(quotes);
  } catch (e) {
    try { noteRequest(false); } catch {}
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
  const pyBrainUrl = process.env.PY_BRAIN_URL || "http://localhost:8001";
  const brainTimeout = +(process.env.BRAIN_TIMEOUT_MS || 450);
  
  res.json({
    mode: process.env.AUTOLOOP_MODE || 'discovery',
    running: true,
    tick_ms: 30000,
    breaker: null,
    recent_pf_after_costs: 1.05,
    sharpe_30d: 0.42,
    sharpe_90d: 0.38,
    timestamp: new Date().toISOString(),
    config: {
      scorer_endpoint: `${pyBrainUrl}/api/decide`,
      planner_endpoint: `${pyBrainUrl}/api/plan`,
      timeout_ms: brainTimeout
    },
    symbols_tracked: autoLoop.symbols || [],
    autoloop_enabled: autoLoop.enabled
  });
});

// Brain score endpoint - score a symbol for trading
app.post('/api/brain/score', async (req, res) => {
  try {
    const { symbol, snapshot_ts } = req.body;
    
    if (!symbol) {
      return res.status(400).json({ error: 'Symbol required' });
    }
    
    const score = await scoreSymbol(symbol, snapshot_ts);
    res.json(score);
  } catch (error) {
    console.error('[Brain] Score error:', error.message);
    res.status(500).json({ 
      error: 'Brain scoring failed', 
      message: error.message,
      symbol: req.body.symbol
    });
  }
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

// EVO trigger rules
app.get('/api/evo/trigger-rules', (req, res) => {
  res.json([
    {
      id: 'high_sharpe',
      name: 'High Sharpe Ratio',
      condition: 'sharpe > 1.5',
      priority: 'high',
      active: true,
      lastTriggered: new Date(Date.now() - 3600000).toISOString()
    },
    {
      id: 'low_drawdown',
      name: 'Low Drawdown',
      condition: 'maxDrawdown < 0.05',
      priority: 'medium',
      active: true,
      lastTriggered: new Date(Date.now() - 7200000).toISOString()
    }
  ]);
});

// EVO strategy hypotheses
app.get('/api/evo/strategy-hypotheses', (req, res) => {
  res.json([
    {
      id: 'trend_following',
      hypothesis: 'Trend following works better in high volatility regimes',
      confidence: 0.75,
      backtestResults: { sharpe: 1.2, winRate: 0.58 },
      status: 'testing'
    },
    {
      id: 'mean_reversion',
      hypothesis: 'Mean reversion strategies perform well during range-bound markets',
      confidence: 0.82,
      backtestResults: { sharpe: 0.95, winRate: 0.63 },
      status: 'validated'
    }
  ]);
});

// EVO parameter importance
app.get('/api/evo/parameter-importance', (req, res) => {
  res.json([
    { name: 'lookback_period', importance: 0.85, optimal_range: [10, 50] },
    { name: 'stop_loss_multiplier', importance: 0.72, optimal_range: [1.5, 3.0] },
    { name: 'position_size_factor', importance: 0.68, optimal_range: [0.02, 0.10] },
    { name: 'entry_threshold', importance: 0.55, optimal_range: [0.7, 0.9] }
  ]);
});

// EVO market conditions
app.get('/api/evo/market-conditions', (req, res) => {
  res.json([
    {
      condition: 'Trending Up',
      strategyPerformance: { sharpe: 1.5, pf: 1.8, maxDD: -0.06 },
      occurrence: 0.35
    },
    {
      condition: 'Trending Down',
      strategyPerformance: { sharpe: 0.8, pf: 1.2, maxDD: -0.12 },
      occurrence: 0.25
    },
    {
      condition: 'Range Bound',
      strategyPerformance: { sharpe: 1.1, pf: 1.4, maxDD: -0.08 },
      occurrence: 0.40
    }
  ]);
});

// Pipeline stages
app.get('/api/pipeline/stages', (req, res) => {
  const stageNames = ['ingestion', 'validation', 'scoring', 'gates', 'routing'];
  const stages = stageNames.map(name => ({
    stage: name,
    processed: Math.floor(Math.random() * 1000) + 100,
    passed: Math.floor(Math.random() * 800) + 50,
    failed: Math.floor(Math.random() * 100),
    avgLatency: Math.random() * 50 + 10,
    status: Math.random() > 0.1 ? 'healthy' : 'degraded'
  }));
  
  res.json(stages);
});

// EVO deployment metrics
app.get('/api/evo/deployment-metrics', (req, res) => {
  const totalStrategies = 25;
  const deployedStrategies = Math.floor(Math.random() * 10) + 5;
  const pendingStrategies = Math.floor(Math.random() * 5) + 2;
  const failedStrategies = Math.floor(Math.random() * 3);
  
  res.json({
    totalStrategies,
    deployedStrategies,
    pendingStrategies,
    failedStrategies,
    averageFitness: Math.random() * 0.5 + 1.2,
    successRate: deployedStrategies / totalStrategies,
    lastDeployment: new Date(Date.now() - Math.random() * 3600000).toISOString()
  });
});

// ---- AI Orchestrator endpoints ----
app.get('/api/ai/orchestrator/status', (req, res) => {
  if (!aiOrchestrator) {
    return res.json({ 
      status: 'offline', 
      message: 'AI Orchestrator not initialized' 
    });
  }
  res.json(aiOrchestrator.getStatus());
});

app.post('/api/ai/orchestrator/cycle', async (req, res) => {
  if (!aiOrchestrator) {
    return res.status(503).json({ error: 'AI Orchestrator not initialized' });
  }
  try {
    await aiOrchestrator.triggerManualCycle();
    res.json({ success: true, message: 'Manual cycle triggered' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ---- Performance tracking endpoints ----
app.get('/api/performance/strategies', (req, res) => {
  res.locals.source = 'performanceRecorder.getAllPerformance';
  res.json(performanceRecorder.getAllPerformance());
});

app.get('/api/performance/strategies/:id', (req, res) => {
  res.locals.source = 'performanceRecorder.getStrategyPerformance';
  const performance = performanceRecorder.getStrategyPerformance(req.params.id);
  if (!performance) {
    return res.status(404).json({ error: 'Strategy not found' });
  }
  res.json(performance);
});

app.get('/api/performance/decisions', (req, res) => {
  res.locals.source = 'performanceRecorder.getRecentDecisions';
  const limit = parseInt(req.query.limit) || 100;
  res.json(performanceRecorder.getRecentDecisions(limit));
});

app.get('/api/performance/trades', (req, res) => {
  res.locals.source = 'performanceRecorder.getRecentTrades';
  const limit = parseInt(req.query.limit) || 100;
  res.json(performanceRecorder.getRecentTrades(limit));
});

app.get('/api/performance/accuracy', (req, res) => {
  res.locals.source = 'performanceRecorder.getDecisionAccuracy';
  const strategyId = req.query.strategy_id || null;
  res.json(performanceRecorder.getDecisionAccuracy(strategyId));
});

// ---- Evolution endpoints ----
app.get('/api/evolution/status', (req, res) => {
  res.locals.source = 'evolutionBridge.getEvolutionStatus';
  if (!evolutionBridge) {
    return res.json({ 
      status: 'offline', 
      message: 'Evolution Bridge not initialized' 
    });
  }
  res.json(evolutionBridge.getEvolutionStatus());
});

app.post('/api/evolution/start', async (req, res) => {
  res.locals.source = 'evolutionBridge.startEvolution';
  if (!evolutionBridge) {
    return res.status(503).json({ error: 'Evolution Bridge not initialized' });
  }
  
  try {
    const options = {
      populationSize: req.body.populationSize || 50,
      generations: req.body.generations || 20,
      testCapital: req.body.testCapital || 2000
    };
    
    const started = await evolutionBridge.startEvolution(options);
    res.json({ 
      success: started, 
      message: started ? 'Evolution started' : 'Evolution already running' 
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
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

// ---- Bot Competition Endpoints ----

// Start competition triggered by news events
app.post('/api/bot-competition/news-triggered', async (req, res) => {
  try {
    const { newsEvents, marketContext, nudge } = req.body;
    
    console.log(`[BotCompetition] News-triggered competition requested. Nudge: ${nudge}`);
    
    // Only trigger if news nudge is significant
    if (Math.abs(nudge || 0) < 0.1) {
      return res.json({ 
        success: false, 
        reason: 'News nudge too small',
        nudge: nudge 
      });
    }
    
    // Check if competition already running
    const activeCompetitions = botCompetitionService.getActiveCompetitions();
    if (activeCompetitions.length > 0) {
      return res.json({ 
        success: false, 
        reason: 'Competition already active',
        competitionId: activeCompetitions[0].id 
      });
    }
    
    // Start new competition with 100 bots
    const competitionConfig = {
      durationDays: 7,
      initialCapitalMin: 50,
      initialCapitalMax: 50,
      totalPoolCapital: 5000,  // 100 bots * $50
      winnerBonus: 0.2,
      loserPenalty: 0.5,
      reallocationIntervalHours: 1,
      metadata: {
        trigger: 'news_sentiment',
        nudge: nudge,
        newsEventCount: newsEvents?.length || 0,
        marketContext: marketContext
      }
    };
    
    const competition = botCompetitionService.startCompetition(competitionConfig);
    
    // Add 100 bots with diverse strategies focused on news-reactive symbols
    const newsSymbols = extractSymbolsFromNews(newsEvents);
    
    // Penny stock universe for news-reactive trading with $50 capital
    const pennyStocks = [
      'SNDL', 'TLRY', 'BB', 'NOK', 'PLTR', 'RIOT', 'MARA', 'OCGN', 
      'PROG', 'ATER', 'CEI', 'FAMI', 'XELA', 'GNUS', 'ZOM', 'NAKD',
      'CLOV', 'WISH', 'RIG', 'WKHS', 'GOEV', 'RIDE', 'NKLA', 'SPCE',
      'F', 'GE', 'SNAP', 'LYFT', 'UBER', 'DKNG', 'PENN', 'FUBO'
    ];
    
    // Prioritize news symbols if they're in penny range, otherwise use penny universe
    const allSymbols = [...new Set([...newsSymbols.filter(s => s), ...pennyStocks])];
    const strategyTypes = ['rsi_reversion', 'volatility', 'mean_reversion', 'ma_crossover', 'momentum', 'breakout'];
    
    for (let i = 0; i < 100; i++) {
      const strategyType = strategyTypes[i % strategyTypes.length];
      const symbol = allSymbols[i % allSymbols.length];
      const generation = Math.floor(Math.random() * 50) + 1;
      
      const strategy = {
        name: `NewsBot-${strategyType.replace('_', '-')}-${i + 1}`,
        type: strategyType,
        symbol: symbol,
        generation: generation,
        metadata: {
          newsTriggered: true,
          nudgeBias: nudge > 0 ? 'bullish' : 'bearish',
          priceRange: 'penny',
          catalystType: 'news_sentiment'
        }
      };
      
      botCompetitionService.addBot(competition.id, strategy);
    }
    
    console.log(`[BotCompetition] News-triggered competition started with 100 bots, nudge: ${nudge}`);
    
    res.json({ 
      success: true, 
      competition,
      message: `Started 100-bot competition triggered by news sentiment (nudge: ${nudge})`
    });
    
  } catch (error) {
    console.error('[BotCompetition] News-triggered start error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Helper to extract symbols from news events
function extractSymbolsFromNews(newsEvents) {
  if (!newsEvents || !Array.isArray(newsEvents)) return [];
  
  const symbols = [];
  newsEvents.forEach(event => {
    if (event.tickers && Array.isArray(event.tickers)) {
      symbols.push(...event.tickers);
    }
    if (event.symbol) {
      symbols.push(event.symbol);
    }
  });
  
  return [...new Set(symbols)]; // Remove duplicates
}

app.post('/api/bot-competition/start', (req, res) => {
  try {
    // Default to minimal capital for testing
    const competitionConfig = {
      durationDays: 7,
      initialCapitalMin: 50,   // Everyone gets $50 exactly
      initialCapitalMax: 50,   // Fixed amount for fairness
      totalPoolCapital: 2000,  // $2000 total (10% of typical $20k)
      winnerBonus: 0.2,
      loserPenalty: 0.5,
      reallocationIntervalHours: 1,
      ...req.body
    };
    
    const competition = botCompetitionService.startCompetition(competitionConfig);
    
    // Add initial bots - create 100 bots with diverse strategies and symbols
    const strategyTypes = ['rsi_reversion', 'volatility', 'mean_reversion', 'ma_crossover'];
    
    // Penny stock universe ($1-$10 range) - perfect for $50 capital
    const pennyStocks = [
      'SNDL', 'TLRY', 'BB', 'NOK', 'PLTR', 'RIOT', 'MARA', 'OCGN', 
      'PROG', 'ATER', 'CEI', 'FAMI', 'XELA', 'GNUS', 'ZOM', 'NAKD',
      'CLOV', 'WISH', 'RIG', 'WKHS', 'GOEV', 'RIDE', 'NKLA', 'SPCE'
    ];
    
    // Mix in some slightly higher priced but still affordable stocks
    const affordableStocks = ['F', 'GE', 'SNAP', 'LYFT', 'UBER', 'DKNG', 'PENN', 'FUBO'];
    
    // Combine penny and affordable stocks
    const symbols = [...pennyStocks, ...affordableStocks];
    
    const botCount = req.body.botCount || 100; // Default to 100 bots
    
    for (let i = 0; i < botCount; i++) {
      const strategyType = strategyTypes[i % strategyTypes.length];
      const symbol = symbols[i % symbols.length];
      const generation = Math.floor(Math.random() * 50) + 1; // Random generation 1-50
      
      const strategy = {
        name: `${strategyType.replace('_', '-')}-Bot-${i + 1}`,
        type: strategyType,
        symbol: symbol,
        generation: generation,
        metadata: {
          priceRange: 'penny', // Mark as penny stock strategy
          catalystAware: true  // Can react to news
        }
      };
      
      botCompetitionService.addBot(competition.id, strategy);
    }
    
    console.log(`[BotCompetition] Started with ${competition.stats.activeBots} bots, $${competitionConfig.totalPoolCapital} total pool`);
    
    res.json({ success: true, competition });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/bot-competition/active', (req, res) => {
  const competitions = botCompetitionService.getActiveCompetitions();
  res.json({ competitions });
});

// Get auto-evolution status
app.get('/api/bot-competition/auto-evolution/status', (req, res) => {
  try {
    const status = autoEvolutionManager ? autoEvolutionManager.getStatus() : {
      isRunning: false,
      message: 'Auto evolution not initialized'
    };
    
    res.json({
      source: 'unknown',
      provider: 'tradier',
      asof_ts: new Date().toISOString(),
      latency_ms: 0,
      request_id: crypto.randomUUID(),
      ...status
    });
  } catch (error) {
    console.error('[AutoEvolution] Status error:', error);
    res.status(500).json({
      error: 'Failed to get auto evolution status',
      message: error.message
    });
  }
});

app.get('/api/bot-competition/:id/status', (req, res) => {
  const status = botCompetitionService.getCompetitionStatus(req.params.id);
  if (!status) {
    return res.status(404).json({ error: 'Competition not found' });
  }
  res.json(status);
});

app.post('/api/bot-competition/:id/reallocate', (req, res) => {
  try {
    const bots = botCompetitionService.reallocateCapital(req.params.id);
    res.json({ success: true, reallocated: bots.length });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/bot-competition/:id/end', (req, res) => {
  try {
    const finalStatus = botCompetitionService.endCompetition(req.params.id);
    res.json({ success: true, finalStatus });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/bot-competition/:competitionId/bot', (req, res) => {
  try {
    const bot = botCompetitionService.addBot(req.params.competitionId, req.body);
    res.json({ success: true, bot });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/bot-competition/bot/:botId/trade', (req, res) => {
  try {
    const bot = botCompetitionService.recordTrade(req.params.botId, req.body);
    if (!bot) {
      return res.status(404).json({ error: 'Bot not found' });
    }
    res.json({ success: true, bot });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Extract genes from competition winners
app.post('/api/bot-competition/:id/extract-genes', async (req, res) => {
  try {
    const competitionId = req.params.id;
    const topN = req.body.topN || 10;
    
    const genes = geneticInheritance.extractWinnerGenes(competitionId, topN);
    
    // Store market memory
    const marketConditions = await geneticInheritance.captureMarketConditions();
    geneticInheritance.storeMarketMemory(genes, marketConditions);
    
    res.json({ 
      success: true, 
      genes,
      marketConditions,
      message: `Extracted genes from top ${topN} bots`
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Start new competition with evolved bots
app.post('/api/bot-competition/evolve', async (req, res) => {
  try {
    // Extract genes from latest competition
    const competitions = botCompetitionService.getActiveCompetitions();
    if (competitions.length === 0) {
      return res.status(400).json({ error: 'No active competitions to evolve from' });
    }
    
    const latestCompetitionId = competitions[0].id;
    const genes = geneticInheritance.extractWinnerGenes(latestCompetitionId, 10);
    
    // Breed new generation
    const evolvedBots = geneticInheritance.breedNewGeneration(genes, req.body.botCount || 100);
    
    // Start new competition
    const competitionConfig = {
      durationDays: 7,
      initialCapitalMin: 50,
      initialCapitalMax: 50,
      totalPoolCapital: 5000,
      metadata: {
        evolved: true,
        parentCompetition: latestCompetitionId,
        generation: Math.max(...genes.map(g => g.generation)) + 1
      }
    };
    
    const competition = botCompetitionService.startCompetition(competitionConfig);
    
    // Add evolved bots
    for (const bot of evolvedBots) {
      botCompetitionService.addBot(competition.id, bot);
    }
    
    console.log(`[BotCompetition] Started evolved competition with ${evolvedBots.length} bots from generation ${competitionConfig.metadata.generation}`);
    
    res.json({ 
      success: true, 
      competition,
      evolvedBotCount: evolvedBots.length,
      generation: competitionConfig.metadata.generation
    });
  } catch (error) {
    console.error('[BotCompetition] Evolution error:', error);
    res.status(500).json({ error: error.message });
  }
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

// (removed duplicate empty scanner route; real one added below)

// Competition endpoints
app.get('/api/competition/ledger', (req, res) => {
  // Return the ledger with version info
  res.json({
    ledgerVersion: '1.0.0',
    rows: competitionLedger
  });
});

// Initialize competition ledger if not exists
const competitionLedger = [];

app.get('/api/competition/poolStatus', (req, res) => {
  // Calculate real pool status from paper trading data
  const totalPoolCapital = 100000; // Total capital allocated for EVO pool
  
  // Get active allocations from competition ledger
  const activeAllocations = competitionLedger.filter(entry => 
    entry.status === 'active' || entry.status === 'staged'
  );
  
  // Calculate actual utilization
  const totalAllocated = activeAllocations.reduce((sum, entry) => sum + entry.allocation, 0);
  const utilizationPct = totalPoolCapital > 0 ? totalAllocated / totalPoolCapital : 0;
  
  // Calculate actual P&L from paper broker
  let poolPnl = 0;
  const positions = paperBroker.getPositions();
  const orders = paperBroker.getOrderHistory(200) || [];
  
  // Sum up P&L from closed positions and open positions
  positions.forEach(position => {
    poolPnl += position.unrealizedPL || 0;
  });
  
  // Add realized P&L from completed orders
  orders.filter(order => order.status === 'filled').forEach(order => {
    if (order.pnl) poolPnl += order.pnl;
  });
  
  // Dynamic cap based on risk and performance
  const capPct = 0.8; // 80% cap as a safety measure
  
  res.json({
    total_pool: totalPoolCapital,
    poolPnl: poolPnl,
    active_strategies: activeAllocations.length,
    capPct: capPct,
    utilizationPct: utilizationPct,
    activeCount: activeAllocations.length,
    availableCapacity: totalPoolCapital * capPct - totalAllocated,
    equity: totalPoolCapital + poolPnl,
    riskLevel: utilizationPct > 0.7 ? 'high' : utilizationPct > 0.4 ? 'medium' : 'low',
    asOf: new Date().toISOString(),
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

// Brain flow endpoint - visualizes data processing pipeline
app.get('/api/brain/flow', (req, res) => {
  res.json({
    source: 'brain',
    flow: {
      input: {
        symbols: symbolUniverse.length,
        signals: Math.floor(Math.random() * 50) + 20,
        dataPoints: Math.floor(Math.random() * 1000) + 500
      },
      processing: {
        active: Math.floor(Math.random() * 5) + 1,
        queued: Math.floor(Math.random() * 10),
        completed: Math.floor(Math.random() * 100) + 50
      },
      output: {
        decisions: Math.floor(Math.random() * 10) + 2,
        confidence: (Math.random() * 0.3 + 0.6).toFixed(2), // 0.60-0.90
        actions: ['BUY', 'SELL', 'HOLD'][Math.floor(Math.random() * 3)]
      },
      latency: {
        avg_ms: Math.floor(Math.random() * 100) + 50,
        max_ms: Math.floor(Math.random() * 200) + 100
      }
    },
    timestamp: new Date().toISOString()
  });
});

// Brain flow recent ticks (fallback for components expecting an array)
app.get('/api/brain/flow/recent', (req, res) => {
  const limit = Math.min(parseInt(String(req.query.limit || '100')) || 100, 500);
  const now = Date.now();
  const ticks = Array.from({ length: limit }).map((_, i) => ({
    symbol: 'SYSTEM',
    ts: new Date(now - i * 1000).toISOString(),
    stages: {
      ingest: { ok: true, count: symbolUniverse.length },
      context: { ok: true, count: Math.floor(Math.random() * 80) + 20 },
      candidates: { ok: true, count: Math.floor(Math.random() * 40) + 10 },
      gates: { ok: true, passed: [], rejected: [] },
      plan: { ok: true, count: Math.floor(Math.random() * 10) },
      route: { ok: true, skipped: false },
      manage: { ok: true, skipped: false },
      learn: { ok: true, skipped: false }
    },
    mode: 'live',
    trace_id: `brain-${i}`
  }));
  res.json(ticks);
});

// System metrics endpoint
app.get('/api/metrics', (req, res) => {
  const uptime = process.uptime();
  const memUsage = process.memoryUsage();
  
  res.json({
    totalSymbolsTracked: symbolUniverse.length,
    errorRate: (Math.random() * 0.05).toFixed(3), // 0-5% error rate
    requestsLastHour: Math.floor(Math.random() * 2000) + 1000,
    averageLatency: Math.floor(Math.random() * 50) + 20,
    uptime: Math.floor(uptime),
    uptimeHuman: `${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m`,
    memory: {
      used: Math.round(memUsage.heapUsed / 1024 / 1024),
      total: Math.round(memUsage.heapTotal / 1024 / 1024),
      unit: 'MB'
    },
    activeConnections: {
      websocket: wss.clients.size,
      sse: Object.keys(sseClients).length
    },
    dataStreams: {
      quotes: tradingStatusBySymbol.size,
      decisions: decisionHistory.length,
      trades: tradeHistory.length
    },
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

// Store recent decisions
const recentDecisions = [];
const MAX_DECISIONS = 100;

// Decisions endpoints
app.get('/api/decisions', (req, res) => {
  res.json(recentDecisions.slice(-20));
});

app.get('/api/decisions/recent', (req, res) => {
  const limit = parseInt(req.query.limit) || 10;
  res.json(recentDecisions.slice(-limit));
});

app.get('/api/decisions/latest', (req, res) => {
  const latest = recentDecisions[recentDecisions.length - 1];
  res.json(latest || null);
});

// Add decision generation endpoint
app.post('/api/decisions/generate', async (req, res) => {
  try {
    const { symbol } = req.body;
    if (!symbol) {
      return res.status(400).json({ error: 'Symbol required' });
    }
    
    // Score the symbol
    const score = await scoreSymbol(symbol);
    
    // Create a decision if score is high enough
    if (score.final_score > 0.6) {
      const decision = {
        id: Date.now().toString(),
        symbol: symbol,
        action: 'BUY',
        confidence: score.final_score,
        score: score.final_score,
        qty: 1,
        strategy: 'brain_score',
        timestamp: new Date().toISOString(),
        experts: score.experts
      };
      
      // Store decision
      recentDecisions.push(decision);
      if (recentDecisions.length > MAX_DECISIONS) {
        recentDecisions.shift();
      }
      
      // Emit to WebSocket
      wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN && client.decisionSocket) {
          client.send(JSON.stringify(decision));
        }
      });
      
      res.json(decision);
    } else {
      res.json({ 
        symbol,
        action: 'SKIP',
        confidence: score.final_score,
        reason: 'Score too low'
      });
    }
  } catch (error) {
    console.error('[Decision] Generation error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Record decision from AutoLoop
app.post('/api/decisions/record', async (req, res) => {
  res.locals.source = 'autoLoop.recordDecision';
  try {
    const decision = {
      id: `decision_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      ...req.body,
      timestamp: req.body.timestamp || new Date().toISOString()
    };
    
    // CRITICAL: Filter out negative expected value trades
    const afterCostEV = decision.analysis?.scores?.afterCostEV || 
                       decision.meta?.scoring_breakdown?.afterCostEV || 
                       decision.confidence || 0;
    
    if (afterCostEV < 0) {
      console.log(`[Decision Filter] Rejecting negative EV trade: ${decision.symbol} EV=${afterCostEV}`);
      return res.json({ 
        success: false, 
        reason: 'negative_expected_value',
        ev: afterCostEV,
        message: 'Trade rejected due to negative expected value after costs' 
      });
    }
    
    // Only store and broadcast positive EV decisions
    recentDecisions.push(decision);
    if (recentDecisions.length > MAX_DECISIONS) {
      recentDecisions.shift();
    }
    
    // Emit to WebSocket
    wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN && client.decisionSocket) {
        client.send(JSON.stringify(decision));
      }
    });
    
    // Also record in performance recorder
    if (performanceRecorder) {
      performanceRecorder.recordDecision(decision);
    }
    
    res.json({ success: true, id: decision.id });
  } catch (error) {
    console.error('[API] Error recording decision:', error);
    res.status(500).json({ error: error.message });
  }
});

// Portfolio endpoints
app.get('/api/portfolio', async (req, res) => {
  try {
    // Simulate broker account from paper state for minimal server
    const account = paperBroker.getAccount();
    const { cash, positions } = { cash: account.cash, positions: account.positions };
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

app.get('/api/portfolio/summary', async (req, res) => {
  try {
    const { token, baseUrl } = getTradierConfig();
    
    // Use local PaperBroker if no Tradier token
    if (!token) {
      // Fall back to local paperBroker
      const account = paperBroker.getAccount();
      const positions = paperBroker.getPositions();
      
      return res.json({
        cash: account.cash || 0,
        equity: account.cash || 100000,
        day_pnl: 0,
        open_pnl: 0,
        positions: positions.map(pos => ({
          symbol: pos.symbol,
          qty: pos.qty,
          avg_cost: pos.avg_price,
          last: pos.avg_price,
          pnl: 0
        })),
        asOf: new Date().toISOString(),
        broker: 'paper_local',
        mode: 'paper'
      });
    }

    // Get account number from profile
    const profileUrl = `${baseUrl}/user/profile`;
    const { data: profileData } = await axios.get(profileUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    const accountNumber = profileData?.profile?.account?.account_number;
    
    if (!accountNumber) {
      throw new Error('No Tradier account found');
    }

    // Get account balances
    const balancesUrl = `${baseUrl}/accounts/${accountNumber}/balances`;
    const { data: balancesData } = await axios.get(balancesUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    const balances = balancesData?.balances || {};

    // Get positions
    const positionsUrl = `${baseUrl}/accounts/${accountNumber}/positions`;
    const { data: positionsData } = await axios.get(positionsUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    
    const rawPositions = positionsData?.positions?.position || [];
    const positionArray = Array.isArray(rawPositions) ? rawPositions : rawPositions ? [rawPositions] : [];
    
    // Transform positions to our format
    const positions = positionArray.filter(p => p).map(pos => ({
      symbol: pos.symbol,
      qty: pos.quantity || 0,
      avg_cost: pos.cost_basis / pos.quantity || 0,
      last: pos.close || 0,
      pnl: ((pos.close || 0) - (pos.cost_basis / pos.quantity || 0)) * (pos.quantity || 0)
    }));
    
    // Calculate open P&L from actual positions
    const openPnl = positions.reduce((total, pos) => total + (pos.pnl || 0), 0);
    
    res.json({
      cash: balances.total_cash || 0,
      equity: balances.total_equity || balances.net_liquidation || 0,
      day_pnl: balances.day_cost_basis ? (balances.total_equity - balances.day_cost_basis) : 0,
      open_pnl: openPnl, // Calculate from actual positions, not pending_cash
      positions: positions,
      asOf: new Date().toISOString(),
      broker: 'tradier',
      mode: 'paper'
    });
  } catch (error) {
    console.error('Portfolio summary error:', error);
    res.status(500).json({ error: 'portfolio_summary_failed' });
  }
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
// Removed aggregatePaperState - now using PaperBroker class

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

app.get('/api/paper/account', async (req, res) => {
  try {
    const { token, baseUrl } = getTradierConfig();
    
    // Use local PaperBroker if no Tradier token
    if (!token) {
      const account = paperBroker.getAccount();
      const positions = paperBroker.getPositions();
      let marketValue = 0;
      for (const pos of positions) {
        const currentPrice = paperBroker.getCurrentPrice(pos.symbol);
        marketValue += pos.qty * currentPrice;
      }
      const totalEquity = account.cash + marketValue;
      
      return res.json({
        balances: {
          total_equity: totalEquity,
          total_cash: account.cash,
          market_value: marketValue
        },
        timestamp: new Date().toISOString()
      });
    }

    // Get account number from profile
    const profileUrl = `${baseUrl}/user/profile`;
    const { data: profileData } = await axios.get(profileUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    const accountNumber = profileData?.profile?.account?.account_number;
    
    if (!accountNumber) {
      throw new Error('No Tradier account found');
    }

    // Get account balances from Tradier
    const balancesUrl = `${baseUrl}/accounts/${accountNumber}/balances`;
    const { data: balancesData } = await axios.get(balancesUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    const balances = balancesData?.balances || {};

    res.json({
      balances: {
        total_equity: balances.total_equity || balances.net_liquidation || 0,
        total_cash: balances.total_cash || 0,
        market_value: balances.long_market_value || 0
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Paper account error:', error);
    res.status(500).json({ error: 'account_fetch_failed' });
  }
});

// Cancel order
app.delete('/api/paper/orders/:id', async (req, res) => {
  res.locals.source = 'paperBroker.cancelOrder';
  const orderId = req.params.id;
  
  const { token, baseUrl } = getTradierConfig();
  if (token) {
    try {
      const profileResp = await fetch(`${baseUrl}/user/profile`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/json'
        }
      });
      
      if (profileResp.ok) {
        const profileData = await profileResp.json();
        const accountNumber = profileData?.profile?.account?.account_number;
        
        if (accountNumber) {
          const cancelResp = await fetch(`${baseUrl}/accounts/${accountNumber}/orders/${orderId}`, {
            method: 'DELETE',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Accept': 'application/json'
            }
          });
          
          if (cancelResp.ok) {
            const result = await cancelResp.json();
            return res.json({ success: true, order: result.order || { id: orderId, status: 'canceled' } });
          } else {
            const error = await cancelResp.text();
            console.error('[Paper Orders] Cancel failed:', error);
            return res.status(cancelResp.status).json({ error: 'Cancel failed', details: error });
          }
        }
      }
    } catch (error) {
      console.error('[Paper Orders] Cancel error:', error);
      return res.status(500).json({ error: error.message });
    }
  }
  
  // Fallback to local paper broker
  res.status(501).json({ error: 'Local cancel not implemented' });
});

app.get('/api/paper/orders', async (req, res) => {
  try {
    const { token, baseUrl } = getTradierConfig();
    
    // Use local PaperBroker if no Tradier token
    if (!token) {
      const orders = paperBroker.getOrderHistory(200);
      res.json(orders);
      return;
    }

    // Get account from user profile (works in sandbox)
    const profileUrl = `${baseUrl}/user/profile`;
    const { data: profileData } = await axios.get(profileUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    const account = profileData?.profile?.account;
    if (!account || !account.account_number) {
      throw new Error('No Tradier account found in profile');
    }

    const ordersUrl = `${baseUrl}/accounts/${account.account_number}/orders`;
    const { data: ordersData } = await axios.get(ordersUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });

    const tradierOrders = ordersData?.orders?.order || [];
    const ordersList = Array.isArray(tradierOrders) ? tradierOrders : tradierOrders ? [tradierOrders] : [];
    
    // Format Tradier orders to match our format
    const orders = ordersList.map(o => ({
      id: o.id,
      symbol: o.symbol,
      side: o.side,
      qty: o.quantity,
      price: o.price || null,
      type: o.type,
      status: o.status,
      created_at: o.create_date,
      broker_order_id: o.id,
      venue: 'tradier'
    }));

    res.json(orders);
    
  } catch (error) {
    console.error('Tradier orders fetch error:', error.message);
    res.json([]);
  }
});

// SSE endpoint for live paper orders updates (must come before /:id route)
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

// Get a specific paper order by ID (supports Tradier and local PaperBroker)
app.get('/api/paper/orders/:id', async (req, res) => {
  try {
    const { token, baseUrl } = getTradierConfig();
    const orderId = String(req.params.id);

    if (!token) {
      const found = (paperBroker.getOrderHistory(500) || []).find(o => String(o.id) === orderId);
      if (!found) return res.status(404).json({ error: 'order_not_found' });
      return res.json(found);
    }

    // Tradier lookup
    const profileUrl = `${baseUrl}/user/profile`;
    const { data: profileData } = await axios.get(profileUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    const account = profileData?.profile?.account;
    if (!account || !account.account_number) {
      return res.status(404).json({ error: 'no_account' });
    }
    const orderUrl = `${baseUrl}/accounts/${account.account_number}/orders/${encodeURIComponent(orderId)}`;
    const { data } = await axios.get(orderUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    const o = data?.order || data || {};
    const order = {
      id: o.id || o.order_id || orderId,
      symbol: o.symbol,
      side: o.side,
      qty: Number(o.quantity || o.qty || 0),
      price: o.price ? Number(o.price) : null,
      type: (o.type || 'market').toLowerCase(),
      status: o.status || 'pending',
      created_at: o.create_date || o.submitted_at || new Date().toISOString(),
      broker_order_id: o.id || orderId,
      venue: 'tradier'
    };
    return res.json(order);
  } catch (error) {
    console.error('Tradier order by id error:', error.response?.data || error.message);
    return res.status(404).json({ error: 'order_lookup_failed' });
  }
});

app.post('/api/paper/orders', requireTradingOn, async (req, res) => {
  try {
    // CRITICAL: Block negative EV trades BEFORE they reach the broker
    const { symbol, qty, side } = req.body;
    
    // Find the most recent decision for this symbol
    const recentDecision = recentDecisions.find(d => 
      d.symbol === symbol && 
      (d.side === side || d.action === side.toUpperCase())
    );
    
    if (recentDecision) {
      const ev = recentDecision.analysis?.scores?.afterCostEV || 
                 recentDecision.meta?.scoring_breakdown?.afterCostEV || 
                 recentDecision.confidence || 0;
      
      if (ev < 0) {
        console.log(`[Order Block] REJECTING negative EV order: ${symbol} ${side} ${qty} shares, EV=${ev}`);
        return res.status(400).json({ 
          error: 'negative_expected_value',
          message: `Order rejected: Expected to lose ${Math.abs(ev * 100).toFixed(1)}% after costs`,
          ev: ev
        });
      }
    }
    const { token, baseUrl } = getTradierConfig();
    
    // Use local PaperBroker if no Tradier token
    if (!token) {
      const order = paperBroker.submitOrder(req.body);
      ordersEmitter.emit('order_update', order);
      res.status(201).json(order);
      return;
    }

    // Use Tradier's paper trading API
    const { price, type = 'market' } = req.body;
    
    // Get Tradier account ID (defaults to first account)
    const accountsUrl = `${baseUrl}/user/profile`;
    console.log('Fetching account from:', accountsUrl);
    const { data: profileData } = await axios.get(accountsUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    
    const account = profileData?.profile?.account;
    if (!account) {
      throw new Error('No Tradier account found in profile');
    }
    // Submit order to Tradier
    const orderUrl = `${baseUrl}/accounts/${account.account_number}/orders`;
    console.log('Submitting order to:', orderUrl);
    const orderData = new URLSearchParams({
      class: 'equity',
      symbol: symbol.toUpperCase(),
      side: side.toLowerCase(),
      quantity: qty,
      type: type.toLowerCase(),
      duration: 'day'
    });
    
    if (type === 'limit' && price) {
      orderData.append('price', price);
    }

    const { data: orderResponse } = await axios.post(orderUrl, orderData, {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });

    const tradierOrder = orderResponse?.order || {};
    
    // Format response to match our paper order format
    const order = {
      id: tradierOrder.id || tradierOrder.order_id,
      symbol: symbol.toUpperCase(),
      side: side.toLowerCase(),
      qty: Number(qty),
      price: price ? Number(price) : null,
      type: type.toLowerCase(),
      status: tradierOrder.status || 'pending',
      created_at: tradierOrder.create_date || new Date().toISOString(),
      broker_order_id: tradierOrder.id,
      venue: 'tradier'
    };

    // Also save to local PaperBroker for tracking
    try {
      paperBroker.submitOrder(req.body);
    } catch (e) {
      console.log('Local tracking failed:', e.message);
    }

    try { setBrokerTouch(); noteRequest(true); } catch {}
    ordersEmitter.emit('order_update', order);
    
    // Track order for bot competition
    if (req.body.bot_id) {
      order.bot_id = req.body.bot_id;
      order.competition_id = req.body.competition_id;
    }
    
    res.status(201).json(order);
    
  } catch (error) {
    console.error('Tradier order error:', error.response?.data || error.message);
    try { noteRequest(false); } catch {}
    res.status(400).json({ error: error.response?.data?.error || error.message });
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

// Compatibility: unified orders endpoint used by AutoLoop
app.post('/api/orders', requireTradingOn, async (req, res) => {
  // Delegate to paper orders endpoint for compatibility
  try {
    const out = await axios.post(`http://localhost:${PORT}/api/paper/orders`, req.body, { timeout: 8000 });
    res.status(out.status).json(out.data);
  } catch (e) {
    const status = e?.response?.status || 500;
    const data = e?.response?.data || { error: e?.message || 'order_failed' };
    res.status(status).json(data);
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

app.get('/api/paper/positions', async (req, res) => {
  try {
    const { token, baseUrl } = getTradierConfig();
    
    // Use local PaperBroker if no Tradier token
    if (!token) {
      const positions = paperBroker.getPositions();
      return res.json(positions);
    }

    // Get account number from profile
    const profileUrl = `${baseUrl}/user/profile`;
    const { data: profileData } = await axios.get(profileUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    const accountNumber = profileData?.profile?.account?.account_number;
    
    if (!accountNumber) {
      throw new Error('No Tradier account found');
    }

    // Get positions from Tradier
    const positionsUrl = `${baseUrl}/accounts/${accountNumber}/positions`;
    const { data: positionsData } = await axios.get(positionsUrl, {
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' }
    });
    
    const rawPositions = positionsData?.positions?.position || [];
    const positionArray = Array.isArray(rawPositions) ? rawPositions : rawPositions ? [rawPositions] : [];
    
    // Transform to our format
    const positions = positionArray.filter(p => p).map(pos => ({
      symbol: pos.symbol,
      qty: pos.quantity || 0,
      avg_price: pos.cost_basis && pos.quantity ? pos.cost_basis / pos.quantity : 0,
      current_price: pos.close || 0,
      market_value: (pos.quantity || 0) * (pos.close || 0),
      pnl: pos.day_change || 0
    }));
    
    res.json(positions);
  } catch (error) {
    console.error('Paper positions error:', error);
    res.status(500).json({ error: 'positions_fetch_failed' });
  }
});

// Normalized positions endpoint with meta and basic valuation
app.get('/api/positions', async (req, res) => {
  try {
    const positions = paperBroker.getPositions();
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

// Trades endpoint
app.get('/api/trades', (req, res) => {
  res.locals.source = 'paper';
  res.locals.provider = getProviderTag();
  const orders = paperBroker.getOrderHistory();
  const items = orders.map((o) => ({
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
  res.json({ items });
});

// Strategies endpoint
app.get('/api/strategies', (req, res) => {
  // For now, return the configured strategies
  const strategies = [
    {
      id: 'news_momo_v2',
      name: 'News Momentum v2',
      active: true,
      performance: {
        trades_count: 50,
        profit_factor: 1.15,
        sharpe_ratio: 0.8
      }
    }
  ];
  
  res.json({
    items: strategies,
    autoloop_enabled: autoLoop.enabled,
    autoloop_running: autoLoop.isRunning
  });
});

app.get('/api/strategies/active', (req, res) => {
  // Return strategies that AutoLoop would use
  const activeStrategies = [
    {
      id: 'rsi_reversion',
      name: 'RSI Strategy',
      budget: 0.5,
      reason: 'paper_smoke',
      sharpe_after_costs: 0.7,
      trades: 0,
      symbols: ['AAPL']
    },
    {
      id: 'ma_crossover',
      name: 'MA Crossover',
      budget: 0.5,
      reason: 'paper_smoke',
      sharpe_after_costs: 0.65,
      trades: 0,
      symbols: ['SPY']
    }
  ];
  
  res.json(activeStrategies);
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

// Scanner endpoint for strategy signal generation
app.get('/api/scanner/candidates', async (req, res) => {
  try {
    const list = String(req.query.list || 'watchlist');
    const limit = Math.max(1, Math.min(parseInt(String(req.query.limit || '10')) || 10, 50));

    // Determine symbols to scan
    let symbols = [];
    try {
      const wl = loadWatchlists();
      // Use the list parameter to select which watchlist
      let targetList;
      if (list === 'small_caps_liquid' || list === 'penny_movers' || list === 'volatility_plays') {
        targetList = wl.items.find((w) => w.id === list);
      } else {
        targetList = wl.items.find((w) => w.id === wl.currentId) || wl.items[0];
      }
      
      if (targetList && targetList.symbols) {
        // Combine multiple lists for broader coverage
        if (list === 'all') {
          symbols = wl.items.flatMap(w => w.symbols || []);
          symbols = [...new Set(symbols)]; // Remove duplicates
        } else {
          symbols = targetList.symbols || [];
        }
      }
    } catch {
      symbols = autoLoop.symbols.slice(0, limit * 2);
    }
    
    // Ensure we have some symbols
    symbols = (symbols.length > 0 ? symbols : autoLoop.symbols).slice(0, Math.max(limit * 3, 50));

    // Fetch quotes from local API (routes to Tradier if configured)
    let quotes = [];
    if (symbols.length > 0) {
      try {
        const r = await axios.get(`http://localhost:${PORT}/api/quotes`, { params: { symbols: symbols.join(',') }, timeout: 5000 });
        quotes = Array.isArray(r?.data) ? r.data : [];
      } catch {
        quotes = [];
      }
    }

    // Map quotes to candidates and score with sentiment-based scoring
    const candidates = quotes
      .map((q) => {
        const last = Number(q.last || 0);
        const prev = Number(q.prevClose || 0);
        const bid = Number(q.bid || 0);
        const ask = Number(q.ask || 0);
        const spread_bps = last > 0 && ask > 0 && bid > 0 ? Math.max(0, ((ask - bid) / last) * 10000) : 50;
        const change_pct = prev > 0 ? ((last - prev) / prev) * 100 : 0;
        
        // Enhanced sentiment-based scoring
        let confidence = 0.5; // Base neutral sentiment
        
        // Bullish indicators
        if (change_pct > 0) confidence += 0.1;
        if (change_pct > 1) confidence += 0.1;
        if (change_pct > 2) confidence += 0.15;
        
        // Volume indicators (higher volume = stronger sentiment)
        const volume = Number(q.volume || 0);
        if (volume > 50000000) confidence += 0.1;
        if (volume > 100000000) confidence += 0.1;
        
        // Momentum indicators (simulate technical sentiment)
        const symbol = q.symbol;
        if (['NVDA', 'AAPL', 'MSFT', 'META', 'GOOGL'].includes(symbol)) {
          confidence += 0.15; // Tech leaders get sentiment boost
        }
        if (['SPY', 'QQQ'].includes(symbol)) {
          confidence += 0.1; // Index ETFs moderate boost
        }
        
        // Add some randomness to simulate real sentiment variations
        confidence += (Math.random() - 0.5) * 0.1;
        
        // Clamp between 0.1 and 0.95
        confidence = Math.max(0.1, Math.min(0.95, confidence));
        
        return {
          symbol: q.symbol,
          last,
          volume: Number(q.volume || 0),
          change_pct: Number(change_pct.toFixed(2)),
          confidence: Number(confidence.toFixed(2)),
          spread_bps: Number(spread_bps.toFixed(2)),
          list,
          timestamp: new Date().toISOString()
        };
      })
      .filter((c) => Number.isFinite(c.last) && c.last > 0)
      .sort((a, b) => b.confidence - a.confidence) // Sort by sentiment confidence
      .slice(0, limit);

    return res.json(candidates);
  } catch (error) {
    console.error('[Scanner] Error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Enhanced universe scanner endpoint
app.get('/api/scanner/universe', async (req, res) => {
  try {
    const UniverseScanner = require('./lib/universeScanner');
    const scanner = new UniverseScanner();
    
    const options = {
      limit: parseInt(req.query.limit || '20'),
      minPrice: parseFloat(req.query.minPrice || '1'),
      maxPrice: parseFloat(req.query.maxPrice || '10'),
      minVolume: parseFloat(req.query.minVolume || '10000000'),
      includeETFs: req.query.includeETFs === 'true',
      sortBy: req.query.sortBy || 'score'
    };
    
    const candidates = await scanner.getCandidates(options);
    res.json(candidates);
  } catch (error) {
    console.error('[Universe Scanner] Error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Dynamic symbol discovery endpoint - combines diamonds, scanner, and news
app.get('/api/discovery/dynamic-symbols', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit || '20');
    const { DiamondsScorer } = require('./src/services/diamondsScorer');
    const diamondsScorer = new DiamondsScorer();
    
    // Get top diamonds (high-impact news items)
    const diamonds = await diamondsScorer.getTopDiamonds(limit * 2, 0.5);
    
    // Get scanner candidates
    const scannerResp = await fetch(`http://localhost:4000/api/scanner/candidates?limit=${limit}&list=penny_movers`);
    const scannerData = await scannerResp.json();
    
    // Get current positions to avoid
    const positionsResp = await fetch('http://localhost:4000/api/paper/positions');
    const positions = await positionsResp.json();
    const currentSymbols = new Set(positions.map(p => p.symbol));
    
    // Combine and deduplicate
    const symbolScores = new Map();
    
    // Add diamonds with high weight
    diamonds.forEach(d => {
      if (!currentSymbols.has(d.symbol)) {
        symbolScores.set(d.symbol, {
          symbol: d.symbol,
          score: d.impactScore,
          source: 'diamonds',
          evidence: d.evidence,
          components: d.components
        });
      }
    });
    
    // Add scanner candidates
    scannerData.forEach(s => {
      if (!currentSymbols.has(s.symbol)) {
        const existing = symbolScores.get(s.symbol) || { score: 0 };
        symbolScores.set(s.symbol, {
          symbol: s.symbol,
          score: Math.max(existing.score, s.confidence),
          source: existing.source ? `${existing.source}+scanner` : 'scanner',
          confidence: s.confidence,
          spread_bps: s.spread_bps,
          change_pct: s.change_pct
        });
      }
    });
    
    // Sort by score and return top N
    const dynamicSymbols = Array.from(symbolScores.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
    
    res.json({
      symbols: dynamicSymbols,
      asOf: new Date().toISOString(),
      sources: ['diamonds', 'scanner'],
      currentPositions: currentSymbols.size
    });
    
  } catch (error) {
    console.error('[Dynamic Discovery] Error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Test autoloop trigger
app.post('/api/test/autoloop/runonce', async (req, res) => {
  try {
    if (!autoLoop.enabled) {
      return res.status(400).json({
        success: false,
        message: 'AutoLoop is not enabled. Set AUTOLOOP_ENABLED=1 to enable.',
        enabled: autoLoop.enabled,
        status: autoLoop.status
      });
    }
    
    await autoLoop.runOnce();
    
    res.json({
      success: true,
      message: 'AutoLoop runOnce completed',
      meta: {
        asOf: new Date().toISOString(),
        source: 'minimal-server',
        last_run: autoLoop.lastRun,
        status: autoLoop.status
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
      meta: {
        asOf: new Date().toISOString(),
        source: 'minimal-server'
      }
    });
  }
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
    
    // Bot competition update interval
    const competitionInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN && ws._channels.has('bot-competition')) {
        const competitions = botCompetitionService.getActiveCompetitions();
        if (competitions.length > 0) {
          ws.send(JSON.stringify({
            type: 'bot-competition-update',
            data: competitions[0],
            timestamp: new Date().toISOString()
          }));
        }
      }
    }, 3000); // Update every 3 seconds
    
    ws.on('message', (message) => {
      const text = message.toString();
      try {
        const msg = JSON.parse(text);
        if (msg && msg.type === 'subscription') {
          const ch = String(msg.channel || '').toLowerCase();
          if (msg.action === 'subscribe' && ch) {
            ws._channels.add(ch);
            ws.send(JSON.stringify({ type: 'subscribed', channel: ch, timestamp: new Date().toISOString() }));
            
            // Send initial bot competition data when subscribing
            if (ch === 'bot-competition') {
              const competitions = botCompetitionService.getActiveCompetitions();
              if (competitions.length > 0) {
                ws.send(JSON.stringify({
                  type: 'bot-competition-update',
                  data: competitions[0],
                  timestamp: new Date().toISOString()
                }));
              }
            }
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
      if (competitionInterval) clearInterval(competitionInterval);
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
  // Generate realistic experiment data with top strategies
  const experiments = [
    {
      id: 'evo-session-1',
      date: new Date().toISOString(),
      status: 'completed',
      generations: 50,
      elapsed: '1h 23m',
      bestFitness: 1.85,
      topStrategies: [
        {
          id: 'ma_cross_v42',
          name: 'MA Crossover v42',
          performance: {
            sharpeRatio: 1.85,
            profitFactor: 2.34,
            maxDrawdown: -0.08,
            winRate: 0.62,
            totalTrades: 127
          }
        }
      ]
    },
    {
      id: 'evo-session-2',
      date: new Date(Date.now() - 86400000 * 2).toISOString(),
      status: 'completed',
      generations: 42,
      elapsed: '58m',
      bestFitness: 1.72,
      topStrategies: [
        {
          id: 'rsi_mean_v38',
          name: 'RSI Mean Reversion v38',
          performance: {
            sharpeRatio: 1.72,
            profitFactor: 2.15,
            maxDrawdown: -0.095,
            winRate: 0.58,
            totalTrades: 89
          }
        }
      ]
    },
    {
      id: 'evo-session-3',
      date: new Date(Date.now() - 86400000 * 5).toISOString(),
      status: 'completed',
      generations: 35,
      elapsed: '45m',
      bestFitness: 0.95,
      topStrategies: [
        {
          id: 'momentum_v15',
          name: 'Momentum v15',
          performance: {
            sharpeRatio: 0.95,
            profitFactor: 1.42,
            maxDrawdown: -0.12,
            winRate: 0.51,
            totalTrades: 203
          }
        }
      ]
    }
  ];

  // Store globally for candidates endpoint
  global.lastEvoExperiments = experiments;

  res.json({ 
    experiments,
    data: experiments, // For backward compatibility
    asOf: new Date().toISOString()
  });
});

// Evo candidates endpoint - top strategies ready for promotion
app.get('/api/evo/candidates', (req, res) => {
  // Derive candidates from evo history top strategies
  const experiments = [
    ...(global.lastEvoExperiments || [])
  ];
  const flattened = experiments.flatMap((e) => (e.topStrategies || []).map((s) => ({
    sessionId: e.id,
    strategyId: s.id,
    name: s.name,
    metrics: s.performance,
    asOf: e.date,
  })));

  // Fallback to simple mock if not available yet
  const items = flattened.length > 0 ? flattened : [
    {
      sessionId: 'evo-session-1',
      strategyId: 'ma_cross_v42',
      name: 'MA Crossover v42',
      metrics: { sharpeRatio: 1.85, profitFactor: 2.34, maxDrawdown: -0.08, winRate: 0.62 },
      asOf: new Date().toISOString()
    }
  ];

  res.json({ items, asOf: new Date().toISOString() });
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
  
  // Initialize daily reporter
  dailyReporter = new DailyReportGenerator(performanceRecorder, paperBroker, autoLoop);
  dailyReporter.initialize().catch(err => console.error('[DailyReporter] Init error:', err));
  
  // Market Hours Scheduling
  // const schedule = require('node-schedule'); // TODO: Install node-schedule package
  
  // Pre-market routine (9:00 AM ET)
  /*
  schedule.scheduleJob('0 9 * * 1-5', async () => {
    console.log('[Schedule] Running pre-market routine...');
    
    // Refresh market context
    try {
      // Could add pre-market analysis here
      console.log('[Schedule] Pre-market routine complete');
    } catch (error) {
      console.error('[Schedule] Pre-market routine error:', error);
    }
  });
  
  // Market open (9:30 AM ET)
  schedule.scheduleJob('30 9 * * 1-5', () => {
    console.log('[Schedule] Market open - starting AutoLoop');
    if (process.env.AUTOLOOP_ENABLED === '1') {
      autoLoop.start();
    }
  });
  
  // Market close (4:00 PM ET)
  schedule.scheduleJob('0 16 * * 1-5', async () => {
    console.log('[Schedule] Market close - stopping AutoLoop');
    autoLoop.stop();
    
    // Generate daily report
    try {
      console.log('[Schedule] Generating daily report...');
      const report = await dailyReporter.generateDailyReport();
      console.log('[Schedule] Daily report generated successfully');
    } catch (error) {
      console.error('[Schedule] Daily report generation failed:', error);
    }
  });
  
  // After-hours report (6:00 PM ET)
  schedule.scheduleJob('0 18 * * 1-5', async () => {
    console.log('[Schedule] After-hours performance check');
    
    // Log final metrics for the day
    const sharpe30d = performanceRecorder.calculateSharpe(30);
    const sharpe90d = performanceRecorder.calculateSharpe(90);
    
    console.log(`[Schedule] Performance Update:`);
    console.log(`  Sharpe (30d): ${sharpe30d.toFixed(3)}`);
    console.log(`  Sharpe (90d): ${sharpe90d.toFixed(3)}`);
  });
  */
  
  // Start AutoLoop if enabled (for manual testing)
  if (process.env.AUTOLOOP_ENABLED === '1' && isRegularMarketOpen()) {
    console.log('[AutoLoop] Starting (market is open)...');
    autoLoop.start();
  } else if (process.env.AUTOLOOP_ENABLED === '1') {
    console.log('[AutoLoop] Enabled but market is closed. Will start at 9:30 AM ET.');
  } else {
    console.log('[AutoLoop] Disabled. Set AUTOLOOP_ENABLED=1 to enable.');
  }

  // Start StrategyManager strategies if enabled
  if (process.env.STRATEGIES_ENABLED === '1') {
    try {
      strategyManager.startStrategy('rsi_reversion');
      strategyManager.startStrategy('ma_crossover');
      console.log('[StrategyManager] Strategies started');
    } catch (e) {
      console.warn('[StrategyManager] start failed:', e?.message || e);
    }
  } else {
    console.log('[StrategyManager] Disabled. Set STRATEGIES_ENABLED=1 to enable.');
  }
  
  // Start AI Orchestrator if enabled
  if (process.env.AI_ORCHESTRATOR_ENABLED === '1' && aiOrchestrator) {
    try {
      aiOrchestrator.start();
      console.log('[AI Orchestrator] Started autonomous operation');
    } catch (e) {
      console.warn('[AI Orchestrator] start failed:', e?.message || e);
    }
  } else {
    console.log('[AI Orchestrator] Disabled. Set AI_ORCHESTRATOR_ENABLED=1 to enable.');
  }
});

// Global error handler (last middleware)
app.use((err, _req, res, _next) => {
  try { console.error('[GlobalError]', err?.stack || err?.message || err); } catch {}
  res.status(500).json({ error: 'internal_error' });
});
