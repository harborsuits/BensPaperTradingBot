import express from 'express';
import type { Request, Response } from 'express';
import prom from 'prom-client';
import { Server as WSServer } from 'ws';
import http from 'http';

// Early-bind app and health/metrics
const app = express();
app.use(express.json());

app.get('/health', (_req: Request, res: Response) => {
  res.json({
    env: process.env.NODE_ENV || 'dev',
    ok: true,
    breaker: 'GREEN',
    asOf: new Date().toISOString(),
    gitSha: process.env.GIT_SHA || 'local-dev',
    timestamp: new Date().toISOString(),
  });
});

prom.collectDefaultMetrics();
app.get('/metrics', async (_req: Request, res: Response) => {
  res.set('Content-Type', prom.register.contentType);
  res.end(await prom.register.metrics());
});

// Simple in-memory EvoTester for alias + param routes
type Session = {
  id: string;
  running: boolean;
  status: 'running' | 'completed' | 'stopped' | 'paused';
  currentGeneration: number;
  totalGenerations: number;
  startTime: string;
  bestFitness: number;
  averageFitness: number;
};

const sessions = new Map<string, Session>();
const generations = new Map<string, Array<{ generation: number; bestFitness: number; averageFitness: number; timestamp: string }>>();
const results = new Map<string, Array<any>>();
let lastSessionId: string | null = null;

function startEvolution(id: string, total: number) {
  const interval = setInterval(() => {
    const s = sessions.get(id);
    if (!s || !s.running) { clearInterval(interval); return; }
    s.currentGeneration += 1;
    s.bestFitness = Math.max(s.bestFitness, Math.random() * 0.5 + 0.5);
    s.averageFitness = Math.max(0, Math.min(1, s.averageFitness + (Math.random() - 0.4) * 0.05));
    const log = generations.get(id) || [];
    log.push({ generation: s.currentGeneration, bestFitness: +s.bestFitness.toFixed(3), averageFitness: +s.averageFitness.toFixed(3), timestamp: new Date().toISOString() });
    generations.set(id, log);
    if (s.currentGeneration >= total) {
      s.running = false;
      s.status = 'completed';
      clearInterval(interval);
      const base = +s.bestFitness.toFixed(2);
      const items = [0,1,2].map((i) => ({
        id: `${id}_best_${i+1}`,
        name: i === 0 ? 'RSI-Momentum-V2' : i === 1 ? 'VWAP-Reversion' : 'News-Momo',
        fitness: +(base - i*0.12).toFixed(2),
        performance: { sharpeRatio: +(base - i*0.12).toFixed(2), winRate: +(0.55 + i*0.03).toFixed(2), maxDrawdown: +(0.10 + i*0.02).toFixed(2), trades: 60 + i*10 },
        created: new Date().toISOString()
      }));
      results.set(id, items);
    }
  }, 1000);
}

// Start
const PORT = Number(process.env.PORT || 4100);
const server = http.createServer(app);
server.listen(PORT, () => console.log(`[BOOT] listening on :${PORT}`));

// WS hubs (accept connections; no payloads required for UI to be happy)
const wss = new WSServer({ noServer: true });
const wssDecisions = new WSServer({ noServer: true });
const wssPrices = new WSServer({ noServer: true });

function safeSend(ws: any, obj: any) {
  try { ws.send(JSON.stringify(obj)); } catch {}
}

wss.on('connection', (ws) => {
  safeSend(ws, { type: 'hello', ts: new Date().toISOString() });
  try {
    ws.on('message', (raw: any) => {
      try {
        const msg = JSON.parse(String(raw));
        if (msg && msg.type === 'ping') {
          safeSend(ws, { type: 'pong', ts: new Date().toISOString() });
        }
      } catch {}
    });
  } catch {}
});
wssDecisions.on('connection', (ws) => {
  safeSend(ws, { type: 'ready', channel: 'decisions' });
});
wssPrices.on('connection', (ws) => {
  safeSend(ws, { type: 'prices', data: {}, time: new Date().toISOString() });
});

server.on('upgrade', (request, socket, head) => {
  try {
    let pathname = new URL(request.url || '/', `http://${request.headers.host}`).pathname || '/ws';
    pathname = pathname.replace(/\/+$/, '');
    if (pathname === '/ws/decisions') {
      wssDecisions.handleUpgrade(request, socket as any, head, (ws) => {
        wssDecisions.emit('connection', ws, request);
      });
    } else if (pathname === '/ws/prices') {
      wssPrices.handleUpgrade(request, socket as any, head, (ws) => {
        wssPrices.emit('connection', ws, request);
      });
    } else if (pathname === '/ws') {
      wss.handleUpgrade(request, socket as any, head, (ws) => {
        wss.emit('connection', ws, request);
      });
    } else {
      (socket as any).destroy();
    }
  } catch {
    try { (socket as any).destroy(); } catch {}
  }
});

// Param routes
app.post('/api/evotester/start', (req: Request, res: Response) => {
  const cfg = req.body || {};
  const id = `evo-${Math.random().toString(36).slice(2, 10)}`;
  const total = Number(cfg.generations || 6);
  const nowIso = new Date().toISOString();
  sessions.set(id, {
    id,
    running: true,
    status: 'running',
    currentGeneration: 0,
    totalGenerations: total,
    startTime: nowIso,
    bestFitness: 0,
    averageFitness: 0,
  });
  generations.set(id, []);
  startEvolution(id, total);
  lastSessionId = id;
  res.json({ session_id: id, sessionId: id });
});

app.get('/api/evotester/:id/status', (req: Request, res: Response) => {
  const s = sessions.get(req.params.id);
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

app.get('/api/evotester/:id/generations', (req: Request, res: Response) => {
  if (!sessions.has(req.params.id)) return res.status(404).json([]);
  res.json(generations.get(req.params.id) || []);
});

app.get('/api/evotester/:id/results', (req: Request, res: Response) => {
  if (!sessions.has(req.params.id)) return res.status(404).json([]);
  res.json(results.get(req.params.id) || []);
});

// Alias routes
app.get('/api/evotester/status', (req: Request, res: Response) => {
  const id = String((req.query.session_id as string) || '').trim();
  if (!id) return res.status(400).json({ error: 'session_id required' });
  const s = sessions.get(id);
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

app.get('/api/evotester/generations', (req: Request, res: Response) => {
  const id = String((req.query.session_id as string) || '').trim();
  if (!id) return res.status(400).json([]);
  res.json(generations.get(id) || []);
});

app.get('/api/evotester/results', (req: Request, res: Response) => {
  const id = String((req.query.session_id as string) || '').trim();
  const limit = Number((req.query.limit as string) || 20) || 20;
  if (!id) return res.status(400).json([]);
  res.json((results.get(id) || []).slice(0, limit));
});

app.post('/api/strategies/promote', (req: Request, res: Response) => {
  const { strategy_id, session_id } = (req.body || {}) as { strategy_id?: string; session_id?: string };
  if (!strategy_id || !session_id) return res.status(400).json({ error: 'strategy_id and session_id required' });
  const items = results.get(session_id) || [];
  const found = items.find((x) => x.id === strategy_id);
  if (!found) return res.status(404).json({ error: 'candidate_not_found' });
  res.json({ success: true, message: 'Promoted to paper candidate (dev)', candidate: { id: found.id, fitness: found.fitness } });
});

// --- Additional lightweight routes used by the dashboard ---
app.get('/api/health', (_req, res) => {
  res.json({
    env: process.env.NODE_ENV || 'dev',
    ok: true,
    breaker: 'GREEN',
    asOf: new Date().toISOString(),
    gitSha: process.env.GIT_SHA || 'local-dev',
    timestamp: new Date().toISOString(),
  });
});

app.get('/api/alerts', (req, res) => res.json([]));
app.get('/api/alerts/recent', (req, res) => res.json([]));
app.get('/api/ingestion/events', (req, res) => res.json([]));
app.get('/api/roster/active', (req, res) => res.json({ symbols: ['SPY','AAPL','QQQ'], count: 3, timestamp: new Date().toISOString() }));

app.get('/api/quotes', (req, res) => {
  const q = String(req.query.symbols || '').trim();
  const list = q ? q.split(',').map(s => s.trim().toUpperCase()).filter(Boolean) : [];
  const now = Date.now();
  const out = list.map((symbol) => {
    const base = 100 + Math.floor(Math.random() * 300);
    const last = +(base + (Math.random() - 0.5) * 2).toFixed(2);
    const prevClose = +(last * (0.99 + Math.random() * 0.02)).toFixed(2);
    const bid = +(last - 0.05).toFixed(2);
    const ask = +(last + 0.05).toFixed(2);
    const mid = (bid + ask) / 2;
    const spreadPct = mid > 0 ? +(((ask - bid) / mid) * 100).toFixed(3) : 0;
    return { symbol, last, prevClose, bid, ask, spreadPct, volume: Math.floor(Math.random()*2_000_000) };
  });
  res.json(out);
});

app.get('/api/context', (req, res) => res.json({
  regime: { type: 'neutral_medium' },
  sentiment: { score: 0.55, sources: { news: { score: 0.5 }, social: { score: 0.6 } } },
  volatility: { value: 17.2 }
}));
app.get('/api/context/news', (req, res) => res.json([]));

app.get('/api/portfolio', (req, res) => res.json({ mode: req.query.mode || 'paper', positions: [], cash: 100000 }));
app.get('/api/portfolio/summary', (req, res) => res.json({ nav: 100000, pnl: 0, dayChangePct: 0 }));
app.get('/api/portfolio/paper', (req, res) => res.json({ positions: [], cash: 100000 }));
app.get('/api/portfolio/paper/history', (req, res) => res.json([]));
app.get('/api/paper/account', (req, res) => res.json({ cash: 100000, buyingPower: 200000 }));
app.get('/api/paper/positions', (req, res) => res.json([]));
app.get('/api/paper/orders', (req, res) => res.json({ items: [] }));

app.get('/api/decisions', (req, res) => res.json([]));
app.get('/api/decisions/latest', (req, res) => res.json([]));
app.get('/api/decisions/recent', (req, res) => res.json([]));

app.get('/api/strategies', (req, res) => res.json([]));
app.get('/api/evotester/history', (req, res) => res.json([]));

app.get('/api/brain/status', (req, res) => res.json({ status: 'idle' }));

// Harmonized brain flow summary for dashboard (dev-friendly, zeroed but structured)
app.get('/api/brain/flow/summary', (req, res) => {
  const now = new Date().toISOString();
  res.json({
    // New concise contract
    counts: { pipelines: 0, active: 0, blocked: 0, gates_passed: 0, gates_failed: 0 },
    by_mode: { live: 0, paper: 0, backtest: 0, discovery: 0, shadow: 0 },
    latency_ms: { p50: 0, p90: 0, p95: 0, p99: 0 },
    asOf: now,
    dev_mode: true
  });
});

// Harmonized decisions summary (includes both legacy and new fields)
app.get('/api/decisions/summary', (req, res) => {
  const now = new Date().toISOString();
  res.json({
    counts: { total: 0, unique_symbols: 0 },
    unique_symbols: 0,
    proposals_per_min: 0,
    by_stage: { proposed: 0, filtered: 0, approved: 0, routed: 0, rejected: 0 },
    asOf: now,
    dev_mode: true
  });
});

// Evo dashboard status shim
app.get('/api/evo/status', (req, res) => {
  const active = Array.from(sessions.values()).filter(s => s.running).length;
  const last = lastSessionId ? sessions.get(lastSessionId) || null : null;
  res.json({
    active_sessions: active,
    last_session: last ? { id: last.id, status: last.status, gens: { current: last.currentGeneration, total: last.totalGenerations } } : null,
    asOf: new Date().toISOString()
  });
});

app.get('/api/competition/ledger', (req, res) => res.json([]));
app.get('/api/competition/poolStatus', (req, res) => res.json({ total_pool: 0, poolPnl: 0 }));

app.get('/api/overview', (req, res) => res.json({}));
app.get('/api/discovery/market', (req, res) => res.json([]));
app.get('/api/fundamentals', (req, res) => res.json({}));

app.get('/api/events/logs', (req, res) => res.json([]));
app.get('/api/logs', (req, res) => res.json([]));

app.get('/api/audit/autoloop/status', (req, res) => res.json({ is_running: false, status: 'IDLE', interval_ms: 30000 }));
// Thin alias the UI sometimes uses
app.get('/api/autoloop/status', (req, res) => res.json({ is_running: false, status: 'IDLE', interval_ms: 30000 }));

app.get('/api/universe', (req, res) => res.json(['SPY','AAPL','QQQ']));
app.get('/api/watchlists', (req, res) => res.json({ default: ['SPY','AAPL','QQQ'] }));

app.get('/api/bars', (req, res) => {
  const symbol = String(req.query.symbol || 'SPY').toUpperCase();
  const timeframe = String(req.query.timeframe || '1Day');
  const limit = Math.min(parseInt(String(req.query.limit || '180')) || 180, 1000);
  const days = timeframe === '1Day' ? limit : Math.ceil(limit/5);
  const today = new Date();
  const bars = Array.from({ length: days }, (_, i) => {
    const d = new Date(today.getTime() - (days - i) * 86400000);
    const o = +(100 + Math.random()*50).toFixed(2);
    const h = +(o * (1 + Math.random()*0.02)).toFixed(2);
    const l = +(o * (1 - Math.random()*0.02)).toFixed(2);
    const c = +(l + Math.random()*(h-l)).toFixed(2);
    const v = Math.floor(1_000_000 + Math.random()*2_000_000);
    return { t: d.toISOString().slice(0,10), o, h, l, c, v };
  });
  res.json(bars);
});

// Keep process alive even if later dynamic init fails
process.on('uncaughtException', (e) => console.error('[uncaught]', e));
process.on('unhandledRejection', (e) => console.error('[unhandled]', e));


