import { Router } from "express";
const r = Router();

// In-memory store for brain flow ticks - replace with your actual store
// This should be populated by your brain processing pipeline
const brainFlowStore = {
  ticks: [] as Array<{
    ts: number;
    symbol: string;
    stageResults: { gates_passed: boolean };
    mode: 'discovery' | 'shadow' | 'live';
    latency_ms: number;
  }>,

  addTick(tick: typeof brainFlowStore.ticks[0]) {
    this.ticks.push(tick);
    // Keep only last 1000 ticks to prevent memory issues
    if (this.ticks.length > 1000) {
      this.ticks.shift();
    }
  },

  querySince(since: number) {
    return this.ticks.filter(tick => tick.ts >= since);
  }
};

// Mock data population - replace with real brain pipeline integration
setInterval(() => {
  if (Math.random() > 0.7) { // Add tick ~30% of the time
    const tick = {
      ts: Date.now(),
      symbol: ['SPY', 'AAPL', 'QQQ', 'TSLA', 'NVDA'][Math.floor(Math.random() * 5)],
      stageResults: {
        ingest_ok: Math.random() > 0.1,
        context_ok: Math.random() > 0.2,
        candidates_ok: Math.random() > 0.3,
        gates_passed: Math.random() > 0.4,
        plan_ok: Math.random() > 0.7,
        route_ok: Math.random() > 0.8,
        manage_ok: Math.random() > 0.9,
        learn_ok: Math.random() > 0.95
      },
      mode: ['discovery', 'shadow', 'live'][Math.floor(Math.random() * 3)] as any,
      latency_ms: Math.random() * 200 + 50,
      costs: {
        spread_bps: Math.floor(Math.random() * 10) + 5,
        fees_per_contract: Math.random() * 1.5 + 0.5,
        slippage_bps: Math.floor(Math.random() * 5) + 1
      }
    };

    brainFlowStore.addTick(tick);

    // Emit SSE event for real-time updates
    if (global.brainClients) {
      const eventData = `data: ${JSON.stringify(tick)}\n\n`;
      global.brainClients.forEach(client => {
        try {
          client.write(eventData);
        } catch (e) {
          // Client disconnected, will be cleaned up
        }
      });
    }
  }
}, 2000); // Add tick every 2 seconds

function parseWindow(windowStr: string): number {
  const match = /^(\d+)(m|h|d)$/.exec(windowStr);
  if (!match) return 15 * 60 * 1000; // Default 15 minutes

  const num = parseInt(match[1]);
  const unit = match[2];

  switch (unit) {
    case 'm': return num * 60 * 1000;
    case 'h': return num * 60 * 60 * 1000;
    case 'd': return num * 24 * 60 * 60 * 1000;
    default: return 15 * 60 * 1000;
  }
}

r.get("/brain/flow/summary", (req, res) => {
  try {
    const windowMs = parseWindow(req.query.window as string || "15m");
    const since = Date.now() - windowMs;

    const rows = brainFlowStore.querySince(since);

    let gates_passed = 0, gates_failed = 0;
    let discovery = 0, shadow = 0, live = 0;
    const latencies: number[] = [];

    for (const tick of rows) {
      if (tick.stageResults?.gates_passed) gates_passed++;
      else gates_failed++;

      if (tick.mode === "discovery") discovery++;
      else if (tick.mode === "shadow") shadow++;
      else if (tick.mode === "live") live++;

      if (typeof tick.latency_ms === "number") latencies.push(tick.latency_ms);
    }

    const ingest_ok = rows.filter(t => t.stageResults?.ingest_ok).length;
    const context_ok = rows.filter(t => t.stageResults?.context_ok).length;
    const candidates_ok = rows.filter(t => t.stageResults?.candidates_ok).length;
    const plan_ok = rows.filter(t => t.stageResults?.plan_ok).length;
    const route_ok = rows.filter(t => t.stageResults?.route_ok).length;
    const manage_ok = rows.filter(t => t.stageResults?.manage_ok).length;
    const learn_ok = rows.filter(t => t.stageResults?.learn_ok).length;

    latencies.sort((a, b) => a - b);
    const p50 = latencies.length > 0 ? latencies[Math.floor(latencies.length * 0.5)] : null;
    const p95 = latencies.length > 0 ? latencies[Math.floor(latencies.length * 0.95)] : null;

    res.json({
      window: req.query.window || "15m",
      counts: {
        ingest_ok,
        context_ok,
        candidates_ok,
        gates_passed,
        gates_failed,
        plan_ok,
        route_ok,
        manage_ok,
        learn_ok,
        total: rows.length
      },
      by_mode: { discovery, shadow, live },
      latency_ms: {
        p50,
        p95,
        avg: latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null
      },
      asOf: new Date().toISOString()
    });
  } catch (e: any) {
    res.status(500).json({ error: "BrainFlowSummaryFailed", message: e.message });
  }
});

export default r;
