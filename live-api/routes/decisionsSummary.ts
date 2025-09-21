import { Router } from "express";
const r = Router();

// Import shared decisions store
const { decisionsStore } = require('./decisionsStore');

// Export decisionsStore for use in main server
export { decisionsStore };

// Mock data population - replace with real decision pipeline integration
setInterval(() => {
  if (Math.random() > 0.8) { // Add decision ~20% of the time
    const decision = {
      ts: Date.now(),
      symbol: ['SPY', 'AAPL', 'QQQ', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'][Math.floor(Math.random() * 7)],
      stage: ['proposed', 'intent', 'executed'][Math.floor(Math.random() * 3)] as any,
      strategy_id: `strategy_${Math.floor(Math.random() * 5)}`,
      confidence: Math.random() * 0.8 + 0.2, // 0.2 to 1.0
      costs: {
        spread_bps: Math.floor(Math.random() * 10) + 5, // 5-15 bps
        fees_per_contract: Math.random() * 1.5 + 0.5, // 0.5-2.0
        slippage_bps: Math.floor(Math.random() * 5) + 1 // 1-6 bps
      }
    };

    decisionsStore.push(decision);

    // Emit SSE event for real-time updates
    if (global.decisionsClients) {
      const eventData = `event: ${decision.stage}\ndata: ${JSON.stringify(decision)}\n\n`;
      global.decisionsClients.forEach(client => {
        try {
          client.write(eventData);
        } catch (e) {
          // Client disconnected, will be cleaned up
        }
      });
    }
  }
}, 3000); // Add decision every 3 seconds

r.get("/decisions/summary", (req, res) => {
  try {
    const windowMs = parseInt(req.query.window as string) || 15 * 60 * 1000; // Default 15 minutes
    const since = Date.now() - windowMs;

    const items = decisionsStore.querySince(since);

    const proposed = items.filter(i => i.stage === "proposed");
    const executed = items.filter(i => i.stage === "executed" || i.stage === "filled");
    const intents = items.filter(i => i.stage === "intent");

    const perMin = items.length / (windowMs / 60000);
    const uniqueSymbols = new Set(items.map(i => i.symbol)).size;
    const avgConfidence = items.length > 0
      ? items.reduce((sum, i) => sum + i.confidence, 0) / items.length
      : 0;

    const last_ts = items.length > 0 ? new Date(items[items.length - 1].ts).toISOString() : null;

    res.json({
      window: `${Math.round(windowMs / 60000)}m`,
      total_decisions: items.length,
      proposals_per_min: Number(perMin.toFixed(2)),
      unique_symbols: uniqueSymbols,
      by_stage: {
        proposed: proposed.length,
        intents: intents.length,
        executed: executed.length
      },
      avg_confidence: Number(avgConfidence.toFixed(3)),
      last_ts,
      asOf: new Date().toISOString()
    });
  } catch (e: any) {
    res.status(500).json({ error: "DecisionsSummaryFailed", message: e.message });
  }
});

// SSE endpoint for real-time decisions
r.get("/decisions/stream", (req, res) => {
  // Set SSE headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Cache-Control',
  });

  // Initialize global clients array if not exists
  if (!global.decisionsClients) {
    global.decisionsClients = new Set();
  }

  // Add this client
  global.decisionsClients.add(res);

  // Send initial connection message
  res.write('data: {"type": "connected"}\n\n');

  // Handle client disconnect
  req.on('close', () => {
    global.decisionsClients.delete(res);
  });

  req.on('error', () => {
    global.decisionsClients.delete(res);
  });
});

export default r;
