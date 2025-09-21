import { Router } from "express";
import { z } from "zod";

// Import the decisionsStore from the summary route
import { decisionsStore } from "./decisionsSummary";

const r = Router();

const Stage = z.enum(["proposed", "intent", "executed", "filled"]).optional();

r.get("/decisions/recent", async (req, res) => {
  try {
    const stage = Stage.parse(req.query.stage);
    const limit = Math.max(1, Math.min(parseInt(String(req.query.limit ?? "100"), 10) || 100, 1000));

    // Get recent decisions from the store
    const items = await decisionsStore.querySince(Date.now() - (15 * 60 * 1000)); // Last 15 minutes

    // Filter by stage if specified
    const filteredItems = stage ? items.filter(item => item.stage === stage) : items;

    // Take the most recent items and format them
    const recentItems = filteredItems
      .slice(-limit)
      .reverse()
      .map(item => ({
        id: item.id || `decision_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        ts: item.ts,
        symbol: item.symbol,
        strategy_id: item.strategy_id,
        confidence: item.confidence,
        stage: item.stage,
        trace_id: item.trace_id || item.id,
        as_of: item.ts,
        explain_layman: item.reason || `Decision for ${item.symbol}`,
        plan: {
          strategyLabel: item.strategy_id
        },
        market_context: {
          regime: { label: "neutral" },
          volatility: { vix: 20 },
          sentiment: { label: "neutral" }
        },
        costs: item.costs
      }));

    // Always return {items: []} format
    res.json({ items: recentItems });
  } catch (e: any) {
    console.error('Decisions recent error:', e);
    res.status(400).json({ error: "BadRequest", message: e.message });
  }
});

export default r;
