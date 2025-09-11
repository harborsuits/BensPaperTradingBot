import { z } from "zod";

/**
 * Standardized Decision Contract
 * AI → UI/Executor
 */
export const DecisionSchema = z.object({
  id: z.string(),
  ts: z.string(), // ISO timestamp
  symbol: z.string(),
  side: z.enum(["buy", "sell"]),
  asset_type: z.enum(["equity", "option"]),
  strategy: z.string(),
  confidence: z.number().min(0).max(1),
  score: z.number(),
  thesis: z.string(),
  risk: z.object({
    max_loss: z.number(),
    stop_pct: z.number(),
    cap_pct: z.number()
  }),
  order: z.object({
    type: z.enum(["market", "limit"]),
    qty: z.number()
  }),
  checks: z.object({
    cooldowns: z.boolean(),
    cbreakers: z.boolean(),
    slippage_ok: z.boolean()
  })
});

export type Decision = z.infer<typeof DecisionSchema>;

/**
 * Standardized Order Contract
 * Executor → UI
 */
export const OrderSchema = z.object({
  decision_id: z.string(),
  broker_order_id: z.string().optional(),
  status: z.enum(["pending", "working", "filled", "cancelled", "rejected", "failed"]),
  symbol: z.string(),
  side: z.enum(["buy", "sell"]),
  qty: z.number(),
  avg_fill: z.number().optional(),
  submitted_at: z.string(),
  filled_at: z.string().optional(),
  slippage_bps: z.number().optional(),
  latency_ms: z.number().optional()
});

export type Order = z.infer<typeof OrderSchema>;

/**
 * Standardized Portfolio Contract
 * Broker → UI
 */
export const PortfolioSchema = z.object({
  cash: z.number(),
  equity: z.number(),
  day_pnl: z.number(),
  open_pnl: z.number(),
  positions: z.array(z.object({
    symbol: z.string(),
    qty: z.number(),
    avg_cost: z.number(),
    last: z.number(),
    pnl: z.number()
  }))
});

export type Portfolio = z.infer<typeof PortfolioSchema>;

/**
 * Order Metrics for Execution Monitor
 */
export const OrderMetricsSchema = z.object({
  fill_rate: z.number(),
  avg_slippage_bps: z.number(),
  avg_exec_ms: z.number()
});

export type OrderMetrics = z.infer<typeof OrderMetricsSchema>;

/**
 * Brain Health Status
 */
export const BrainHealthSchema = z.object({
  status: z.enum(["ok", "degraded", "down"]),
  model: z.string().optional(),
  up_secs: z.number().optional(),
  last_decision: z.string().optional(),
  queue_size: z.number().optional()
});

export type BrainHealth = z.infer<typeof BrainHealthSchema>;
