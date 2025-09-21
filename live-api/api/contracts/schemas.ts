import { z } from 'zod';

// Health endpoint schema
export const Health = z.object({
  ok: z.boolean(),
  breaker: z.string(),
  quote_age_s: z.number().nullable(),
  broker: z.object({ ok: z.boolean() }).optional(),
  version: z.string().optional(),
  env: z.string().optional(),
  region: z.string().optional(),
  services: z.object({
    api: z.object({
      status: z.string(),
      lastUpdated: z.string().optional()
    })
  }).optional(),
  broker_age_s: z.number().optional(),
  slo_error_budget: z.number().optional(),
  asOf: z.string().optional(),
  timestamp: z.string().optional(),
  uptime: z.number().optional(),
});

// Decision schemas
export const DecisionProposed = z.object({
  id: z.string(),
  ts: z.string(),
  symbol: z.string(),
  strategy_id: z.string(),
  confidence: z.number(),
  reason: z.string().optional(),
  mode: z.string().optional(),
  trace_id: z.string().optional(),
  as_of: z.string().optional(),
  explain_layman: z.string().optional(),
  plan: z.object({
    strategyLabel: z.string().optional(),
  }).optional(),
  market_context: z.object({
    regime: z.object({ label: z.string() }).optional(),
    volatility: z.object({ vix: z.number() }).optional(),
    sentiment: z.object({ label: z.string() }).optional(),
  }).optional(),
});

// Brain activity schema
export const BrainActivity = z.object({
  ts: z.string(),
  symbol: z.string(),
  final_score: z.number(),
  confidence: z.number(),
  regime: z.string(),
  news_delta: z.number(),
  latency_ms: z.number(),
  fallback: z.boolean(),
  decision_id: z.string(),
  experts: z.array(z.any()).optional(),
  gates: z.object({
    earnings_window: z.boolean(),
    dd_ok: z.boolean(),
    fresh_ok: z.boolean(),
  }).optional(),
});

// Autoloop status schema
export const AutoloopStatus = z.object({
  mode: z.enum(['discovery', 'shadow', 'live']),
  active: z.boolean().optional(),
  last_run: z.string().optional(),
});

// Paper position schema
export const PaperPosition = z.object({
  symbol: z.string(),
  quantity: z.number(),
  avg_price: z.number(),
  current_price: z.number().optional(),
  market_value: z.number(),
  unrealized_pnl: z.number(),
});

// Paper order schema
export const PaperOrder = z.object({
  id: z.string(),
  symbol: z.string(),
  side: z.string(),
  quantity: z.number(),
  price: z.number(),
  status: z.string(),
  created_at: z.string(),
  updated_at: z.string().optional(),
});

// Portfolio summary schema
export const PortfolioSummary = z.object({
  equity: z.number(),
  cash: z.number(),
  positions: z.array(PaperPosition),
  total_value: z.number(),
  day_pnl: z.number(),
  total_pnl: z.number(),
});

// Array or wrapped array schemas (for flexible API responses)
export const DecisionsRecent = z.array(DecisionProposed).or(
  z.object({ items: z.array(DecisionProposed) })
);

export const BrainActivityRecent = z.array(BrainActivity).or(
  z.object({ items: z.array(BrainActivity) })
);

export const PositionsList = z.array(PaperPosition).or(
  z.object({ items: z.array(PaperPosition) })
);

export const OrdersList = z.array(PaperOrder).or(
  z.object({ items: z.array(PaperOrder) })
);

// Metrics schema
export const Metrics = z.object({
  totalSymbolsTracked: z.number(),
  errorRate: z.number(),
  requestsLastHour: z.number(),
  averageLatency: z.number(),
});

// Export types for use in other files
export type HealthType = z.infer<typeof Health>;
export type DecisionProposedType = z.infer<typeof DecisionProposed>;
export type BrainActivityType = z.infer<typeof BrainActivity>;
export type AutoloopStatusType = z.infer<typeof AutoloopStatus>;
export type PaperPositionType = z.infer<typeof PaperPosition>;
export type PaperOrderType = z.infer<typeof PaperOrder>;
export type PortfolioSummaryType = z.infer<typeof PortfolioSummary>;
export type MetricsType = z.infer<typeof Metrics>;
