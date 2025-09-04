// src/types/DecisionTrace.ts
import { z } from 'zod';

// ---- Instrument Schema ----
export const InstrumentSchema = z.object({
  type: z.enum(['equity', 'option', 'futures', 'crypto', 'forex']),
  symbol: z.string(),
  underlier: z.string().optional(),
  expiry: z.string().optional(),
  strike: z.number().optional(),
  right: z.enum(['call', 'put']).optional(),
});

// ---- Market Context Schema ----
export const MarketContextSchema = z.object({
  regime: z.object({
    label: z.string(),
    confidence: z.number(),
    since: z.string().optional(),
  }).optional(),
  volatility: z.object({
    vix: z.number().optional(),
    percentile: z.number().optional(),
    trend: z.enum(['rising', 'falling', 'stable']).optional(),
  }).optional(),
  sentiment: z.object({
    label: z.enum(['bullish', 'bearish', 'neutral', 'mixed']),
    score: z.number().optional(),
    sources: z.array(z.string()).optional(),
  }).optional(),
  breadth: z.object({
    advDecRatio: z.number().optional(),
    newHighsLows: z.number().optional(),
    interpretation: z.string().optional(),
  }).optional(),
});

// ---- Signal Schema ----
export const SignalSchema = z.object({
  source: z.string(),
  name: z.string(),
  value: z.any().optional(),
  threshold: z.any().optional(),
  direction: z.enum(['bullish', 'bearish']).optional(),
  lookback: z.string().optional(),
});

// ---- News Evidence Schema ----
export const NewsEvidenceSchema = z.object({
  url: z.string(),
  headline: z.string(),
  snippet: z.string(),
  entities: z.array(z.string()).default([]),
  sentiment: z.enum(['positive', 'neutral', 'negative']).optional(),
  recency_min: z.number().optional(),
  credibility: z.enum(['high', 'medium', 'low']).optional(),
});

// ---- Candidate Score Schema ----
export const CandidateScoreSchema = z.object({
  alpha: z.number(),
  rank_in_universe: z.number().optional(),
  alternatives: z.array(z.record(z.any())).default([]),
});

// ---- Risk Gate Schema ----
export const RiskGateSchema = z.object({
  position_limits_ok: z.boolean().optional(),
  portfolio_heat_ok: z.boolean().optional(),
  drawdown_ok: z.boolean().optional(),
  notes: z.array(z.string()).default([]),
});

// ---- Plan Schema ----
export const PlanSchema = z.object({
  action: z.enum(['OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'ADJUST']),
  entry: z.record(z.any()).default({}),
  sizing: z.record(z.any()).default({}),
  exits: z.record(z.any()).default({}),
  expected_move: z.record(z.any()).default({}),
  strategyLabel: z.string().optional(),
});

// ---- Execution Schema ----
export const ExecutionSchema = z.object({
  status: z.enum(['PROPOSED', 'BLOCKED', 'SENT', 'PARTIAL', 'FILLED', 'CANCELED']).default('PROPOSED'),
  route: z.string().optional(),
  broker: z.string().optional(),
  broker_ids: z.array(z.string()).optional(),
  latency_ms: z.number().optional(),
});

// ---- Decision Trace Schema ----
export const DecisionTraceSchema = z.object({
  trace_id: z.string(),
  as_of: z.string(), // ISO8601
  schema_version: z.string().default('1.0'),
  symbol: z.string(),
  instrument: InstrumentSchema.optional(),
  account: z.object({ mode: z.enum(['paper', 'live']).default('paper'), broker: z.string().optional() }).optional(),
  market_context: MarketContextSchema.optional(),
  signals: z.array(SignalSchema).default([]),
  news_evidence: z.array(NewsEvidenceSchema).default([]),
  candidate_score: CandidateScoreSchema.optional(),
  risk_gate: RiskGateSchema.optional(),
  plan: PlanSchema,
  execution: ExecutionSchema.default({ status: 'PROPOSED' }),
  explain_layman: z.string().default(''),
  explain_detail: z.array(z.string()).default([]),
});

export type DecisionTrace = z.infer<typeof DecisionTraceSchema>;

// ---- Helpers ----
export function instrumentLabel(d: DecisionTrace): string {
  if (d.plan?.strategyLabel) return d.plan.strategyLabel;
  const i = d.instrument;
  if (!i) return 'Unknown';
  if (i.type === 'option' && i.underlier && i.expiry && i.right && i.strike !== undefined) {
    return `${i.underlier} ${i.expiry} ${i.right} ${i.strike}`;
  }
  if (i.type === 'equity' && i.symbol) return i.symbol;
  return i.type ?? 'Unknown';
}

export function proofStrength(d: DecisionTrace): 'Strong' | 'Medium' | 'Weak' {
  const n = d.news_evidence.length;
  const s = d.signals.length;
  const credible = d.news_evidence.filter(e => e.credibility !== 'low').length;
  if ((n >= 2 && credible >= 1) || (s >= 3 && n >= 1)) return 'Strong';
  if (n >= 1 && s >= 1) return 'Medium';
  return 'Weak';
}

export function validateTrace(obj: unknown): { ok: true; value: DecisionTrace } | { ok: false; error: string } {
  const r = DecisionTraceSchema.safeParse(obj);
  if (r.success) {
    // Require at least one source OR one signal; else downgrade via error
    if ((r.data.news_evidence?.length ?? 0) === 0 && (r.data.signals?.length ?? 0) === 0) {
      return { ok: false, error: 'Insufficient evidence: need at least one news source or one signal.' };
    }
    return { ok: true, value: r.data };
  }
  return { ok: false, error: r.error.message };
}
