export type GateDecision = "PASS" | "SKIP" | "PROBE";

export interface Candidate {
  id: string;                // trace or opportunity ID
  symbol: string;            // "SPY"
  kind?: string;             // "equity" | "etf" | "option" | "crypto"
  timestamp?: number;        // seconds or ms since epoch

  // WHY
  reason_tags?: string[];    // ["tariff", "semiconductors", "fresh_headline"]
  novelty_score?: number;    // 0..1
  sentiment_z?: number;      // e.g. -2.1
  iv_change_1d?: number;     // implied vol Δ in %
  headline?: string;         // optional headline snippet

  // FIT
  regime?: string;           // "risk_off" | "mixed" | "risk_on"
  regime_alignment?: "bullish"|"bearish"|"neutral"|"divergent";

  // COSTS & RISK
  fees_bps?: number;
  slip_bps?: number;
  spread_bps?: number;
  spread_cap_bps?: number;
  cluster_heat_delta?: number; // +0.01 => +1% heat if filled

  // PLAN SUGGESTION (if you have it)
  plan_strategy?: string;      // "put_debit_spread"
  plan_risk_usd?: number;      // 25
  plan_target_r?: number;      // 2
  plan_horizon?: string;       // "1–2 days"

  // GO/NO-GO
  meta_prob?: number;          // 0..1
  meta_threshold?: number;     // 0..1
  decision?: GateDecision;     // "PASS" | "SKIP" | "PROBE"
  skip_codes?: string[];       // ["SPREAD_TOO_WIDE", "NOVELTY_LOW"]
}
