export type BiasLean = "left" | "center" | "right" | "unknown";
export type CredScore = 1|2|3|4|5;
export type GateName =
  | "fresh_news" | "max_drawdown_ok" | "portfolio_heat_ok"
  | "broker_healthy" | "liquidity_ok" | "spread_ok" | "risk_budget_ok";

export interface SourceDoc {
  id: string; url: string; title: string; publisher: string;
  publishedAt: string; capturedAt: string;
  biasLean: BiasLean; credibility: CredScore;
  sentiment: number; relevance: number; keyClaims: string[];
}

export interface Prediction {
  thesis: string; direction: "up"|"down"|"volatile"; horizonHours: number;
  expectedMovePct: number; prob: number;
  bandsPct: { p10: number; p50: number; p90: number };
  invalidation: string;
}

export interface RiskPlan {
  maxLossUsd: number; maxPortfolioHeatAfter: number;
  stopPlan: string; rollOrHedgeRules?: string[];
  greeks?: { delta?:number; theta?:number; vega?:number; gamma?:number };
}

export interface OrderPlan {
  venue:"live"|"paper"; instrument:"equity"|"option"|"futures"|"crypto"|"forex";
  symbol:string; quantity:number; limit?:number; timeInForce:"day"|"gtc"|"ioc";
  preTrade:{ mid:number; spreadPct:number; iv?:number };
  idempotencyKey:string;
}

export interface GateCheck { name: GateName; passed: boolean; details?: string; }

export interface EvidencePacket {
  id:string; createdAt:string; strategyId:string; strategyName:string; symbol:string;
  context:{ regime:string; vix?:number; marketBias:"bull"|"bear"|"neutral"|"volatile" };
  tlDr:string; confidence:number; sources:SourceDoc[]; crossConfirmations:number;
  interpretation:string[]; prediction:Prediction;
  plan:{ strategyLabel:string; params:Record<string,any>; alternatives?:string[] };
  risk:RiskPlan; gates:GateCheck[]; orders:OrderPlan[];
  execution?:{ status:"pending"|"live"|"filled"|"rejected"|"cancelled"|"closed"; brokerOrderIds?:string[]; avgFill?:number; slippagePct?:number; };
  monitoring?: Array<{ at:string; note:string }>;
  outcome?: { closedAt:string; pnlUsd:number; lessons:string[]; };
}


