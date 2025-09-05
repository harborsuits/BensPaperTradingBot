export type Allocation = { name: string; value: number; pct: number };

export interface PortfolioAllocationsResponse {
  data: {
    equity: number;
    cash: number;
    buying_power: number;
    pl_day: number;
    totalMV: number;
    typeAlloc: Allocation[];
    symbolAlloc: Allocation[];
  };
  asOf?: string;
  meta?: { asOf?: string; [k: string]: any };
}

export interface DecisionRow {
  id?: string;
  trace_id?: string;
  symbol: string;
  action?: "BUY" | "SELL" | "HOLD" | string;
  score?: number; // 0..100
  one_liner?: string;
  reason?: string;
  strategy?: string;
  strategyId?: string;
  createdAt?: string;
  entry?: number; stop?: number; target?: number; rr?: number;
  size?: number; maxLoss?: number; spreadPct?: number;
  instrument?: "equity"|"option"|"futures"|"crypto"|"forex";
  optionSymbol?: string; qty?: number; mid?: number; iv?: number;
  greeks?: { delta?:number; theta?:number; vega?:number; gamma?:number };
  gates?: Array<{ name: string; passed: boolean; details?: string }>;
}

export interface IngestEvent {
  ts: string; // ISO or epoch
  symbol: string;
  trace_id?: string;
  stage: "INGEST"|"CONTEXT"|"CANDIDATES"|"GATES"|"PLAN"|"ROUTE"|"MANAGE"|"LEARN"|string;
  status: "ok"|"fail"|"pending"|string;
  message?: string;
}

export interface AlertRow {
  id?: string;
  ts: string; // ISO
  level: "INFO"|"WARN"|"ERROR"|string;
  source?: string;
  message: string;
}

export interface ContextRow {
  regime: string; // e.g., "Neutral"
  vix?: number;
  bias?: "bull"|"bear"|"neutral"|"volatile";
  asOf?: string;
  note?: string;
}


