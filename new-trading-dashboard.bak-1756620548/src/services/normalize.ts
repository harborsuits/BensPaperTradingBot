// Normalizers to protect the UI from missing fields / shapes

const asNumber = (value: any, defaultValue = 0): number => {
  const n = Number(value);
  return Number.isFinite(n) ? n : defaultValue;
};

export type PortfolioNorm = {
  equity: number;
  cash: number;
  positions: any[];
  updated_at?: string;
  mode?: 'paper' | 'live';
};

export function toPortfolio(raw: any): PortfolioNorm {
  const r = raw ?? {};
  const summary = r.summary ?? r;
  return {
    equity: asNumber(
      summary.total_equity ?? summary.equity ?? summary.account_value ?? 0
    ),
    cash: asNumber(summary.cash_balance ?? summary.cash ?? summary.available_cash ?? 0),
    positions: Array.isArray(r.positions)
      ? r.positions
      : Array.isArray(r.holdings)
      ? r.holdings
      : [],
    updated_at: summary.updated_at ?? summary.last_updated ?? summary.timestamp,
    mode: r.mode,
  };
}

/**
 * Safely normalize any value to an array
 * Handles common API response structures automatically
 */
export const toArray = <T = any>(v: unknown): T[] => {
  if (Array.isArray(v)) return v as T[];
  const o = v as any;
  if (Array.isArray(o?.data)) return o.data as T[];
  if (Array.isArray(o?.items)) return o.items as T[];
  return v == null ? [] : [v] as T[];
};


