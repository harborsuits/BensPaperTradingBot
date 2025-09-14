import { get, post, del } from "@/lib/api";

export type Account = { equity:number; cash:number; day_pl_dollar:number; day_pl_pct:number; };
export const AccountSvc = {
  balance: () => get<{ data:{ equity:number; cash:number } }>("/api/portfolio/allocations").then(r => ({
    equity: r.data?.equity ?? 0,
    cash: r.data?.cash ?? 0,
    day_pl_dollar: 0,
    day_pl_pct: 0,
  })),
};

export type Position = { symbol:string; qty:number; avg_price:number; last:number; pl_dollar:number; pl_pct:number; };
export const PositionsSvc = {
  list: () => get<any[]>("/api/paper/positions").then(rows => rows.map(p => ({
    symbol: p.symbol,
    qty: p.quantity ?? p.qty ?? 0,
    avg_price: p.avg_price ?? 0,
    last: p.last_price ?? p.last ?? 0,
    pl_dollar: ((p.last_price ?? 0) - (p.avg_price ?? 0)) * (p.quantity ?? 0),
    pl_pct: ((p.last_price ?? 0) / Math.max(1, (p.avg_price ?? 1)) - 1) * 100,
  }) as Position)),
};

export type Order = { id:string; symbol:string; side:string; qty:number; type:string; limit_price:number|null; status:string; ts:string; };
export const OrdersSvc = {
  open:   () => get<Order[]>("/api/paper/orders/open"),
  recent: () => get<{ items: any[] }>("/api/trades").then(x => x.items || []),
  place:  (p:{symbol:string; side:"buy"|"sell"; qty:number; type:"market"|"limit"; limit_price?:number|null;}) =>
           post<{order_id?:string}|any>("/api/paper/orders/dry-run", p),
  cancel: (id:string) => post<{ok:boolean}>(`/api/paper/orders/${id}/cancel`, {}),
};

export type StrategyCard = { id:string; name:string; active:boolean; exposure_pct:number; last_signal_time?:string; last_signal_strength?:number; p_l_30d:number; };
export const StrategiesSvc = {
  list: () => get<{ strategies: any[] }>("/api/live/strategies").then(x => (x.strategies || []).map(s => ({
    id: s.id,
    name: s.name,
    active: s.status === 'active',
    exposure_pct: (s.config?.allocated_capital || 0) / 100000, // Convert to percentage
    last_signal_time: s.last_updated,
    last_signal_strength: 0.5, // Default value
    p_l_30d: s.performance?.total_return || 0,
  }))),
  activate: (id:string)   => post<{ok:boolean}>(`/api/live/strategies/${id}/activate`, {}),
  deactivate: (id:string) => post<{ok:boolean}>(`/api/live/strategies/${id}/deactivate`, {}),
};

export type LiveSignal = { ts:string; strategy:string; symbol:string; action:string; size:number; reason:string; };
export const SignalsSvc = {
  live: () => get<any[]>("/api/alerts").then(x => x?.items || []),
};

export const RiskSvc = {
  status: () => get<any>("/api/risk/status"),
};

export const HealthSvc = {
  status: () => get<any>("/api/health"),
};

export const JobsSvc = {
  startBacktest: () => post<{job_id:string}>("/jobs/backtests", {}),
  status: (id:string) => get<{job_id:string; status:string; progress:number; result_ref?:string; error?:string; }>(`/jobs/${id}`),
};
