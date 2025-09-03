import { get, post, del } from "@/lib/api";

export type Account = { equity:number; cash:number; day_pl_dollar:number; day_pl_pct:number; };
export const AccountSvc = {
  balance: () => get<Account>("/api/v1/account/balance"),
};

export type Position = { symbol:string; qty:number; avg_price:number; last:number; pl_dollar:number; pl_pct:number; };
export const PositionsSvc = {
  list: () => get<Position[]>("/api/v1/positions"),
};

export type Order = { id:string; symbol:string; side:string; qty:number; type:string; limit_price:number|null; status:string; ts:string; };
export const OrdersSvc = {
  open:   () => get<Order[]>("/api/v1/orders/open"),
  recent: () => get<Order[]>("/api/v1/orders/recent"),
  place:  (p:{symbol:string; side:"buy"|"sell"; qty:number; type:"market"|"limit"; limit_price?:number|null;}) =>
           post<{order_id:string}>("/api/v1/orders", p, true),
  cancel: (id:string) => del<{ok:boolean}>(`/api/v1/orders/${id}`, true),
};

export type StrategyCard = { id:string; name:string; active:boolean; exposure_pct:number; last_signal_time?:string; last_signal_strength?:number; p_l_30d:number; };
export const StrategiesSvc = {
  list: () => get<StrategyCard[]>("/api/v1/strategies"),
  activate: (id:string)   => post<{ok:boolean}>(`/api/v1/strategies/${id}/activate`, {}, true),
  deactivate: (id:string) => post<{ok:boolean}>(`/api/v1/strategies/${id}/deactivate`, {}, true),
};

export type LiveSignal = { ts:string; strategy:string; symbol:string; action:string; size:number; reason:string; };
export const SignalsSvc = {
  live: () => get<LiveSignal[]>("/api/v1/signals/live"),
};

export const RiskSvc = {
  status: () => get<{ portfolio_heat:number; dd_pct:number; concentration_flag:boolean; blocks:string[]; }>("/api/v1/risk/status"),
};

export const HealthSvc = {
  status: () => get<{ broker:"UP"|"DOWN"; data:"UP"|"DEGRADED"|"DOWN"; last_heartbeat:string; }>("/api/v1/health"),
};

export const JobsSvc = {
  startBacktest: () => post<{job_id:string}>("/jobs/backtests", {}),
  status: (id:string) => get<{job_id:string; status:string; progress:number; result_ref?:string; error?:string; }>(`/jobs/${id}`),
};
