// live-api/routes/metrics.ts
import { Router } from "express";
export const metrics = Router();

/* Assume you have journals or DB tables for trades/orders and PnL
   with fields: ts, symbol, side, qty, mid_at_submit, fill, spread_bps, realized_slippage_bps, pnl_after_cost, regime_label
*/
import { getRecentTrades, getBacktestOOS } from "../services/metricsAgg";

metrics.get("/live", async (req, res) => {
  const strat = req.query.strategy ?? "ma_crossover_v1";
  const live20 = await getRecentTrades(strat, 20);
  const live60 = await getRecentTrades(strat, 60);
  const oos = await getBacktestOOS(strat);

  const roll = (rows:any[]) => {
    const pnl = rows.map(r => r.pnl_after_cost);
    const mean = pnl.reduce((a,b)=>a+b,0)/Math.max(pnl.length,1);
    const stdev = Math.sqrt(pnl.reduce((a,b)=>a+(b-mean)**2,0)/Math.max(pnl.length-1,1));
    const sharpe = stdev>0 ? mean/stdev : 0;
    const wins = rows.filter(r=>r.pnl_after_cost>0).length;
    const pf = (rows.filter(r=>r.pnl_after_cost>0).reduce((a,b)=>a+b.pnl_after_cost,0) || 0.0001) /
               Math.abs(rows.filter(r=>r.pnl_after_cost<0).reduce((a,b)=>a+b.pnl_after_cost,0) || 0.0001);
    const dd = Math.min(0, rows.reduce((acc,r)=> {
      acc.e += r.pnl_after_cost; acc.p=Math.min(acc.p, acc.e); return acc;
    }, {e:0,p:0}).p);
    const avgSpread = rows.reduce((a,b)=>a+b.spread_bps,0)/Math.max(rows.length,1);
    const avgSlip = rows.reduce((a,b)=>a+b.realized_slippage_bps,0)/Math.max(rows.length,1);
    const breaches = rows.filter(r=> (r.breaches||[]).length>0).length / Math.max(rows.length,1);
    return { sharpe, pf, dd: Math.abs(dd), trades: rows.length, avg_spread_bps: avgSpread, realized_slippage_bps: avgSlip, breach_rate: breaches }
  };

  const rb = (rows:any[], label:string) => {
    const sub = rows.filter(r=>r.regime_label===label);
    return roll(sub);
  };

  const live20m = roll(live20);
  const live60m = roll(live60);

  res.json({
    strategy: String(strat),
    oos,
    live_paper_20d: live20m,
    live_paper_60d: live60m,
    regime_breakdown: {
      trend: rb(live60, "trend"),
      chop:  rb(live60, "chop"),
      high_vol: rb(live60, "high_vol")
    }
  });
});
