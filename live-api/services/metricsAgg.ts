// live-api/services/metricsAgg.ts
// Placeholder implementation - replace with actual DB queries when ready

export async function getRecentTrades(strategy: string, days: number): Promise<any[]> {
  // TODO: Implement actual DB query for recent trades
  // For now, return mock data
  return [
    {
      ts: new Date().toISOString(),
      symbol: "SPY",
      side: "buy",
      qty: 100,
      mid_at_submit: 450.0,
      fill: 450.25,
      spread_bps: 5,
      realized_slippage_bps: 6,
      pnl_after_cost: 125.50,
      regime_label: "trend"
    }
  ];
}

export async function getBacktestOOS(strategy: string): Promise<any> {
  // TODO: Implement actual DB query for backtest OOS results
  // For now, return mock data
  return {
    sharpe: 0.86,
    pf: 1.23,
    max_dd: 0.08,
    trades: 147,
    q_value: 0.06
  };
}
