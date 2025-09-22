import { useEffect, useState } from "react";

export default function GoNoGoBadge({ strategy = "ma_crossover_v1" }) {
  const [m, setM] = useState<any>(null);
  useEffect(() => {
    const f = async () => {
      const r = await fetch(`/api/metrics/live?strategy=${strategy}`);
      setM(await r.json());
    };
    f();
    const id = setInterval(f, 10_000);
    return () => clearInterval(id);
  }, [strategy]);

  if (!m) return <div className="badge">Loading…</div>;

  // Get thresholds from metrics response or use defaults
  const thresholds = m.thresholds || {
    oos_sharpe_min: 0.8,
    oos_pf_min: 1.2,
    oos_max_dd: 0.12,
    oos_q_value_max: 0.10,
    live_sharpe_min: 0.8,
    live_pf_min: 1.1,
    live_dd_max: 0.04,
    slippage_tolerance_bps: 5
  };

  const ok =
    m.oos?.sharpe >= thresholds.oos_sharpe_min &&
    m.oos?.pf >= thresholds.oos_pf_min &&
    m.oos?.max_dd <= thresholds.oos_max_dd &&
    m.oos?.q_value <= thresholds.oos_q_value_max &&
    m.live_paper_60d?.sharpe >= thresholds.live_sharpe_min &&
    m.live_paper_60d?.pf >= thresholds.live_pf_min &&
    m.live_paper_60d?.dd <= thresholds.live_dd_max &&
    m.live_paper_60d?.realized_slippage_bps <= (m.live_paper_60d?.avg_spread_bps + thresholds.slippage_tolerance_bps);

  return (
    <div className={`rounded px-3 py-1 font-medium ${ok ? "bg-green-600 text-white" : "bg-red-600 text-white"}`}>
      {ok ? "GO" : "NO-GO"}
    </div>
  );
}
