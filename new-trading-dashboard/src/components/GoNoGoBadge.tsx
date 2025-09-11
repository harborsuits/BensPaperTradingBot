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

  const ok =
    m.oos?.sharpe >= 0.8 &&
    m.oos?.pf >= 1.2 &&
    m.oos?.max_dd <= 0.12 &&
    m.oos?.q_value <= 0.10 &&
    m.live_paper_60d?.sharpe >= 0.8 &&
    m.live_paper_60d?.pf >= 1.1 &&
    m.live_paper_60d?.dd <= 0.04 &&
    m.live_paper_60d?.realized_slippage_bps <= (m.live_paper_60d?.avg_spread_bps + 5);

  return (
    <div className={`rounded px-3 py-1 font-medium ${ok ? "bg-green-600 text-white" : "bg-red-600 text-white"}`}>
      {ok ? "GO" : "NO-GO"}
    </div>
  );
}
