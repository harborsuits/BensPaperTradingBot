import { useDashboardData } from "@/hooks/useDashboardData";

export default function RiskPanel() {
  const { risk } = useDashboardData();
  const r = risk.data;
  return (
    <div className="flex gap-8">
      <div><div className="text-xs text-gray-500">Portfolio Heat</div><div className="text-xl">{r ? r.portfolio_heat.toFixed(1) : "-"}%</div></div>
      <div><div className="text-xs text-gray-500">Drawdown</div><div className="text-xl">{r ? r.dd_pct.toFixed(1) : "-"}%</div></div>
      <div><div className="text-xs text-gray-500">Concentration</div><div className="text-xl">{r?.concentration_flag ? "⚠️" : "OK"}</div></div>
    </div>
  );
}
