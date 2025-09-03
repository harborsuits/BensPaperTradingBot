import { useDashboardData } from "@/hooks/useDashboardData";

export default function DashboardTop() {
  const { account, health } = useDashboardData();
  const a = account.data;
  const h = health.data;

  return (
    <div className="grid grid-cols-3 gap-12">
      <div>
        <div className="text-sm text-gray-500">Equity</div>
        <div className="text-2xl font-semibold">${(a?.equity ?? 0).toLocaleString()}</div>
        <div className="text-xs text-gray-500">Cash: ${(a?.cash ?? 0).toLocaleString()} • Day P/L: {(a?.day_pl_pct ?? 0).toFixed(2)}%</div>
      </div>
      <div>
        <div className="text-sm text-gray-500">Session</div>
        <div className="text-2xl">{h?.broker === "UP" ? "🟢 Connected (Tradier paper)" : "🔴 Broker DOWN"}</div>
      </div>
      <div>
        <div className="text-sm text-gray-500">Health</div>
        <div className="text-2xl">{h ? `${h.broker}/${h.data}` : "-"}</div>
        <div className="text-xs text-gray-500">{h ? new Date(h.last_heartbeat).toLocaleTimeString() : ""}</div>
      </div>
    </div>
  );
}
