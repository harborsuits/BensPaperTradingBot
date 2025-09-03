import { useDashboardData } from "@/hooks/useDashboardData";
import { StrategiesSvc } from "@/services/trading";

export default function Strategies() {
  const { strategies } = useDashboardData();
  const data = strategies.data ?? [];

  const toggle = async (id:string, active:boolean) => {
    if (active) await StrategiesSvc.deactivate(id);
    else await StrategiesSvc.activate(id);
    // simple refresh:
    location.reload();
  };

  return (
    <div>
      <h3 className="font-semibold mb-2">Strategies</h3>
      {data.map(s => (
        <div key={s.id} className="flex items-center justify-between border p-2 rounded mb-2">
          <div>
            <div className="font-medium">{s.name} {s.active ? "🟢" : "⚪"}</div>
            <div className="text-xs text-gray-500">Exposure: {(s.exposure_pct*100).toFixed(1)}% • 30d P/L: {s.p_l_30d.toFixed(2)}</div>
          </div>
          <button onClick={() => toggle(s.id, s.active)} className="px-3 py-1 bg-gray-200 rounded">
            {s.active ? "Deactivate" : "Activate"}
          </button>
        </div>
      ))}
    </div>
  );
}
