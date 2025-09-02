import { useActiveStrategies } from "@/hooks/useActiveStrategies";
import { Card } from "@/components/ui/Card";

export default function StrategiesPage(){
  const { data } = useActiveStrategies();
  const items = Array.isArray(data) ? data : data?.items ?? [];

  return (
    <Card className="card">
      <div className="card-header">
        <div className="card-title">Active Strategies</div>
        <div className="card-subtle">as of {items?.[0]?.asOf ?? "—"}</div>
      </div>
      <div className="card-content">
        <table className="w-full text-sm">
          <thead><tr><th className="text-left">Name</th><th>Signal</th><th>Conf</th><th>Pos</th><th className="text-right">P&L (Day)</th><th>Health</th></tr></thead>
          <tbody>
            {items.map((s:any)=>(
              <tr key={s.name}>
                <td>{s.name}</td><td>{s.signal ?? "FLAT"}</td>
                <td className="text-center">{s.confidence ?? 0}%</td>
                <td className="text-center">{s.position ?? 0}</td>
                <td className="text-right">${(s.dayPL ?? 0).toFixed(2)}</td>
                <td className="text-center">{s.health ?? "ok"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}