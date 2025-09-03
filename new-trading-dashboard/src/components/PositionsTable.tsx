import { useDashboardData } from "@/hooks/useDashboardData";

export default function PositionsTable() {
  const { positions } = useDashboardData();
  const rows = positions.data ?? [];
  return (
    <div>
      <h3 className="font-semibold mb-2">Positions</h3>
      {rows.length === 0 ? <div className="text-sm text-gray-500">No open positions</div> :
      <table className="w-full text-sm">
        <thead><tr><th>Symbol</th><th>Qty</th><th>Avg</th><th>Last</th><th>P/L $</th><th>P/L %</th></tr></thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.symbol}>
              <td>{p.symbol}</td><td>{p.qty}</td><td>{p.avg_price.toFixed(2)}</td>
              <td>{p.last.toFixed(2)}</td><td>{p.pl_dollar.toFixed(2)}</td><td>{p.pl_pct.toFixed(2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>}
    </div>
  );
}
