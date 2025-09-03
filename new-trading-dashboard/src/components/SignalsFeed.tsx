import { useDashboardData } from "@/hooks/useDashboardData";

export default function SignalsFeed() {
  const { signals } = useDashboardData();
  const rows = signals.data ?? [];
  return (
    <div>
      <h3 className="font-semibold mb-2">Live Signals</h3>
      {rows.length === 0 ? <div className="text-sm text-gray-500">No signals yet</div> :
      <table className="w-full text-sm">
        <thead><tr><th>Time</th><th>Strategy</th><th>Symbol</th><th>Action</th><th>Size</th><th>Reason</th></tr></thead>
        <tbody>
          {rows.slice().reverse().map((s, i) => (
            <tr key={i}>
              <td>{new Date(s.ts).toLocaleTimeString()}</td><td>{s.strategy}</td><td>{s.symbol}</td>
              <td>{s.action}</td><td>{s.size}</td><td>{s.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>}
    </div>
  );
}
