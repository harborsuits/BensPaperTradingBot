import { useDashboardData } from "@/hooks/useDashboardData";

export default function OpenOrders() {
  const { ordersOpen, cancelOrder } = useDashboardData();
  const rows = ordersOpen.data ?? [];
  return (
    <div>
      <h3 className="font-semibold mb-2">Open Orders</h3>
      {rows.length === 0 ? <div className="text-sm text-gray-500">No open orders</div> :
      <table className="w-full text-sm">
        <thead><tr><th>ID</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Type</th><th>Limit</th><th>Status</th><th></th></tr></thead>
        <tbody>
          {rows.map((o) => (
            <tr key={o.id}>
              <td>{o.id}</td><td>{o.symbol}</td><td>{o.side}</td><td>{o.qty}</td>
              <td>{o.type}</td><td>{o.limit_price ?? "-"}</td><td>{o.status}</td>
              <td><button onClick={() => cancelOrder.mutate(o.id)} className="px-2 py-1 bg-gray-200 rounded">Cancel</button></td>
            </tr>
          ))}
        </tbody>
      </table>}
    </div>
  );
}
