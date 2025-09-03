import { FormEvent, useState } from "react";
import { useDashboardData } from "@/hooks/useDashboardData";

export default function QuickTrade() {
  const { placeOrder } = useDashboardData();
  const [symbol, setSymbol] = useState("AAPL");
  const [qty, setQty] = useState(1);
  const [type, setType] = useState<"market"|"limit">("limit");
  const [limit, setLimit] = useState<number>(1.00);
  const [side, setSide] = useState<"buy"|"sell">("buy");

  const submit = (e:FormEvent) => {
    e.preventDefault();
    placeOrder.mutate({ symbol, side, qty, type, limit_price: type === "limit" ? limit : undefined });
  };

  return (
    <form onSubmit={submit} className="flex gap-2 items-end">
      <div><label>Symbol</label><input value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())} className="border p-1 w-24"/></div>
      <div><label>Qty</label><input type="number" value={qty} onChange={e=>setQty(+e.target.value)} className="border p-1 w-20"/></div>
      <div><label>Type</label>
        <select value={type} onChange={e=>setType(e.target.value as any)} className="border p-1">
          <option value="market">market</option><option value="limit">limit</option>
        </select>
      </div>
      <div><label>Limit</label><input type="number" step="0.01" value={limit} disabled={type!=="limit"} onChange={e=>setLimit(+e.target.value)} className="border p-1 w-28"/></div>
      <div><label>Side</label>
        <select value={side} onChange={e=>setSide(e.target.value as any)} className="border p-1">
          <option value="buy">buy</option><option value="sell">sell</option>
        </select>
      </div>
      <button className="px-3 py-2 bg-black text-white rounded">Place</button>
    </form>
  );
}
