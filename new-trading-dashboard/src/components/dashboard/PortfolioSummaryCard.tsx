import { useQuery } from "@tanstack/react-query";
import type { PortfolioAllocationsResponse } from "@/contracts/types";

function Stat({label,value}:{label:string;value:string}){
  return (
    <div className="rounded-xl border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}

export default function PortfolioSummaryCard(){
  const { data, isLoading } = useQuery<PortfolioAllocationsResponse>({
    queryKey:["portfolio","allocations"],
    queryFn: async ()=> (await fetch("/api/portfolio/allocations")).json(),
    refetchInterval: 15000, staleTime: 10000,
  });

  const d = data?.data;
  return (
    <div className="border rounded-2xl p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold">Portfolio Summary</h3>
        <div className="text-xs text-muted-foreground">as of {data?.meta?.asOf ? new Date(data.meta.asOf).toLocaleTimeString() : new Date().toLocaleTimeString()}</div>
      </div>
      {isLoading || !d ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ):(
        <div className="grid sm:grid-cols-4 gap-4 mt-3">
          <Stat label="Equity" value={`$${d.equity.toLocaleString()}`} />
          <Stat label="Cash" value={`$${d.cash.toLocaleString()}`} />
          <Stat label="Buying Power" value={`$${d.buying_power.toLocaleString()}`} />
          <Stat label="P&L (Today)" value={`$${d.pl_day.toLocaleString()}`} />
          <div className="sm:col-span-2">
            <h4 className="font-medium mb-1">Allocation by Type</h4>
            <ul className="text-sm">
              {d.typeAlloc.map(a=> <li key={a.name}>{a.name} — {a.pct}% (${a.value.toLocaleString()})</li>)}
            </ul>
          </div>
          <div className="sm:col-span-2">
            <h4 className="font-medium mb-1">Top Positions</h4>
            <ul className="text-sm">
              {d.symbolAlloc.map(a=> <li key={a.name}>{a.name} — {a.pct}% (${a.value.toLocaleString()})</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}


