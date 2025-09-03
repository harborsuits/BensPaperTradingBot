import { useQuery } from "@tanstack/react-query";
import type { IngestEvent } from "@/contracts/types";

export default function TickerHighlightsCard(){
  const { data } = useQuery<IngestEvent[]>({
    queryKey:["ingestion","events","highlights"],
    queryFn: async ()=> (await fetch("/api/ingestion/events?limit=30")).json(),
    refetchInterval: 8000, staleTime: 5000,
  });

  const now = Date.now();
  const rows = (data ?? [])
    .filter(e=> e.status==="ok" && now - new Date(e.ts).getTime() <= 5*60*1000)
    .sort((a,b)=> new Date(b.ts).getTime() - new Date(a.ts).getTime())
    .slice(0,10);

  return (
    <div className="border rounded-2xl p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold">Ticker Highlights</h3>
        <div className="text-xs text-muted-foreground">last 5m</div>
      </div>
      <div className="mt-2 space-y-1">
        {rows.map(r=> (
          <a key={(r.trace_id ?? "")+r.symbol+r.ts} href={`/decisions?symbol=${encodeURIComponent(r.symbol)}${r.trace_id?`&trace=${encodeURIComponent(r.trace_id)}`:""}`} className="block border rounded-xl p-2 hover:bg-muted">
            <div className="flex justify-between"><span className="font-medium">{r.symbol}</span><span className="text-xs text-muted-foreground">{new Date(r.ts).toLocaleTimeString()}</span></div>
            <div className="text-xs text-muted-foreground">{r.stage} • {r.status}</div>
          </a>
        ))}
        {rows.length===0 && <div className="text-sm text-muted-foreground">No highlights in the last 5 minutes.</div>}
      </div>
    </div>
  );
}


