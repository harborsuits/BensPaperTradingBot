import { useQuery } from "@tanstack/react-query";
import type { IngestEvent } from "@/contracts/types";
// Removed useState - no longer needed
import { useNavigate } from "react-router-dom";
// Removed EvidenceDrawer and evidence builders - now using DecisionCard for evidence display

const STAGES = ["INGEST","CONTEXT","CANDIDATES","GATES","PLAN","ROUTE","MANAGE","LEARN"] as const;

export default function BrainFlowNowCard(){
  const navigate = useNavigate();
  const { data: events } = useQuery<IngestEvent[]>({
    queryKey:["ingestion","events"],
    queryFn: async ()=> (await fetch("/api/ingestion/events?limit=200")).json(),
    refetchInterval: 8000, staleTime: 5000,
  });
  // Removed context query - no longer needed since we're navigating to decisions page

  // Group by symbol to show all active symbols, not just latest traces
  const symbolMap = new Map<string, IngestEvent>();
  (events ?? []).forEach(e=>{
    const prev = symbolMap.get(e.symbol);
    if (!prev || new Date(e.ts) > new Date(prev.ts)) symbolMap.set(e.symbol, e);
  });
  const activeSymbols = Array.from(symbolMap.values()).sort((a,b)=> new Date(b.ts).getTime() - new Date(a.ts).getTime());

  const openEvidence = (ev:IngestEvent)=>{
    // Always navigate to the decisions page - let the DecisionCard handle evidence display
    const symbol = ev.symbol;
    const trace = ev.trace_id;
    const params = new URLSearchParams();
    if (symbol) params.set("symbol", symbol);
    if (trace) params.set("trace", String(trace));
    navigate(`/decisions?${params.toString()}`);
  };

  return (
    <div className="border rounded-2xl p-4 w-full max-w-full overflow-x-auto">
      <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2">
        <h3 className="text-lg font-semibold">🧠 Brain Flow (Discovery Mode)</h3>
        <div className="text-xs text-muted-foreground">{activeSymbols.length} active • open discovery • last 5m</div>
      </div>
      {activeSymbols.length===0 ? (
        <div className="text-sm text-muted-foreground mt-2">No active symbols in the last 5 minutes.</div>
      ) : (
        <div className="mt-2 space-y-3 max-h-[400px] overflow-y-auto scrollbar-thin scrollbar-thumb-muted scrollbar-track-muted/30">
          {activeSymbols.map(item=>{
            const reached = STAGES.indexOf(item.stage as any);
            return (
              <div key={(item.trace_id ?? "")+item.symbol} className="border rounded-xl p-3 w-full">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="font-medium truncate">{item.symbol}</div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(item.ts).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                    </div>
                  </div>
                  <button onClick={()=>openEvidence(item)} className="text-xs underline whitespace-nowrap">Open Evidence</button>
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {STAGES.map((s,i)=>{
                    const color = i<reached ? "bg-green-600 text-white" : i===reached ? (item.status==="ok"?"bg-amber-500 text-white":"bg-red-600 text-white") : "bg-slate-200 text-slate-700";
                    return <span key={s} className={`px-1.5 py-0.5 rounded text-xs ${color} whitespace-nowrap`}>{s}</span>;
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


