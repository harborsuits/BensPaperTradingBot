import { useQuery } from "@tanstack/react-query";
import type { IngestEvent, DecisionRow, ContextRow } from "@/contracts/types";
import { useState } from "react";
import EvidenceDrawer from "@/components/trading/EvidenceDrawer";
import { buildEvidenceFromUi } from "@/lib/evidence/builders";

const STAGES = ["INGEST","CONTEXT","CANDIDATES","GATES","PLAN","ROUTE","MANAGE","LEARN"] as const;

export default function BrainFlowNowCard(){
  const { data: events } = useQuery<IngestEvent[]>({
    queryKey:["ingestion","events"],
    queryFn: async ()=> (await fetch("/api/ingestion/events?limit=30")).json(),
    refetchInterval: 8000, staleTime: 5000,
  });
  const { data: decisions } = useQuery<DecisionRow[]>({
    queryKey:["decisions","recent"],
    queryFn: async ()=> (await fetch("/api/decisions/recent")).json(),
    refetchInterval: 10000, staleTime: 7000,
  });
  const { data: context } = useQuery<ContextRow>({
    queryKey:["context","now"],
    queryFn: async ()=> (await fetch("/api/context")).json(),
    refetchInterval: 45000, staleTime: 30000,
  });

  const latestMap = new Map<string, IngestEvent>();
  (events ?? []).forEach(e=>{
    const k = `${e.symbol}|${e.trace_id}`;
    const prev = latestMap.get(k);
    if (!prev || new Date(e.ts) > new Date(prev.ts)) latestMap.set(k, e);
  });
  const latest = Array.from(latestMap.values()).slice(0,4);

  const [open, setOpen] = useState(false);
  const [packet, setPacket] = useState<any>(null);
  const openEvidence = (ev:IngestEvent)=>{
    try {
      const d = (decisions ?? []).find(x=> (x.trace_id && ev.trace_id && x.trace_id===ev.trace_id) || x.symbol===ev.symbol);
      if (!d) { setPacket(null); setOpen(false); return; }
      const p = buildEvidenceFromUi({ decision: d, context });
      setPacket(p);
      setOpen(true);
    } catch (e) {
      console.error("Failed to build evidence", e);
      setPacket(null);
      setOpen(false);
    }
  };

  return (
    <div className="border rounded-2xl p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold">Brain Flow (Now)</h3>
        <div className="text-xs text-muted-foreground">last 5m</div>
      </div>
      {latest.length===0 ? (
        <div className="text-sm text-muted-foreground mt-2">No active traces in the last 5 minutes.</div>
      ) : (
        <div className="mt-2 space-y-3">
          {latest.map(item=>{
            const reached = STAGES.indexOf(item.stage as any);
            return (
              <div key={(item.trace_id ?? "")+item.symbol} className="border rounded-xl p-3">
                <div className="flex items-center justify-between">
                  <div className="font-medium">{item.symbol}</div>
                  <button onClick={()=>openEvidence(item)} className="text-xs underline">Open Evidence</button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {STAGES.map((s,i)=>{
                    const color = i<reached ? "bg-green-600 text-white" : i===reached ? (item.status==="ok"?"bg-amber-500 text-white":"bg-red-600 text-white") : "bg-slate-200 text-slate-700";
                    return <span key={s} className={`px-2 py-0.5 rounded text-xs ${color}`}>{s}</span>;
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
      <EvidenceDrawer open={open} onOpenChange={setOpen} data={packet} />
    </div>
  );
}


