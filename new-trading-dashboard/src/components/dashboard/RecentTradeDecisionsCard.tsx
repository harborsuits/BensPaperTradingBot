import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useDecisionsRecent } from "@/hooks/useDecisionsRecent";
import EvidenceDrawer from "@/components/trading/EvidenceDrawer";
import { buildEvidenceFromUi } from "@/lib/evidence/builders";
import type { DecisionRow, ContextRow } from "@/contracts/types";

export default function RecentTradeDecisionsCard(){
  const { data: decisions } = useDecisionsRecent(20);
  const { data: context } = useQuery<ContextRow>({
    queryKey:["context","now"],
    queryFn: async ()=> (await fetch("/api/context")).json(),
    refetchInterval: 45000, staleTime: 30000,
  });

  const rows = useMemo(()=>{
    const map = new Map<string, DecisionRow>();
    (decisions ?? []).forEach(d=>{
      const key = `${d.trace_id ?? d.id ?? d.symbol}-${d.symbol}`;
      const prev = map.get(key);
      if (!prev || new Date(d.createdAt ?? 0) > new Date(prev.createdAt ?? 0)) map.set(key, d);
    });
    return Array.from(map.values()).sort((a,b)=> new Date(b.createdAt ?? 0).getTime() - new Date(a.createdAt ?? 0).getTime()).slice(0,4);
  }, [decisions]);

  const [open, setOpen] = useState(false);
  const [packet, setPacket] = useState<any>(null);

  const openEvidence = (d:DecisionRow)=>{
    const p = buildEvidenceFromUi({ decision: d, context });
    setPacket(p); setOpen(true);
  };

  return (
    <div className="border rounded-2xl p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold">Recent Trade Decisions</h3>
        <a href="/decisions" className="text-xs underline">View all</a>
      </div>
      <div className="mt-2 space-y-2">
        {rows.map(d=> (
          <button key={(d.trace_id ?? d.id) + d.symbol} className="w-full text-left border rounded-xl p-3 hover:bg-muted" onClick={()=>openEvidence(d)}>
            <div className="flex justify-between">
              <div className="font-medium">{d.symbol} • {d.action ?? ""}</div>
              <div className="text-xs text-muted-foreground">{d.createdAt ? new Date(d.createdAt).toLocaleTimeString() : ""}</div>
            </div>
            <div className="text-xs mt-1">Score: {Math.round((d.score ?? 0) * 100)} • {d.one_liner ?? d.reason ?? "—"}</div>
          </button>
        ))}
        {rows.length===0 && <div className="text-sm text-muted-foreground">No recent decisions.</div>}
      </div>

      <EvidenceDrawer open={open} onOpenChange={setOpen} data={packet} />
    </div>
  );
}


