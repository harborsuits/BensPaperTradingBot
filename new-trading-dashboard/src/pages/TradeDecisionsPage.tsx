import React, { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useDecisionsRecent } from "@/hooks/useDecisionsRecent";
import { usePlacePaperOrder } from "@/hooks/usePlacePaperOrder";
import { useHealth } from "@/hooks/useHealth";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Lock, CheckCircle, XCircle } from "lucide-react";
import { fmtContextBar, fmtTimeAgo } from "@/utils/formatters";
import EvidenceDrawer from "@/components/trading/EvidenceDrawer";
import { buildEvidenceFromUi, enrichWithWhy } from "@/lib/evidence/builders";
import DecisionCard from "@/components/decisions/DecisionCard";
import type { IngestEvent, DecisionRow } from "@/contracts/types";
import { enrichDecisionsWithStage } from "@/lib/flow/utils";

function TraceLinks({ id }: { id: string }) {
  async function openTrace() {
    try {
      const r = await fetch(`/trace/${id}`);
      const j = await r.json();
      const blob = new Blob([JSON.stringify(j, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
    } catch {}
  }
  async function replay() {
    try {
      const r = await fetch(`/trace/${id}/replay`, { method: 'POST' });
      const j = await r.json();
      alert(j?.ok && JSON.stringify(j.diff) === '{}' ? 'Replay OK ✅' : `Replay diff ❌: ${Object.keys(j?.diff||{}).length}`);
    } catch {}
  }
  return (
    <div className="flex items-center gap-2 text-xs">
      <button onClick={openTrace} className="px-2 py-1 rounded border border-neutral-700 hover:bg-neutral-800">Trace</button>
      <button onClick={replay} className="px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700">Replay</button>
    </div>
  );
}

// Safe date formatter
function formatDecisionTime(dateStr?: string) {
  if (!dateStr) return "—";
  try {
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return "—";
    return date.toLocaleTimeString();
  } catch {
    return "—";
  }
}

export default function TradeDecisionsPage(){
  const [searchParams, setSearchParams] = useSearchParams();
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const highlightRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [packet, setPacket] = useState<any>(null);

  const symbolFilter = searchParams.get("symbol");
  const traceFilter = searchParams.get("trace");

  const { data: decisions } = useDecisionsRecent(50);
  const { data: health } = useHealth();
  const { mutate: place, isLoading } = usePlacePaperOrder();

  const { data: events } = useQuery<IngestEvent[]>({
    queryKey:["ingestion","events"],
    queryFn: async ()=> (await fetch("/api/ingestion/events?limit=30")).json(),
    refetchInterval: 8000, staleTime: 5000,
  });

  // Fetch open orders for context bar
  const { data: openOrders } = useQuery({
    queryKey: ['paper', 'orders', 'open'],
    queryFn: async () => {
      const res = await fetch('/api/paper/orders/open');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    refetchInterval: 10000,
  });

  // Enrich with latest stage and filter by symbol
  const enriched = useMemo(()=> enrichDecisionsWithStage(decisions ?? [], events ?? []), [decisions, events]);
  const filteredDecisions = useMemo(()=> (
    enriched.filter((d: any)=> !symbolFilter || d.symbol?.toUpperCase() === symbolFilter)
  ), [enriched, symbolFilter]);

  // Find last decision for context bar
  const lastDecision = filteredDecisions[0];

  // Auto-scroll & highlight; open Evidence if trace matches
  useEffect(() => {
    if (traceFilter && filteredDecisions.length > 0) {
      const targetDecision = filteredDecisions.find((d: any) => (d.trace_id ?? d.id) === traceFilter);
      if (targetDecision) {
        setHighlightedId(targetDecision.id);
        setTimeout(() => {
          highlightRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 500);

        // Remove highlight after 3 seconds
        setTimeout(() => setHighlightedId(null), 3000);

        // Open Evidence automatically
        try {
          // @ts-ignore
          const base = buildEvidenceFromUi({ decision: targetDecision, context: undefined });
          setPacket(enrichWithWhy(base));
          setOpen(true);
        } catch {}
      }
    }
  }, [traceFilter, filteredDecisions]);

  const isHealthy = health?.breaker === 'GREEN';

  const contextBarText = symbolFilter && lastDecision
    ? fmtContextBar(symbolFilter, lastDecision, openOrders)
    : symbolFilter
    ? `${symbolFilter} • no recent decisions • orders: ${openOrders?.length || 0}`
    : 'All recent decisions';

  return (
    <div className="container py-6 space-y-6">
      {/* Context Bar */}
      <div className="flex items-center justify-between p-4 border rounded-lg bg-muted/20">
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold">{contextBarText}</h2>
          {symbolFilter && (
            <Badge variant="outline">{symbolFilter}</Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={isHealthy ? "default" : "destructive"}>
            {isHealthy ? <CheckCircle size={12} className="mr-1" /> : <Lock size={12} className="mr-1" />}
            Health: {health?.breaker || 'UNKNOWN'}
          </Badge>
          {!isHealthy && (
            <Badge variant="secondary">
              <AlertTriangle size={12} className="mr-1" />
              Read-only mode
            </Badge>
          )}
        </div>
      </div>

      {/* Decisions Grid */}
      <div className="grid gap-3 md:grid-cols-2">
        {filteredDecisions.length === 0 ? (
          <div className="col-span-full text-center py-12 text-muted-foreground">
            <AlertTriangle className="mx-auto h-8 w-8 mb-2 opacity-50" />
            <p>No decisions found {symbolFilter ? `for ${symbolFilter}` : 'yet'}</p>
          </div>
        ) : (
          filteredDecisions
            .sort((a:any,b:any)=> new Date(b.createdAt ?? 0).getTime() - new Date(a.createdAt ?? 0).getTime())
            .map((d: any) => {
              const key = d.id || d.symbol;
              return (
                <div key={key} ref={highlightedId === d.id ? highlightRef : null} className={`transition-all duration-500 ${highlightedId === d.id ? 'ring-2 ring-primary shadow-lg bg-primary/5' : ''}`}>
                  <DecisionCard d={d} context={undefined as any} onOpenEvidence={(p:any)=> { setPacket(enrichWithWhy(p)); setOpen(true); }} />
                </div>
              );
            })
        )}
      </div>
      <EvidenceDrawer open={open} onOpenChange={setOpen} data={packet} />
    </div>
  );
}