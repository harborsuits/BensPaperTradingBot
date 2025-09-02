import { useDecisionsRecent } from "@/hooks/useDecisionsRecent";
import { usePlacePaperOrder } from "@/hooks/usePlacePaperOrder";
import { Card } from "@/components/ui/Card";

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
  const { data } = useDecisionsRecent(20);
  const { mutate: place, isLoading } = usePlacePaperOrder();

  return (
    <div className="grid gap-3 md:grid-cols-2">
      {(data ?? []).filter((d:any) => d.symbol && d.strategy).map((d:any) => (
        <Card key={d.id || d.symbol} className="card">
          <div className="card-header">
            <div className="card-title">{d.symbol} · {d.strategy} · {d.account ?? "paper"}</div>
            <div className="card-subtle">{formatDecisionTime(d.decidedAt)}</div>
          </div>
          <div className="card-content space-y-2">
            <div className="flex gap-2 flex-wrap">
              {(d.reasons ?? []).slice(0,3).map((r:string)=> <span key={r} className="chip">{r}</span>)}
            </div>
            <div className="text-sm text-gray-600">
              Plan: Size {d.plan?.sizePct}% · SL {d.plan?.slPct}% · TP {d.plan?.tpPct}%
            </div>
            <button className="btn btn-primary"
              disabled={isLoading}
              onClick={()=>place({ symbol: d.symbol, side: "buy", qty: 5, type: "market" })}>
              {isLoading ? "Placing…" : "Paper Order"}
            </button>
          </div>
        </Card>
      ))}
    </div>
  );
}