import { useMemo } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/Tabs";
import { Badge } from "@/components/ui/Badge";
import type { EvidencePacket } from "@/contracts/evidence";

type Props = { open: boolean; onOpenChange: (v: boolean) => void; data: EvidencePacket | null };

export default function EvidenceDrawer({ open, onOpenChange, data }: Props) {
  if (!open || !data) return null;

  const gateChip = (g: boolean) => g ? "bg-green-600 text-white" : "bg-red-600 text-white";

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/50" onClick={() => onOpenChange(false)} />
      <div className="absolute right-0 top-0 h-full w-[720px] max-w-full bg-background border-l p-4 overflow-y-auto">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 text-xl font-semibold">
              <span>{data.symbol}</span>
              <Badge>{data.strategyName}</Badge>
              <Badge variant="secondary">{Math.round(data.confidence * 100)}% conf</Badge>
            </div>
            <p className="text-sm text-muted-foreground mt-1">{data.tlDr}</p>
            <div className="flex flex-wrap gap-1 pt-2">
              {data.gates.map((g) => (
                <span key={g.name} className={`px-2 py-0.5 rounded text-xs ${gateChip(g.passed)}`}>
                  {g.name.replaceAll("_", " ")}
                </span>
              ))}
            </div>
          </div>
          <button className="text-sm underline" onClick={() => onOpenChange(false)}>Close</button>
        </div>

        <Tabs defaultValue="summary" className="mt-4">
          <TabsList className="grid grid-cols-4 w-full">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="sources">Sources</TabsTrigger>
            <TabsTrigger value="prediction">Prediction</TabsTrigger>
            <TabsTrigger value="plan">Plan & Risk</TabsTrigger>
          </TabsList>

          <TabsContent value="summary" className="space-y-2">
            <ul className="list-disc pl-5">{data.interpretation.map((b, i) => (<li key={i}>{b}</li>))}</ul>
            <div className="text-xs text-muted-foreground">
              Regime {data.context.regime} • VIX {data.context.vix ?? "—"}
            </div>
          </TabsContent>

          <TabsContent value="sources" className="space-y-2">
            {data.sources.map((s) => (
              <div key={s.id} className="border rounded-xl p-3">
                <div className="flex justify-between gap-2">
                  <a href={s.url} target="_blank" rel="noreferrer" className="font-medium hover:underline">
                    {s.title}
                  </a>
                  <div className="flex gap-2">
                    <Badge variant="outline">{s.publisher}</Badge>
                    <Badge variant="secondary">{s.biasLean}</Badge>
                    <Badge>{s.credibility}/5</Badge>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  pub {new Date(s.publishedAt).toLocaleString()} • captured {new Date(s.capturedAt).toLocaleString()}
                </div>
                {s.keyClaims?.length > 0 && (
                  <ul className="list-disc pl-5 mt-1 text-sm">{s.keyClaims.map((k, i) => (<li key={i}>{k}</li>))}</ul>
                )}
              </div>
            ))}
          </TabsContent>

          <TabsContent value="prediction" className="space-y-1 text-sm">
            <div>{data.prediction.thesis}</div>
            <div>Expected move {data.prediction.expectedMovePct}% • Horizon {data.prediction.horizonHours}h</div>
            <div>Prob {Math.round(data.prediction.prob * 100)}% • p10/p50/p90 {data.prediction.bandsPct.p10}/{data.prediction.bandsPct.p50}/{data.prediction.bandsPct.p90}%</div>
            <div className="text-muted-foreground">Invalidation: {data.prediction.invalidation}</div>
          </TabsContent>

          <TabsContent value="plan" className="space-y-2 text-sm">
            <div className="font-medium">{data.plan.strategyLabel}</div>
            <pre className="bg-muted/40 rounded p-2 overflow-x-auto text-xs">{JSON.stringify(data.plan.params, null, 2)}</pre>
            <div className="font-medium">Orders</div>
            {data.orders.map((o, i) => (
              <div key={i} className="flex flex-wrap gap-2 text-xs">
                <Badge>{o.venue}</Badge>
                <span>{o.instrument}</span>
                <span>{o.symbol}</span>
                <span>qty {o.quantity}</span>
                {o.limit && <span>limit {o.limit}</span>}
                <span>spr {Math.round(o.preTrade.spreadPct * 100) / 100}%</span>
              </div>
            ))}
            <div className="font-medium">Risk</div>
            <div>Max loss ${data.risk.maxLossUsd} • Heat after {Math.round(data.risk.maxPortfolioHeatAfter * 100)}%</div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}


