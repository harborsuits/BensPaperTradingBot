// src/pages/TradeDecisions.tsx
import React, { useState } from "react";
import { useDecisionFeed } from "@/hooks/useDecisionFeed";
import { DecisionCard } from "@/components/trade-decisions/DecisionCard";
import { Badge } from "@/components/ui/badge";
import EvidenceDrawer from "@/components/trading/EvidenceDrawer";
import { buildEvidenceFromUi, enrichWithWhy } from "@/lib/evidence/builders";

export default function TradeDecisionsPage() {
  const { decisions, status, error, invalid } = useDecisionFeed({ basePath: "" });
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [evidencePacket, setEvidencePacket] = useState<any>(null);

  // Filter by symbol if selected
  const filteredDecisions = selectedSymbol 
    ? decisions.filter(d => d.symbol === selectedSymbol)
    : decisions;

  // Get unique symbols for filter
  const symbols = Array.from(new Set(decisions.map(d => d.symbol)));

  // Handle opening evidence drawer
  const handleOpenEvidence = (d: any) => {
    try {
      // Convert DecisionTrace to EvidencePacket format
      const baseEvidence = {
        decision: {
          symbol: d.symbol,
          trace_id: d.trace_id,
          score: d.candidate_score?.alpha ? Math.round(d.candidate_score.alpha * 100) : undefined,
          reason: d.explain_layman,
          strategy: d.plan.strategyLabel,
          createdAt: d.as_of,
        },
        context: d.market_context ? {
          regime: d.market_context.regime?.label,
          vix: d.market_context.volatility?.vix,
          bias: d.market_context.sentiment?.label,
        } : undefined,
      };
      
      const packet = buildEvidenceFromUi(baseEvidence);
      setEvidencePacket(enrichWithWhy(packet));
      setEvidenceOpen(true);
    } catch (e) {
      console.error("Failed to build evidence packet:", e);
    }
  };

  return (
    <div className="container mx-auto py-6 space-y-4">
      {/* Header with status */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Trade Decisions</h1>
        <div className="flex items-center gap-2">
          <Badge variant={status === 'live' ? "default" : "destructive"}>
            {status}
          </Badge>
          {error && (
            <Badge variant="destructive">
              Error: {error}
            </Badge>
          )}
          {invalid > 0 && (
            <Badge variant="destructive">
              {invalid} invalid trace(s)
            </Badge>
          )}
        </div>
      </div>

      {/* Symbol filter */}
      <div className="flex flex-wrap gap-2">
        <Badge 
          className="cursor-pointer" 
          variant={selectedSymbol === null ? "default" : "outline"}
          onClick={() => setSelectedSymbol(null)}
        >
          All
        </Badge>
        {symbols.map(symbol => (
          <Badge 
            key={symbol}
            className="cursor-pointer" 
            variant={selectedSymbol === symbol ? "default" : "outline"}
            onClick={() => setSelectedSymbol(symbol)}
          >
            {symbol}
          </Badge>
        ))}
      </div>

      {/* Decision cards */}
      <div className="grid md:grid-cols-2 gap-4">
        {filteredDecisions.length === 0 && (
          <div className="col-span-2 text-center py-10 text-muted-foreground">
            No decisions found.
          </div>
        )}
        {filteredDecisions.map(decision => (
          <DecisionCard 
            key={decision.trace_id} 
            d={decision} 
            onOpenEvidence={handleOpenEvidence}
          />
        ))}
      </div>

      {/* Evidence drawer */}
      <EvidenceDrawer 
        open={evidenceOpen} 
        onOpenChange={setEvidenceOpen} 
        data={evidencePacket} 
      />
    </div>
  );
}

