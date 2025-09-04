// src/components/trade-decisions/DecisionCard.tsx
import { useState } from 'react';
import { DecisionTrace, proofStrength } from '@/types/DecisionTrace';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';

interface DecisionCardProps {
  d: DecisionTrace;
  onOpenEvidence?: (d: DecisionTrace) => void;
}

export function DecisionCard({ d, onOpenEvidence }: DecisionCardProps) {
  const [tab, setTab] = useState<'summary' | 'evidence' | 'context' | 'plan'>('summary');
  
  // Format helpers
  const formatDate = (isoString: string) => {
    try {
      return new Date(isoString).toLocaleString();
    } catch (e) {
      return isoString;
    }
  };
  
  // Determine status color
  const statusColor = () => {
    switch (d.execution.status) {
      case 'BLOCKED': return 'bg-red-600 text-white';
      case 'FILLED': return 'bg-green-600 text-white';
      case 'SENT': 
      case 'PARTIAL': return 'bg-amber-500 text-white';
      default: return 'bg-blue-600 text-white';
    }
  };
  
  const strength = proofStrength(d);
  const strengthColor = {
    'Strong': 'bg-green-600 text-white',
    'Medium': 'bg-amber-500 text-white',
    'Weak': 'bg-red-600 text-white'
  }[strength];

  return (
    <Card className="p-4 border border-slate-700 bg-slate-800/50 rounded-xl">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold">{d.symbol}</h3>
          <Badge className={statusColor()}>{d.execution.status}</Badge>
          <Badge className={strengthColor}>Proof: {strength}</Badge>
        </div>
        <div className="text-xs text-muted-foreground">
          {formatDate(d.as_of)}
        </div>
      </div>
      
      <div className="text-sm mb-3">{d.explain_layman}</div>
      
      <Tabs value={tab} onValueChange={(v) => setTab(v as any)} className="w-full">
        <TabsList className="grid grid-cols-4 mb-2">
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="evidence">Evidence</TabsTrigger>
          <TabsTrigger value="context">Context</TabsTrigger>
          <TabsTrigger value="plan">Plan</TabsTrigger>
        </TabsList>
        
        {tab === 'summary' && (
          <TabsContent value="summary" className="space-y-2">
            <ul className="list-disc pl-5 text-sm">
              {d.explain_detail.map((detail, i) => (
                <li key={i}>{detail}</li>
              ))}
            </ul>
          </TabsContent>
        )}
        
        {tab === 'evidence' && (
          <TabsContent value="evidence" className="space-y-2">
            {d.news_evidence.map((e, i) => (
              <div key={i} className="border rounded-lg p-3 text-sm">
                <div className="font-medium">{e.headline}</div>
                <div className="text-xs opacity-70 mb-1">
                  {e.sentiment && <span className={`inline-block px-1.5 py-0.5 rounded mr-1 ${
                    e.sentiment === 'positive' ? 'bg-green-700/30' : 
                    e.sentiment === 'negative' ? 'bg-red-700/30' : 'bg-slate-700/30'
                  }`}>{e.sentiment}</span>}
                  {e.credibility && <span>{e.credibility}</span>}
                  {e.recency_min !== undefined && <span> • {e.recency_min}m ago</span>}
                </div>
                <div className="text-sm opacity-80">{e.snippet}</div>
                <div className="mt-1 text-xs">
                  <a className="underline" href={e.url} target="_blank" rel="noreferrer">Open source</a>
                </div>
              </div>
            ))}
            {(!d.news_evidence || d.news_evidence.length === 0) && (
              <div className="opacity-70">No news evidence attached.</div>
            )}
          </TabsContent>
        )}
        
        {tab === 'context' && (
          <TabsContent value="context" className="grid grid-cols-2 gap-3">
            <div className="border rounded-lg p-3">
              <div className="font-medium">Regime</div>
              <div className="opacity-80">{d.market_context?.regime?.label ?? "–"} ({Math.round((d.market_context?.regime?.confidence ?? 0)*100)}%)</div>
            </div>
            <div className="border rounded-lg p-3">
              <div className="font-medium">Volatility</div>
              <div className="opacity-80">VIX {d.market_context?.volatility?.vix ?? "–"}</div>
            </div>
            <div className="border rounded-lg p-3">
              <div className="font-medium">Risk Gates</div>
              <div className="opacity-80 space-y-1">
                <div>Position limits: {formatGate(d.risk_gate?.position_limits_ok)}</div>
                <div>Portfolio heat: {formatGate(d.risk_gate?.portfolio_heat_ok)}</div>
                <div>Drawdown: {formatGate(d.risk_gate?.drawdown_ok)}</div>
                {(d.risk_gate?.notes ?? []).length > 0 && (
                  <ul className="list-disc ml-5">
                    {d.risk_gate!.notes!.map((n, i) => <li key={i}>{n}</li>)}
                  </ul>
                )}
              </div>
            </div>
          </TabsContent>
        )}
        
        {tab === 'plan' && (
          <TabsContent value="plan" className="grid grid-cols-3 gap-3">
            <div className="border rounded-lg p-3">
              <div className="font-medium">Action</div>
              <div className="opacity-80">{d.plan.action}</div>
            </div>
            <div className="border rounded-lg p-3">
              <div className="font-medium">Entry</div>
              <div className="opacity-80">{d.plan.entry?.type} {d.plan.entry?.px ?? ""}</div>
            </div>
            <div className="border rounded-lg p-3">
              <div className="font-medium">Exits</div>
              <div className="opacity-80">stop: {fmtAny(d.plan.exits?.stop)} | take: {fmtAny(d.plan.exits?.take_profit)}</div>
            </div>
          </TabsContent>
        )}
      </Tabs>
      
      <div className="mt-2 text-xs opacity-60">
        trace: {d.trace_id} • {d.schema_version}
      </div>
      
      {onOpenEvidence && (
        <div className="mt-3">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => onOpenEvidence(d)}
          >
            Open Evidence
          </Button>
        </div>
      )}
    </Card>
  );
}

function formatGate(val?: boolean) {
  if (val === undefined) return "–";
  return val ? "✅ OK" : "❌ Blocked";
}

function fmtAny(x: any) { return x === undefined || x === null ? "–" : String(x); }
