// src/components/trade-decisions/DecisionCard.tsx
import { useState } from 'react';
import { DecisionTrace, proofStrength } from '@/types/DecisionTrace';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';

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
    <Card className="p-4 border border-blue-600 bg-blue-900/40 rounded-xl">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-white">{d.symbol}</h3>
          <Badge className={statusColor()}>{d.execution.status}</Badge>
          <Badge className={strengthColor}>Proof: {strength}</Badge>
        </div>
        <div className="text-xs text-white">
          {formatDate(d.as_of)}
        </div>
      </div>

      <div className="text-sm mb-3 text-white bg-blue-800/50 p-2 rounded">
        {d.explain_layman || 'Trade decision processed'}
      </div>
      
      <Tabs value={tab} onValueChange={(v) => setTab(v as any)} className="w-full">
        <TabsList className="grid grid-cols-4 mb-2 bg-blue-800/60">
          <TabsTrigger value="summary" className="text-white">Summary</TabsTrigger>
          <TabsTrigger value="evidence" className="text-white">Evidence</TabsTrigger>
          <TabsTrigger value="context" className="text-white">Context</TabsTrigger>
          <TabsTrigger value="plan" className="text-white">Plan</TabsTrigger>
        </TabsList>

        {tab === 'summary' && (
          <TabsContent value="summary" className="space-y-2 mt-4">
            {d.explain_detail && d.explain_detail.length > 0 ? (
              <div className="bg-blue-800/50 p-3 rounded">
                <ul className="list-disc pl-5 text-sm text-white space-y-1">
                  {d.explain_detail.map((detail, i) => (
                    <li key={i} className="text-white">{detail}</li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="bg-blue-800/50 p-3 rounded space-y-3">
                <div className="text-white">
                  <strong>Action:</strong> {d.plan?.action || d.action || 'Unknown'}
                </div>
                <div className="text-white">
                  <strong>Symbol:</strong> {d.symbol}
                </div>
                <div className="text-white">
                  <strong>Status:</strong> {d.execution?.status || 'Unknown'}
                </div>
                {d.plan && (
                  <div className="text-white">
                    <strong>Plan:</strong> {d.plan.orderType || 'market'} order for {d.plan.qty || 1} shares
                  </div>
                )}
                {d.signals && d.signals.length > 0 && (
                  <div className="text-white">
                    <strong>Signals:</strong> {d.signals.length} signal(s) detected
                  </div>
                )}
              </div>
            )}
          </TabsContent>
        )}

        {tab === 'evidence' && (
          <TabsContent value="evidence" className="space-y-2 mt-4">
            {d.news_evidence && d.news_evidence.length > 0 ? (
              d.news_evidence.map((e, i) => (
                <div key={i} className="border border-blue-600 rounded-lg p-3 text-sm bg-blue-800/50">
                  <div className="font-medium text-white mb-2">{e.headline}</div>
                  <div className="text-xs text-blue-200 mb-2 flex items-center gap-2">
                    {e.sentiment && <span className={`inline-block px-1.5 py-0.5 rounded text-xs ${
                      e.sentiment === 'positive' ? 'bg-green-600 text-white' :
                      e.sentiment === 'negative' ? 'bg-red-600 text-white' : 'bg-slate-600 text-white'
                    }`}>{e.sentiment}</span>}
                    {e.credibility && <span className="text-blue-200">{e.credibility}</span>}
                    {e.recency_min !== undefined && <span className="text-blue-200"> • {e.recency_min}m ago</span>}
                  </div>
                  <div className="text-sm text-white mb-2">{e.snippet}</div>
                  <div className="mt-2 text-xs">
                    <a className="text-blue-300 hover:text-blue-200 underline" href={e.url} target="_blank" rel="noreferrer">Open source</a>
                  </div>
                </div>
              ))
                ) : (
              <div className="space-y-3">
                <div className="text-white bg-blue-800/50 p-2 rounded">No news evidence attached.</div>
                {d.signals && d.signals.length > 0 && (
                  <div className="border border-blue-600 rounded-lg p-3 bg-blue-800/50">
                    <div className="font-medium text-white mb-2">Detected Signals</div>
                    <div className="space-y-2">
                      {d.signals.map((signal, i) => (
                        <div key={i} className="text-sm text-white">
                          <div className="flex justify-between">
                            <span><strong>{signal.name}</strong></span>
                            <span className={`px-2 py-0.5 rounded text-xs ${
                              signal.direction === 'bullish' ? 'bg-green-600 text-white' :
                              signal.direction === 'bearish' ? 'bg-red-600 text-white' : 'bg-slate-600 text-white'
                            }`}>
                              {signal.direction}
                            </span>
                          </div>
                          <div className="text-xs text-blue-200 mt-1">
                            Source: {signal.source}
                            {signal.value && <span> • Value: {signal.value}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </TabsContent>
        )}
        
        {tab === 'context' && (
          <TabsContent value="context" className="space-y-4 mt-4">
            {(d.market_context || d.risk_gate) ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {d.market_context?.regime && (
                  <div className="border border-blue-600 rounded-lg p-3 bg-blue-800/50">
                    <div className="font-medium text-white mb-1">Regime</div>
                    <div className="text-white">
                      {d.market_context.regime.label}
                      {d.market_context.regime.confidence !== undefined && (
                        <span className="text-blue-200 ml-1">
                          ({Math.round((d.market_context.regime.confidence ?? 0)*100)}%)
                        </span>
                      )}
                    </div>
                  </div>
                )}
                {d.market_context?.volatility && (
                  <div className="border border-blue-600 rounded-lg p-3 bg-blue-800/50">
                    <div className="font-medium text-white mb-1">Volatility</div>
                    <div className="text-white">
                      VIX {d.market_context.volatility.vix}
                    </div>
                  </div>
                )}
                {d.risk_gate && (
                  <div className="border border-blue-600 rounded-lg p-3 bg-blue-800/50 md:col-span-2">
                    <div className="font-medium text-white mb-1">Risk Gates</div>
                    <div className="text-white space-y-1">
                      <div>Position limits: <span className={d.risk_gate.position_limits_ok ? 'text-green-400' : 'text-red-400'}>
                        {formatGate(d.risk_gate.position_limits_ok)}
                      </span></div>
                      <div>Portfolio heat: <span className={d.risk_gate.portfolio_heat_ok ? 'text-green-400' : 'text-red-400'}>
                        {formatGate(d.risk_gate.portfolio_heat_ok)}
                      </span></div>
                      <div>Drawdown: <span className={d.risk_gate.drawdown_ok ? 'text-green-400' : 'text-red-400'}>
                        {formatGate(d.risk_gate.drawdown_ok)}
                      </span></div>
                      {(d.risk_gate.notes ?? []).length > 0 && (
                        <ul className="list-disc ml-5 mt-2 space-y-1">
                          {d.risk_gate.notes.map((n, i) => <li key={i} className="text-white">{n}</li>)}
                        </ul>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-white bg-blue-800/50 p-2 rounded">No market context or risk information available.</div>
            )}
          </TabsContent>
        )}

        {tab === 'plan' && (
          <TabsContent value="plan" className="space-y-4 mt-4">
            {d.plan ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="border border-blue-600 rounded-lg p-3 bg-blue-800/50">
                  <div className="font-medium text-white mb-1">Action</div>
                  <div className="text-white">{d.plan.action || 'Unknown'}</div>
                </div>
                <div className="border border-blue-600 rounded-lg p-3 bg-blue-800/50">
                  <div className="font-medium text-white mb-1">Order Type</div>
                  <div className="text-white">{d.plan.orderType || 'market'}</div>
                </div>
                <div className="border border-blue-600 rounded-lg p-3 bg-blue-800/50">
                  <div className="font-medium text-white mb-1">Quantity</div>
                  <div className="text-white">{d.plan.qty || 1} shares</div>
                </div>
              </div>
            ) : (
              <div className="text-white bg-blue-800/50 p-2 rounded">No detailed trading plan available.</div>
            )}
          </TabsContent>
        )}
      </Tabs>
      
      <div className="mt-4 pt-3 border-t border-blue-600 text-xs text-white">
        trace: {d.trace_id} • {d.schema_version}
      </div>

      {onOpenEvidence && (
        <div className="mt-3">
          <Button
            variant="outline"
            size="sm"
            className="bg-blue-800/60 border-blue-600 text-white hover:bg-blue-700"
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
