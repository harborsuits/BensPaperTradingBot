import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, AlertCircle, Clock, BarChart, DollarSign } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Progress } from '@/components/ui/Progress';
import { Badge } from '@/components/ui/Badge';
import { ScrollArea } from '@/components/ui/ScrollArea';
import { api } from '@/services/api';
import { getMeta } from '@/lib/meta';
import { toast } from '@/utils/toast';
import { useHealth } from '@/hooks/useHealth';
import { formatRelativeTime } from '@/utils/date';

export interface TradeCandidate {
  id: string;
  symbol: string;
  strategy_id: string;
  strategy_name: string;
  direction: 'buy' | 'sell';
  score: number;
  entry_price: number;
  target_price: number;
  stop_loss: number;
  potential_profit_pct: number;
  risk_reward_ratio: number;
  confidence: number;
  time_validity: string;
  timeframe: string;
  created_at: string;
  status: 'pending' | 'watching' | 'ready' | 'executed' | 'expired';
  tags: string[];
  entry_conditions: string[];
  indicators: {
    name: string;
    value: number;
    signal: 'bullish' | 'bearish' | 'neutral';
  }[];
}

interface TradeCandidatesProps {
  className?: string;
}

// Normalize raw decision traces into TradeCandidate shape with safe defaults
function normalizeDecision(dec: any): TradeCandidate {
  const score = Number(dec?.candidate_score?.alpha);
  const entryPx = Number(dec?.plan?.entry?.px);
  const tp = Number(dec?.plan?.exits?.take_profit);
  const sl = Number(dec?.plan?.exits?.stop);
  const direction = (String(dec?.plan?.action || '').toLowerCase().includes('sell') ? 'sell' : 'buy') as 'buy'|'sell';
  const signals = Array.isArray(dec?.signals) ? dec.signals : [];
  return {
    id: String(dec?.trace_id || dec?.id || Math.random().toString(36).slice(2)),
    symbol: String(dec?.symbol || 'SPY').toUpperCase(),
    strategy_id: String(dec?.strategy_id || 'unknown'),
    strategy_name: String(dec?.strategy_name || 'Pipeline'),
    direction,
    score: Number.isFinite(score) ? score : 0.5,
    entry_price: Number.isFinite(entryPx) ? entryPx : 0,
    target_price: Number.isFinite(tp) ? tp : 0,
    stop_loss: Number.isFinite(sl) ? sl : 0,
    potential_profit_pct: 0,
    risk_reward_ratio: 1,
    confidence: Number(dec?.market_context?.sentiment?.score ?? 0.5),
    time_validity: String(dec?.execution?.status || 'PROPOSED'),
    timeframe: String(dec?.timeframe || 'intraday'),
    created_at: String(dec?.as_of || new Date().toISOString()),
    status: 'watching',
    tags: Array.isArray(dec?.tags) ? dec.tags : [],
    entry_conditions: Array.isArray(dec?.explain_detail) ? dec.explain_detail : [],
    indicators: signals.map((s: any) => ({ name: String(s?.name || s?.source || 'signal'), value: Number(s?.value ?? 0), signal: (s?.direction === 'bearish' ? 'bearish' : s?.direction === 'bullish' ? 'bullish' : 'neutral') }))
  };
}

// This would ideally be in our tradeApi module
const getTradeCandidates = async () => {
  try {
    // Use existing backend contract: /api/decisions/recent
    const resp = await api.get('/decisions/recent');
    const data = Array.isArray(resp.data)
      ? resp.data
      : Array.isArray((resp.data as any)?.items)
        ? (resp.data as any).items
        : [];
    const mapped = (data as any[]).map(normalizeDecision);
    return { success: true as const, data: mapped as TradeCandidate[] };
  } catch (e: any) {
    return { success: false as const, error: e?.message ?? 'error' };
  }
};

const TradeCandidates: React.FC<TradeCandidatesProps> = ({ className }) => {
  const { data, isLoading, error } = useQuery({
    queryKey: ['trade-candidates'],
    queryFn: async () => {
      const response = await getTradeCandidates();
      return response.success ? response.data : null;
    },
    staleTime: 15000, // 15 seconds - this needs frequent updates
  });

  const getStatusBadge = (status: TradeCandidate['status']) => {
    switch (status) {
      case 'pending':
        return <Badge variant="outline" className="bg-muted/30">Pending</Badge>;
      case 'watching':
        return <Badge variant="secondary">Watching</Badge>;
      case 'ready':
        return <Badge variant="success" className="bg-success/10 text-success">Ready</Badge>;
      case 'executed':
        return <Badge variant="default">Executed</Badge>;
      case 'expired':
        return <Badge variant="outline" className="text-muted-foreground">Expired</Badge>;
    }
  };

  const getDirectionIcon = (direction: 'buy' | 'sell') => {
    return direction === 'buy' 
      ? <TrendingUp className="text-bull" size={16} /> 
      : <TrendingDown className="text-bear" size={16} />;
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-success';
    if (score >= 0.5) return 'text-primary';
    if (score >= 0.3) return 'text-amber-500';
    return 'text-muted-foreground';
  };

  async function dryRun(candidate: TradeCandidate) {
    try {
      const r = await fetch('/api/paper/orders/dry-run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol: candidate.symbol, side: candidate.direction, type: 'market', qty: 1, strategyId: candidate.strategy_id }) });
      const j = await r.json();
      const gate = j.gate || j;
      if (gate.decision === 'REJECT') {
        toast.error(`REJECT: ${gate.reason || 'unknown'}`);
      } else {
        toast.success('PASS: gate accepted (paper)');
      }
    } catch (e: any) {
      toast.error(e?.message || 'Dry-run failed');
    }
  }

  const { data: health } = useHealth();
  const placeDisabled = (health as any)?.breaker && (health as any)?.breaker !== 'GREEN';

  async function place(candidate: TradeCandidate) {
    try {
      const key = 'ui-' + (crypto as any).randomUUID?.() || String(Date.now());
      const r = await fetch('/api/paper/orders', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': key }, body: JSON.stringify({ symbol: candidate.symbol, side: candidate.direction, type: 'market', qty: 1, strategyId: candidate.strategy_id }) });
      const j = await r.json();
      if (r.ok && j?.success) {
        toast.success(`Accepted @ broker id ${j.trade?.id || 'paper'}`);
      } else {
        toast.error(j?.reason || 'Order failed');
      }
    } catch (e: any) {
      toast.error(e?.message || 'Order failed');
    }
  }

  const renderCandidateCard = (candidate: TradeCandidate) => {
    const profitLossRatio = Number.isFinite(candidate.risk_reward_ratio) ? candidate.risk_reward_ratio.toFixed(2) : '—';
    const scorePercentage = Number.isFinite(candidate.score) ? Math.round(candidate.score * 100) : 0;
    
    return (
      <div 
        key={candidate.id} 
        className="p-3 border rounded-lg mb-3 hover:border-primary/30 transition-all"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            {getDirectionIcon(candidate.direction)}
            <span className={`ml-2 font-medium ${candidate.direction === 'buy' ? 'text-bull' : 'text-bear'}`}>
              ${candidate.symbol}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {getStatusBadge(candidate.status)}
            <Badge variant="outline" className="bg-muted/30">{candidate.timeframe}</Badge>
          </div>
        </div>
        
        <div className="mt-2">
          <div className="flex justify-between text-xs mb-1">
            <span className={getScoreColor(candidate.score)}>Score: {scorePercentage}%</span>
            <span>R/R: {profitLossRatio}</span>
          </div>
          <Progress 
            value={scorePercentage} 
            className={`h-2 ${candidate.direction === 'buy' ? 'bg-bull/20' : 'bg-bear/20'}`}
            indicatorClassName={candidate.direction === 'buy' ? 'bg-bull' : 'bg-bear'}
          />
        </div>
        
        <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
          <div className="flex flex-col">
            <span className="text-muted-foreground">Entry</span>
            <span className="font-medium">${Number(candidate.entry_price ?? 0).toFixed(2)}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-muted-foreground">Target</span>
            <span className="font-medium text-success">${Number(candidate.target_price ?? 0).toFixed(2)}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-muted-foreground">Stop</span>
            <span className="font-medium text-destructive">${Number(candidate.stop_loss ?? 0).toFixed(2)}</span>
          </div>
        </div>
        
        <div className="flex justify-between items-center mt-3 text-xs">
          <div className="flex items-center text-muted-foreground">
            <Clock size={12} className="mr-1" />
            {candidate.created_at ? formatRelativeTime(new Date(candidate.created_at)) : '—'}
          </div>
          <div className="flex items-center">
            <span className="text-muted-foreground mr-1">Strategy:</span>
            <span className="font-medium">{candidate.strategy_name}</span>
          </div>
        </div>
        
        {(candidate.entry_conditions?.length ?? 0) > 0 && (
          <div className="mt-2">
            <div className="text-xs text-muted-foreground mb-1">Entry Conditions:</div>
            <div className="flex flex-wrap gap-1">
              {candidate.entry_conditions.map((condition, index) => (
                <div 
                  key={index} 
                  className="text-xs px-2 py-0.5 rounded-full bg-muted/30"
                >
                  {condition}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {(candidate.indicators?.length ?? 0) > 0 && (
          <div className="mt-2 pt-2 border-t border-border/50">
            <div className="text-xs font-medium mb-1">Key Indicators:</div>
            <div className="grid grid-cols-2 gap-2">
              {candidate.indicators.map(indicator => (
                <div key={indicator.name} className="flex items-center justify-between text-xs">
                  <span>{indicator.name}</span>
                  <span className={
                    indicator.signal === 'bullish' ? 'text-bull' : 
                    indicator.signal === 'bearish' ? 'text-bear' : 
                    'text-muted-foreground'
                  }>
                    {Number.isFinite(indicator.value) ? indicator.value.toFixed(2) : '—'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="mt-3 flex items-center gap-2">
          <button onClick={() => dryRun(candidate)} className="px-2 py-1 text-xs rounded border border-neutral-700 hover:bg-neutral-800">Dry-run</button>
          <button
            disabled={placeDisabled}
            title={placeDisabled ? 'Breaker active or data stale' : 'Place paper order'}
            onClick={() => place(candidate)}
            className={`px-2 py-1 text-xs rounded ${placeDisabled ? 'opacity-50 cursor-not-allowed border border-neutral-700' : 'bg-primary text-primary-foreground hover:opacity-90'}`}
          >
            Place (paper)
          </button>
        </div>
      </div>
    );
  };
  
  const renderSkeleton = () => {
    return Array(3).fill(0).map((_, index) => (
      <div key={index} className="p-3 border rounded-lg mb-3">
        <div className="flex items-center justify-between">
          <div className="h-5 bg-muted/20 animate-pulse rounded-md w-1/4"></div>
          <div className="flex items-center gap-2">
            <div className="h-5 bg-muted/20 animate-pulse rounded-md w-16"></div>
            <div className="h-5 bg-muted/20 animate-pulse rounded-md w-16"></div>
          </div>
        </div>
        <div className="mt-2">
          <div className="flex justify-between mb-1">
            <div className="h-4 bg-muted/20 animate-pulse rounded-md w-16"></div>
            <div className="h-4 bg-muted/20 animate-pulse rounded-md w-16"></div>
          </div>
          <div className="h-2 bg-muted/20 animate-pulse rounded-md w-full"></div>
        </div>
        <div className="grid grid-cols-3 gap-2 mt-3">
          <div className="h-10 bg-muted/20 animate-pulse rounded-md"></div>
          <div className="h-10 bg-muted/20 animate-pulse rounded-md"></div>
          <div className="h-10 bg-muted/20 animate-pulse rounded-md"></div>
        </div>
      </div>
    ));
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-base">Trade Candidates</CardTitle>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className="flex items-center">
              <BarChart size={14} className="mr-1" />
              Score
            </div>
            <div className="flex items-center">
              <DollarSign size={14} className="mr-1" />
              Risk/Reward
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="flex items-center text-destructive text-sm">
            <AlertCircle size={16} className="mr-2" />
            Failed to load trade candidates
          </div>
        ) : (
          <ScrollArea className="h-[400px] pr-4">
            {isLoading ? renderSkeleton() : (
              data && data.length > 0 ? (
                data.map(renderCandidateCard)
              ) : (
                <p className="text-sm text-muted-foreground">No trade candidates available</p>
              )
            )}
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
};

export default TradeCandidates;
