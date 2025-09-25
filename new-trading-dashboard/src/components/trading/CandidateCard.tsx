import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/Card';
import { format } from 'date-fns';

import { CheckCircle, XCircle, Beaker, Send, AlertTriangle, RefreshCw } from 'lucide-react';
import MarketAwarenessLine from './MarketAwarenessLine';
import { formatCandidateNarrative } from '@/utils/candidateNarrative';
import { useHealth } from '@/hooks/useHealth';
import { useTopCandidate } from '@/queries/useCandidates';
import EmptyState from '@/components/ui/EmptyState';
import { CardSkeleton } from '@/components/ui/CardSkeleton';
// Build trade plan based on real candidate data and policy
const buildTradePlan = (candidate: any, policy: any) => {
  // Use actual price from candidate or market quote
  const entry = candidate.last || candidate.price || candidate.marketPrice || 0;
  
  // Use actual risk from policy or default to a small percentage of account
  const risk_dollars = Math.min(
    policy?.max_risk_dollars || 100,
    policy?.max_position_size || 1000
  );
  
  // Use actual ATR from candidate data for stop calculation
  const atr = candidate.atr || candidate.volatility || (entry * 0.02); // 2% as fallback
  const stopDistance = atr * (policy?.stop_atr_multiplier || 1.5);
  const stop_price = candidate.side === 'buy' 
    ? entry - stopDistance 
    : entry + stopDistance;
  
  // Calculate target based on risk/reward ratio from policy
  const riskRewardRatio = policy?.risk_reward_ratio || 2;
  const target_price = candidate.side === 'buy' 
    ? entry + (stopDistance * riskRewardRatio)
    : entry - (stopDistance * riskRewardRatio);
  
  // Calculate position size based on actual risk
  const position_size = stopDistance > 0 
    ? Math.floor(risk_dollars / stopDistance) 
    : 0;
  
  // Use actual commission/fee data if available
  const commission = policy?.commission_per_share || 0.005;
  const expected_cost_per_100 = (commission * 100).toFixed(2);
  
  // Use actual time horizon from policy or candidate
  const timebox_days = policy?.max_hold_days || candidate.timeHorizon || 5;

  return {
    risk_dollars: risk_dollars.toFixed(2),
    stop_price: stop_price.toFixed(2),
    target_price: target_price.toFixed(2),
    position_size: position_size,
    expected_cost_per_100: expected_cost_per_100,
    timebox_days: timebox_days,
    rationale_text: `Risk $${risk_dollars.toFixed(2)} (R). Stop ~${(stopDistance/atr).toFixed(1)}×ATR, target +${riskRewardRatio}R. Time-box ${timebox_days} days. Cash-only.`,
  };
};

const CandidateCard = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: health } = useHealth();
  const { data: rawCandidate, isLoading, error, refetch } = useTopCandidate();

  const dryRunMutation = useMutation({
    mutationFn: async (candidate) => {
      const res = await fetch('/api/paper/orders/dry-run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: candidate.symbol,
          side: candidate.side,
          qty: candidate.risk.suggestedQty,
          type: 'market',
        }),
      });
      if (!res.ok) {
        throw new Error(`Dry-run error: ${res.status} ${res.statusText}`);
      }
      return res.json();
    },
  });

  const placeOrderMutation = useMutation({
    mutationFn: async (candidate) => {
      const res = await fetch('/api/paper/orders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: candidate.symbol,
          side: candidate.side,
          qty: candidate.risk.suggestedQty,
          type: 'market',
        }),
      });
      if (!res.ok) {
        throw new Error(`Place order error: ${res.status} ${res.statusText}`);
      }
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paper', 'orders'] });
      queryClient.invalidateQueries({ queryKey: ['paper', 'positions'] });
      queryClient.invalidateQueries({ queryKey: ['paper', 'account'] });
    },
  });

  if (isLoading) {
    return <CardSkeleton />;
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <EmptyState
            title="Couldn't load candidate"
            description={error.message}
            icon="error"
            action={
              <Button size="sm" variant="outline" onClick={() => refetch()}>
                <RefreshCw className="mr-2 h-4 w-4" /> Retry
              </Button>
            }
          />
        </CardContent>
      </Card>
    );
  }

  if (!rawCandidate) {
    return (
      <Card>
        <CardContent>
          <EmptyState
            title="No candidates available"
            description="The scanner hasn't found any candidates matching your criteria."
            icon="empty"
            showHealth={true}
          />
        </CardContent>
      </Card>
    );
  };

  // ✅ Only render dynamic body once we *have* a candidate
  const candidate = formatCandidateNarrative(rawCandidate);
  const narrative = candidate.narrative;
  const isPlaceOrderEnabled = health?.breaker === 'GREEN';

  // Extract key data for display
  const symbol = candidate.symbol || 'UNKNOWN';
  const side = candidate.side || 'buy';
  const entry = candidate.plan?.entry || candidate.last || 100;
  const score = candidate.score || 0;
  const confidence = candidate.confidence || 0;

  // Get policy from candidate or use defaults based on account data
  const policy = candidate.policy || {
    max_risk_dollars: candidate.maxRisk || 50,
    equity: candidate.accountEquity || 100000,
    risk_pct_equity: candidate.riskPercent || 0.005,
    atr_stop_mult: candidate.stopMultiplier || 1.5,
    target_r_mult: candidate.targetMultiplier || 2,
    commission_per_share: candidate.commission || 0.005,
    max_hold_days: candidate.maxHoldDays || 5
  };
  const plan = buildTradePlan(candidate, policy);
  
  // Get watermark timestamp if available
  const asOf = rawCandidate.asOf ? format(new Date(rawCandidate.asOf), 'HH:mm:ss') : null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-bold">{symbol}</h3>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
              side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {side.toUpperCase()}
            </span>
            <span className="text-sm text-muted-foreground">
              ${entry.toFixed(2)}
            </span>
            {asOf && (
              <span className="text-xs text-muted-foreground ml-2">
                as of {asOf}
              </span>
            )}
          </div>
          <div className="text-right">
            <div className="text-sm font-medium">
              Score: {(score * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-muted-foreground">
              Conf: {(confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 space-y-4 text-sm">
          {/* Go / No-go at top with layman details */}
          <div className="rounded-md border border-border p-3 bg-muted/30">
            <h4 className="font-semibold mb-1">Go / No-go</h4>
            <p className="text-foreground mb-2 flex items-center">
              {narrative.goNoGo.includes('SKIP') ? <XCircle size={16} className="mr-2 text-red-500" /> : <CheckCircle size={16} className="mr-2 text-green-500" />}
              {narrative.goNoGo}
            </p>
            <ul className="text-muted-foreground text-sm list-disc list-inside space-y-1">
              <li>News tone: {rawCandidate?.explain?.impact24h && rawCandidate.explain.impact24h !== 0 ? (rawCandidate.explain.impact24h > 0 ? 'leaning positive' : 'leaning negative') : 'mixed'}; headlines count last 24h: {rawCandidate?.explain?.count24h ?? 'n/a'}.</li>
              <li>Liquidity: relative volume ~{(rawCandidate?.explain?.rvol ?? 1).toFixed(1)}×; spread ~{(rawCandidate?.explain?.spreadPct ?? 0.5).toFixed(2)}%.</li>
              <li>Policy fit: tiny probe only; cash-only; max loss ≈ ${plan.risk_dollars}.</li>
            </ul>
          </div>

          {/* Main narrative sections */}
          <div>
            <h4 className="font-semibold mb-1">Why this is on the list</h4>
            <p className="text-muted-foreground">{narrative.whyOnList}</p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">How it fits the market right now</h4>
            <p className="text-muted-foreground">{narrative.marketFit}</p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">What it costs & what could go wrong (plain)</h4>
            <ul className="list-disc list-inside text-muted-foreground space-y-1">
              <li>{narrative.costs.feesAndSlip}</li>
              <li>{narrative.costs.priceWiggle}</li>
              <li>{narrative.costs.riskMeter}</li>
            </ul>
          </div>
           <div>
            <h4 className="font-semibold mb-1">Profit plan (the brain’s plan)</h4>
            <p className="text-muted-foreground">{plan.rationale_text}</p>
          </div>
          <div className="w-full h-[1px] bg-border my-4" />
          <div>
            <h4 className="font-semibold mb-1">Go / No-go</h4>
            <p className="text-muted-foreground flex items-center">
              {narrative.goNoGo.startsWith('SKIP') ? <XCircle size={16} className="mr-2 text-red-500" /> : <CheckCircle size={16} className="mr-2 text-green-500" />}
              {narrative.goNoGo}
            </p>
          </div>
        </div>

        <div className="space-y-3">
          <Button className="w-full" onClick={() => dryRunMutation.mutate(candidate)}>
            <Beaker size={16} className="mr-2" /> Dry-run
          </Button>
          <div className="relative">
            <Button className="w-full" onClick={() => placeOrderMutation.mutate(candidate)} disabled={!isPlaceOrderEnabled}>
              <Send size={16} className="mr-2" /> Place (paper)
            </Button>
            {!isPlaceOrderEnabled && (
               <div className="text-xs text-muted-foreground text-center mt-1">Health must be GREEN to send.</div>
            )}
          </div>
          <Button
            variant="outline"
            className="w-full"
            onClick={() => navigate(`/decisions?symbol=${encodeURIComponent(symbol)}`)}
            aria-label={`View decisions for ${symbol}`}
          >
            Trace & Decisions
          </Button>
          {dryRunMutation.data && (
            <div className="text-xs p-2 bg-muted rounded-md">
              <pre>{JSON.stringify(dryRunMutation.data, null, 2)}</pre>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="p-4 border-t bg-muted/50">
        <MarketAwarenessLine />
      </CardFooter>
    </Card>
  );
};

export default CandidateCard;
