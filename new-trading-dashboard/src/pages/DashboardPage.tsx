import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { TrendingUp, TrendingDown, ChevronRight, AlertTriangle, ShieldAlert } from 'lucide-react';

import { contextApi, strategyApi, portfolioApi, decisionApi, loggingApi, ingestionApi } from '@/services/api';
import { qk } from '@/services/qk';
import { useWebSocketChannel, useWebSocketMessage } from '@/services/websocket';
import { MarketContext, Strategy, LogEvent, TradeCandidate, DataStatusSummary, DataSourceStatusModel, IngestionMetricsModel, SafetyStatus } from '@/types/api.types';
import { StatusBadge } from '@/components/ui/StatusBadge';
import SafetyControls from '@/components/trading/SafetyControls';
import { ErrorBoundary } from '@/components/util/ErrorBoundary';
import { SimpleCard } from '@/components/ui/SimpleCard';
import ActiveStrategiesCard from '@/components/cards/ActiveStrategiesCard';
import CardFrame from '@/components/CardFrame';
import { formatAsOf } from '@/lib/staleness';
import { toPortfolio, toArray } from '@/services/normalize';
import UniverseSwitcher from '@/components/UniverseSwitcher';
import AutoRunnerStrip from '@/components/trading/AutoRunnerStrip';
import ActivityTicker from '@/components/trading/ActivityTicker';
import LoopStripBanner from '@/components/trading/LoopStripBanner';
import CandidateCard from '@/components/trading/CandidateCard';
import TradeCandidates from '@/components/strategy/TradeCandidates';

// Helpers to make rendering resilient to undefined data
const asArray = <T,>(v: T[] | undefined | null): T[] => (Array.isArray(v) ? v : []);
const numberOr = (v: unknown, d = 0): number => (typeof v === 'number' && !Number.isNaN(v) ? v : d);

const DashboardPage: React.FC = () => {
  // State for data that might be updated via WebSocket
  const [marketContext, setMarketContext] = useState<MarketContext | null>(null);
  const [activeStrategies, setActiveStrategies] = useState<Strategy[]>([]);
  const [recentDecisions, setRecentDecisions] = useState<TradeCandidate[]>([]);
  const [recentAlerts, setRecentAlerts] = useState<LogEvent[]>([]);
  const [dataStatus, setDataStatus] = useState<DataStatusSummary | null>(null);
  const [safetyStatus, setSafetyStatus] = useState<SafetyStatus | null>(null);
  
  // Connect to WebSocket channels
  const { isConnected } = useWebSocketChannel('context', true);
  useWebSocketChannel('strategy', true);
  useWebSocketChannel('trading', true);
  useWebSocketChannel('portfolio', true);
  useWebSocketChannel('logging', true);
  useWebSocketChannel('safety', true);
  const { isConnected: isDataConnected } = useWebSocketChannel('data', true);
  
  // Initial data fetching
  useQuery({
    queryKey: qk.context,
    queryFn: () => contextApi.getMarketContext(),
    refetchInterval: 45_000,
    staleTime: 30_000,
    onSuccess: (response) => {
      if (response.success && response.data) {
        setMarketContext(response.data);
      }
    },
  });
  
  useQuery({
    queryKey: ['dataStatus'],
    queryFn: () => ingestionApi.getDataStatus(),
    onSuccess: (response) => {
      if (response.success && response.data) setDataStatus(response.data);
    },
    refetchInterval: 30_000,
    staleTime: 20_000,
  });

  useQuery({
    queryKey: qk.strategies,
    queryFn: () => strategyApi.getActiveStrategies(),
    refetchInterval: 60_000,
    staleTime: 45_000,
    onSuccess: (response) => {
      if (response.success && response.data) {
        setActiveStrategies(response.data);
      }
    },
  });
  
  useQuery({
    queryKey: qk.decisions(50),
    queryFn: () => decisionApi.getLatestDecisions(),
    refetchInterval: 7_000,
    staleTime: 5_000,
    onSuccess: (response) => {
      if (response.success && response.data) {
        setRecentDecisions(response.data);
      }
    },
  });
  
  const { data: portfolioData } = useQuery({
    queryKey: qk.portfolio('paper'),
    queryFn: () => portfolioApi.getPortfolio('paper'),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
  
  const { data: livePortfolioData } = useQuery({
    queryKey: qk.portfolio('live'),
    queryFn: () => portfolioApi.getPortfolio('live'),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  // Normalize portfolio responses to a stable shape
  const paperPortfolio = toPortfolio(portfolioData?.data);
  const livePortfolio = toPortfolio(livePortfolioData?.data);
  
  useQuery({
    queryKey: ['recentAlerts'],
    queryFn: () => loggingApi.getAlerts(5),
    refetchInterval: 30_000,
    staleTime: 10_000,
    onSuccess: (response) => {
      if (response.success && response.data) {
        setRecentAlerts(response.data);
      }
    },
  });
  
  // WebSocket message handlers
  useWebSocketMessage<MarketContext>(
    'context_update',
    (message) => {
      setMarketContext(message.data);
    },
    []
  );
  
  useWebSocketMessage<Strategy[]>(
    'strategy_update',
    (message) => {
      setActiveStrategies(message.data);
    },
    []
  );
  
  useWebSocketMessage<TradeCandidate[]>(
    'decision_update',
    (message) => {
      const arr = Array.isArray(message?.data) ? (message.data as TradeCandidate[]) : [];
      setRecentDecisions(arr);
    },
    []
  );
  
  useWebSocketMessage<LogEvent>(
    'log',
    (message) => {
      const lvl = (message as any)?.data?.level as string | undefined;
      if (lvl === 'WARNING' || lvl === 'ERROR') {
        const evt = (message as any)?.data as LogEvent | undefined;
        if (evt) setRecentAlerts(prev => [evt, ...prev].slice(0, 5));
      }
    },
    []
  );

  useWebSocketMessage<DataStatusSummary>(
    'ingestion_sources_status',
    (message) => { setDataStatus(prev => prev ? { ...prev, sources: message.data } : null); },
    []
  );
  useWebSocketMessage<IngestionMetricsModel>(
    'ingestion_metrics',
    (message) => { setDataStatus(prev => prev ? { ...prev, metrics: message.data } : prev); },
    []
  );
  
  useWebSocketMessage<SafetyStatus>(
    'safety_status_update',
    (message) => { setSafetyStatus(message.data); },
    []
  );

  // Format asset class badge
  const getAssetClassBadge = (assetClass: string) => {
    const classes: Record<string, string> = {
      stocks: 'bg-blue-800/30 text-blue-300',
      options: 'bg-purple-800/30 text-purple-300',
      forex: 'bg-green-800/30 text-green-300',
      crypto: 'bg-orange-800/30 text-orange-300',
    };
    
    return classes[assetClass] || 'bg-gray-800/30 text-gray-300';
  };

  return (
    <div className="container py-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <UniverseSwitcher />
      </div>

      {/* Activity loop strip and ingestion activity ticker */}
      <div className="space-y-3 mb-6">
        <AutoRunnerStrip />
        <ActivityTicker />
      </div>
      
      {/* Stacked full-width cards */}
      <ErrorBoundary>
      <div className="space-y-6">
        {/* Portfolio Summary first */}
        <SimpleCard title="Portfolio Summary" action={<Link to="/portfolio" className="text-sm text-primary flex items-center">View details <ChevronRight size={16} /></Link>}>
          <div className="space-y-4">
            {/* Paper Trading Account */}
            <div className="border border-border rounded-md p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium flex items-center gap-2">
                  Paper Trading
                  <span className="text-xs bg-blue-800/30 text-blue-300 px-2 py-0.5 rounded-full">
                    Simulated
                  </span>
                </span>
              </div>
              {portfolioData?.success && portfolioData.data ? (
                <>
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    <div className="text-sm">
                      <div className="text-muted-foreground">Total Equity</div>
                      <div className="font-medium">
                        ${paperPortfolio.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                    </div>
                    <div className="text-sm">
                      <div className="text-muted-foreground">Cash Balance</div>
                      <div className="font-medium">
                        ${paperPortfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-muted-foreground">Daily P/L:</span>
                      <span className={(portfolioData as any)?.data?.summary?.daily_pl >= 0 ? 'text-bull' : 'text-bear'}>
                        {((portfolioData as any)?.data?.summary?.daily_pl ?? 0) >= 0 ? '+' : ''}
                        ${Math.abs(((portfolioData as any)?.data?.summary?.daily_pl ?? 0)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        ({(((portfolioData as any)?.data?.summary?.daily_pl_percent ?? 0) as number) >= 0 ? '+' : ''}
                        {Number(((portfolioData as any)?.data?.summary?.daily_pl_percent ?? 0)).toFixed(2)}%)
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">{paperPortfolio.positions.length} positions</span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="h-16 flex items-center justify-center">
                  <p className="text-muted-foreground">Loading...</p>
                </div>
              )}
            </div>
            {/* Live Trading Account */}
            <div className="border border-border rounded-md p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium flex items-center gap-2">
                  Live Trading
                  <span className="text-xs bg-green-800/30 text-green-300 px-2 py-0.5 rounded-full">
                    Real Money
                  </span>
                </span>
              </div>
              {livePortfolioData?.success && livePortfolioData.data ? (
                <>
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    <div className="text-sm">
                      <div className="text-muted-foreground">Total Equity</div>
                      <div className="font-medium">
                        ${livePortfolio.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                    </div>
                    <div className="text-sm">
                      <div className="text-muted-foreground">Cash Balance</div>
                      <div className="font-medium">
                        ${livePortfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-muted-foreground">Daily P/L:</span>
                      <span className={(livePortfolioData as any)?.data?.summary?.daily_pl >= 0 ? 'text-bull' : 'text-bear'}>
                        {((livePortfolioData as any)?.data?.summary?.daily_pl ?? 0) >= 0 ? '+' : ''}
                        ${Math.abs(((livePortfolioData as any)?.data?.summary?.daily_pl ?? 0)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        ({(((livePortfolioData as any)?.data?.summary?.daily_pl_percent ?? 0) as number) >= 0 ? '+' : ''}
                        {Number(((livePortfolioData as any)?.data?.summary?.daily_pl_percent ?? 0)).toFixed(2)}%)
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">{livePortfolio.positions.length} positions</span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="h-16 flex items-center justify-center">
                  <p className="text-muted-foreground">Loading...</p>
                </div>
              )}
            </div>
          </div>
        </SimpleCard>

        {/* Recent Trade Decisions overview */}
        <SimpleCard title="Recent Trade Decisions" action={<Link to="/decisions" className="text-sm text-primary flex items-center">View all <ChevronRight size={16} /></Link>}>
          <div className="min-h-[240px] space-y-3">
            {(() => {
              try {
                const decisions = Array.isArray(recentDecisions) ? recentDecisions : [];
                return decisions.length > 0 ? (
                  decisions.slice(0, 4).map((decision, idx) => {
                    const d = decision as any;
                    if (!d?.symbol) return null;
                    return (
                      <div key={String(d.id || `${d.symbol}-${d.timestamp || d.created_at || d.decidedAt || (d.asOf) || idx}-${d.direction || d.action || ''}`)} className="border border-border rounded-md p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{d.symbol}</span>
                            <span className={`text-xs px-2 py-0.5 rounded-full 
                              ${d.action === 'buy' ? 'bg-bull/20 text-bull' : 'bg-bear/20 text-bear'}`}
                            >
                              {(d.action || 'HOLD').toString().toUpperCase()}
                            </span>
                            <span className={`text-xs px-2 py-0.5 rounded-full
                              ${d.decided === 'executed' ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'}`}
                            >
                              {d.decided === 'executed' ? 'Executed' : 'Pending'}
                            </span>
                          </div>
                          <span className="text-sm flex items-center gap-1">
                            Score: <span className="font-medium">{Math.round(numberOr(d.score, 0) * 100)}</span>
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-1">{d.reason || d.reasons?.[0] || 'No reason provided'}</p>
                        <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
                          <span>Strategy: {d.strategy_name || d.strategy || '—'}</span>
                          <span>{(() => {
                            try {
                              const ts = d.timestamp || d.decidedAt;
                              return ts ? new Date(ts).toLocaleTimeString() : '—';
                            } catch { return '—'; }
                          })()}</span>
                        </div>
                      </div>
                    );
                  }).filter(Boolean)
                ) : (
                  <div className="text-center text-muted-foreground py-8">
                    <TrendingUp className="mx-auto h-8 w-8 mb-2 opacity-50" />
                    <p>No recent trade decisions</p>
                  </div>
                );
              } catch (error) {
                console.error('Dashboard decisions error:', error);
                return (
                  <div className="text-center text-red-500 py-8">
                    <AlertTriangle className="mx-auto h-8 w-8 mb-2" />
                    <p>Error loading decisions</p>
                  </div>
                );
              }
            })()}
          </div>
        </SimpleCard>

        {/* Brain Flow pipeline (arrow chart) */}
        <SimpleCard title="Brain Flow">
          <LoopStripBanner />
        </SimpleCard>

        {/* Market Context */}
        <CardFrame title="Market Context" asOf={marketContext?.timestamp} right={<Link to="/context" className="text-sm text-primary flex items-center">View details <ChevronRight size={16} /></Link>}>
          {marketContext ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-border pb-2">
                <span className="text-muted-foreground">Regime</span>
                <div className="flex items-center gap-2">
                  <span className={`font-medium px-2 py-1 rounded-full text-sm
                    ${marketContext?.regime?.type === 'Bullish' ? 'bg-bull/20 text-bull' : 
                      marketContext?.regime?.type === 'Bearish' ? 'bg-bear/20 text-bear' : 
                      'bg-neutral/20 text-neutral'}`}
                  >
                    {marketContext?.regime?.type ?? 'Unknown'}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {Math.round(numberOr(marketContext?.regime?.confidence, 0) * 100)}%
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between border-b border-border pb-2">
                <span className="text-muted-foreground">Volatility</span>
                <div className="flex items-center gap-2">
                  <span className={`font-medium text-sm
                    ${marketContext?.volatility?.classification === 'High' || marketContext?.volatility?.classification === 'Extreme' 
                      ? 'text-bear' : 'text-foreground'}`}
                  >
                    {numberOr(marketContext?.volatility?.value, 0).toFixed(2)}
                  </span>
                  <span className={`text-sm flex items-center
                    ${numberOr(marketContext?.volatility?.change, 0) > 0 ? 'text-bear' : 'text-bull'}`}
                  >
                    {numberOr(marketContext?.volatility?.change, 0) > 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                    {Math.abs(numberOr(marketContext?.volatility?.change, 0)).toFixed(2)}%
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between border-b border-border pb-2">
                <span className="text-muted-foreground">Sentiment</span>
                <span className={`font-medium text-sm
                  ${numberOr(marketContext?.sentiment?.score, 0.5) > 0.6 ? 'text-bull' : 
                    numberOr(marketContext?.sentiment?.score, 0.5) < 0.4 ? 'text-bear' : 'text-neutral'}`}
                >
                  {numberOr(marketContext?.sentiment?.score, 0.5) > 0.6 ? 'Positive' : 
                    numberOr(marketContext?.sentiment?.score, 0.5) < 0.4 ? 'Negative' : 'Neutral'} 
                  ({numberOr(marketContext?.sentiment?.score, 0.5).toFixed(2)})
                </span>
              </div>
              <div className="text-xs text-muted-foreground">Source timestamp: {formatAsOf(marketContext?.timestamp)}</div>
            </div>
          ) : (
            <div className="h-32 flex items-center justify-center">
              <p className="text-muted-foreground">Loading market context...</p>
            </div>
          )}
        </CardFrame>

        {/* Active Strategies */}
        <ActiveStrategiesCard />

        {/* Ticker Highlights (to be implemented next) */}

        {/* Safety Controls */}
        <SimpleCard title="Safety Controls">
          <SafetyControls />
        </SimpleCard>
      </div>
      </ErrorBoundary>

      {/* Candidate card and trade candidates stacked */}
      <div className="space-y-6 mb-6">
        <CandidateCard />
        <TradeCandidates />
      </div>

      {/* Recent Alerts full width */}
      <SimpleCard title="Recent Alerts" action={<Link to="/logs" className="text-sm text-primary flex items-center">View all logs <ChevronRight size={16} /></Link>}>
        <div className="max-h-[360px] min-h-[240px] overflow-auto space-y-2" role="log" aria-live="polite" aria-relevant="additions">
          {asArray(recentAlerts).length > 0 ? (
            asArray(recentAlerts).filter(Boolean).map((alert: any, idx: number) => (
              <div key={String(alert?.id ?? idx)} className="border border-border rounded-md p-3">
                <div className="flex items-start gap-3">
                  <div className={`mt-0.5 rounded-full p-1
                    ${alert?.level === 'ERROR' ? 'bg-bear/10 text-bear' : 
                      alert?.level === 'WARNING' ? 'bg-highImpact/10 text-highImpact' : 
                      'bg-info/10 text-info'}`}
                  >
                    <AlertTriangle size={16} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className={`text-sm font-medium
                        ${alert?.level === 'ERROR' ? 'text-bear' : 
                          alert?.level === 'WARNING' ? 'text-highImpact' : 
                          'text-info'}`}
                      >
                        {alert?.level ?? 'INFO'}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {alert?.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : '—'}
                      </span>
                    </div>
                    <p className="text-sm truncate">{alert?.message ?? ''}</p>
                    <div className="text-xs text-muted-foreground mt-1">
                      Source: {alert?.source ?? '—'}
                    </div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="h-40 flex items-center justify-center">
              <p className="text-muted-foreground">No recent alerts</p>
            </div>
          )}
        </div>
      </SimpleCard>
    </div>
  );
};

export default DashboardPage;
