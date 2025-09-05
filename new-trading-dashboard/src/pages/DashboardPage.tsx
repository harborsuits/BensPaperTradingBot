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
import PortfolioSummaryCard from '@/components/dashboard/PortfolioSummaryCard';
import RecentTradeDecisionsCard from '@/components/dashboard/RecentTradeDecisionsCard';
import BrainFlowNowCard from '@/components/dashboard/BrainFlowNowCard';
import TickerHighlightsCard from '@/components/dashboard/TickerHighlightsCard';
import RecentAlertsCard from '@/components/dashboard/RecentAlertsCard';
import { toPortfolio, toArray } from '@/services/normalize';
import UniverseSwitcher from '@/components/UniverseSwitcher';
import AutoRunnerStrip from '@/components/trading/AutoRunnerStrip';
import ActivityTicker from '@/components/trading/ActivityTicker';
import CandidateCard from '@/components/trading/CandidateCard';
import TradeCandidates from '@/components/strategy/TradeCandidates';
import PriceTape from '@/components/cards/PriceTape';
import ScannerCandidatesCard from '@/components/cards/ScannerCandidatesCard';
import RealtimeBanner from '@/components/Banners/RealtimeBanner';

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
  
  // Live portfolio disabled for now; focusing on paper trading only

  // Normalize portfolio responses to a stable shape
  const paperPortfolio = toPortfolio(portfolioData?.data);
  // const livePortfolio = toPortfolio(livePortfolioData?.data);
  
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
      <RealtimeBanner />
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <UniverseSwitcher />
      </div>

      {/* Activity loop strip and ingestion activity ticker */}
      <div className="space-y-3 mb-6">
        <AutoRunnerStrip />
        <ActivityTicker />
      </div>
      
      {/* Live prices */}
      <div className="mb-6">
        <PriceTape />
      </div>

      {/* Stacked full-width cards */}
      <ErrorBoundary>
      <div className="space-y-6">
        <PortfolioSummaryCard />
        <RecentTradeDecisionsCard />
        <BrainFlowNowCard />
        <TickerHighlightsCard />
        <ScannerCandidatesCard />

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

        {/* Removed extra Brain Flow card (kept BrainFlowNowCard above) */}

        {/* Removed Market Context per request */}

        {/* Removed Active Strategies per request */}

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

      <RecentAlertsCard />
    </div>
  );
};

export default DashboardPage;
