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
import PortfolioCard from '@/components/dashboard/PortfolioCard';
import BrainFlowNowCard from '@/components/dashboard/BrainFlowNowCard';
import TickerHighlightsCard from '@/components/dashboard/TickerHighlightsCard';
import BrainScoringActivityCard from '@/components/dashboard/BrainScoringActivityCard';
import BrainEvoFlow from '@/components/BrainEvoFlow';
import NewsSentimentDashboard from '@/components/dashboard/NewsSentimentDashboard';
import PaperExecutionMonitor from '@/components/dashboard/PaperExecutionMonitor';
import AIOrchestratorStatus from '@/components/AIOrchestratorStatus';
import { toPortfolio, toArray } from '@/services/normalize';
import AutoRunnerStrip from '@/components/trading/AutoRunnerStrip';
import ActivityTicker from '@/components/trading/ActivityTicker';
import PriceTape from '@/components/cards/PriceTape';
import ScannerCandidatesCard from '@/components/cards/ScannerCandidatesCard';
import RealtimeBanner from '@/components/Banners/RealtimeBanner';
import EvoTestCard from '@/components/ui/EvoTestCard';

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
    <div className="w-full min-h-screen bg-background">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Header Section */}
        <div className="mb-8">
          <RealtimeBanner />
          <div className="flex items-center justify-between mt-4">
            <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          </div>
        </div>

        {/* Activity Strips - Full Width */}
        <div className="mb-8 space-y-4">
          <AutoRunnerStrip />
          <ActivityTicker />
        </div>

        {/* Price Tape - Full Width */}
        <div className="mb-8">
          <PriceTape />
        </div>

        {/* Main Dashboard Content - Vertical Flow */}
        <ErrorBoundary>
          <div className="dashboard-container">
            {/* Portfolio Summary - Primary Card */}
            <div className="dashboard-section">
              <PortfolioCard />
            </div>

            {/* Brain Flow AI Insights */}
            <div className="dashboard-section">
              <BrainFlowNowCard />
            </div>

            {/* Brain Scoring Activity */}
            <div className="dashboard-section">
              <BrainScoringActivityCard />
            </div>

            {/* Brain & Evo Flow */}
            <div className="dashboard-section">
              <BrainEvoFlow />
            </div>

            {/* AI Orchestrator Status */}
            <div className="dashboard-section">
              <AIOrchestratorStatus />
            </div>

            {/* News Sentiment Dashboard */}
            <div className="dashboard-section">
              <NewsSentimentDashboard />
            </div>

            {/* Paper Execution Monitor */}
            <div className="dashboard-section">
              <PaperExecutionMonitor />
            </div>

            {/* Safety Controls */}
            <div className="dashboard-section">
              <SimpleCard title="Safety Controls">
                <SafetyControls />
              </SimpleCard>
            </div>

            {/* Scanner Candidates */}
            <div className="dashboard-section">
              <ScannerCandidatesCard />
            </div>

            {/* Evolution Testing */}
            <div className="dashboard-section">
              <EvoTestCard />
            </div>

            {/* Market Intelligence */}
            <div className="dashboard-section">
              <TickerHighlightsCard />
            </div>

            {/* Recent Trade Decisions */}
            <div className="dashboard-section">
              <SimpleCard
                title="Recent Trade Decisions"
                action={
                  <Link to="/decisions" className="text-sm text-primary flex items-center hover:text-primary/80 transition-colors">
                    View all <ChevronRight size={16} className="ml-1" />
                  </Link>
                }
                className="w-full"
              >
                <div className="min-h-[280px]">
                  {(() => {
                    try {
                      const decisions = Array.isArray(recentDecisions) ? recentDecisions : [];
                      return decisions.length > 0 ? (
                        <div className="space-y-4">
                          {decisions.slice(0, 4).map((decision, idx) => {
                            const d = decision as any;
                            if (!d?.symbol) return null;
                            return (
                              <div
                                key={String(d.id || `${d.symbol}-${d.timestamp || d.created_at || d.decidedAt || (d.asOf) || idx}-${d.direction || d.action || ''}`)}
                                className="border border-border rounded-lg p-4 bg-card/50 hover:bg-card/70 transition-colors"
                              >
                                <div className="flex items-center justify-between mb-3">
                                  <div className="flex items-center gap-3">
                                    <span className="font-semibold text-lg text-foreground">{d.symbol}</span>
                                    <span className={`text-xs px-3 py-1 rounded-full font-medium
                                      ${d.action === 'buy' ? 'bg-green-500/20 text-green-400 border border-green-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}
                                    >
                                      {(d.action || 'HOLD').toString().toUpperCase()}
                                    </span>
                                    <span className={`text-xs px-3 py-1 rounded-full font-medium
                                      ${d.decided === 'executed' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 'bg-gray-500/20 text-gray-400 border border-gray-500/30'}`}
                                    >
                                      {d.decided === 'executed' ? 'Executed' : 'Pending'}
                                    </span>
                                  </div>
                                  <div className="text-sm font-medium text-foreground">
                                    Score: <span className="text-primary text-lg">{Math.round(numberOr(d.score, 0) * 100)}</span>
                                  </div>
                                </div>
                                <p className="text-sm text-muted-foreground mb-3 leading-relaxed">
                                  {d.reason || d.reasons?.[0] || 'No reason provided'}
                                </p>
                                <div className="flex items-center justify-between text-xs text-muted-foreground">
                                  <span className="font-medium">Strategy: {d.strategy_name || d.strategy || '—'}</span>
                                  <span className="font-medium">
                                    {(() => {
                                      try {
                                        const ts = d.timestamp || d.decidedAt;
                                        return ts ? new Date(ts).toLocaleTimeString() : '—';
                                      } catch { return '—'; }
                                    })()}
                                  </span>
                                </div>
                              </div>
                            );
                          }).filter(Boolean)}
                        </div>
                      ) : (
                        <div className="text-center text-muted-foreground py-12">
                          <TrendingUp className="mx-auto h-12 w-12 mb-4 opacity-50" />
                          <p className="text-lg font-medium">No recent trade decisions</p>
                          <p className="text-sm mt-2">Trade decisions will appear here as they are made.</p>
                        </div>
                      );
                    } catch (error) {
                      console.error('Dashboard decisions error:', error);
                      return (
                        <div className="text-center text-red-500 py-12">
                          <AlertTriangle className="mx-auto h-12 w-12 mb-4" />
                          <p className="text-lg font-medium">Error loading decisions</p>
                          <p className="text-sm mt-2">Please check the console for details.</p>
                        </div>
                      );
                    }
                  })()}
                </div>
              </SimpleCard>
            </div>
          </div>
        </ErrorBoundary>
      </div>
    </div>
  );
};

export default DashboardPage;
