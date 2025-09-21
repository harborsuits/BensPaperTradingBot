import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  TrendingUp,
  TrendingDown,
  ChevronRight,
  AlertTriangle,
  ShieldAlert,
  Activity,
  Target,
  DollarSign,
  BarChart3,
  Zap,
  Clock,
  Users,
  Brain,
  PlayCircle,
  CheckCircle,
  XCircle,
  Eye,
  RefreshCw
} from 'lucide-react';

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
import PositionsTable from '@/components/PositionsTable';
import TradesTable from '@/components/TradesTable';
import ScannerCandidatesCard from '@/components/cards/ScannerCandidatesCard';
import RealtimeBanner from '@/components/Banners/RealtimeBanner';
import EvoTestCard from '@/components/ui/EvoTestCard';

// Helpers to make rendering resilient to undefined data
const asArray = <T,>(v: T[] | undefined | null): T[] => (Array.isArray(v) ? v : []);
const numberOr = (v: unknown, d = 0): number => (typeof v === 'number' && !Number.isNaN(v) ? v : d);

// Dashboard Card Components

const HealthCard: React.FC = () => {
  const { data: healthData, isLoading, error } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const response = await fetch('/api/health');
      if (!response.ok) throw new Error('Failed to fetch health');
      return response.json();
    },
    refetchInterval: 30000,
    staleTime: 15000,
  });

  return (
    <div className="border rounded-lg p-4 bg-card">
      <div className="flex items-center gap-2 mb-3">
        <CheckCircle className={`w-4 h-4 ${healthData?.breaker === 'GREEN' ? 'text-green-600' : healthData?.breaker === 'AMBER' ? 'text-yellow-500' : 'text-red-600'}`} />
        <h3 className="font-semibold">Health</h3>
      </div>
      {isLoading ? (
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded"></div>
        </div>
      ) : error ? (
        <div className="text-red-500 text-sm">Error loading health</div>
      ) : healthData ? (
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Breaker:</span>
            {(() => {
              const br = healthData.breaker ?? (healthData.ok === true ? 'GREEN' : (healthData.ok === false ? 'RED' : 'AMBER'));
              const cls = br === 'GREEN' ? 'text-green-600' : br === 'AMBER' ? 'text-yellow-500' : 'text-red-600';
              return <span className={cls}>{br}</span>;
            })()}
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Quotes age:</span>
            {(() => {
              const ageSec = (typeof healthData.marketData?.ageSec === 'number' ? healthData.marketData.ageSec : undefined) ?? (typeof healthData.quote_age_s === 'number' ? healthData.quote_age_s : undefined);
              return <span>{typeof ageSec === 'number' ? `${Math.max(0, Math.round(ageSec))}s` : '—'}</span>;
            })()}
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Broker RTT:</span>
            {(() => {
              const rttMs = (typeof healthData.broker?.rttMs === 'number' ? healthData.broker.rttMs : undefined) ?? (typeof healthData.broker_age_s === 'number' ? Math.round(healthData.broker_age_s * 1000) : undefined);
              return <span>{typeof rttMs === 'number' ? `${rttMs}ms` : '—'}</span>;
            })()}
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">As of:</span>
            {(() => {
              const t = healthData.asOf || healthData.timestamp;
              const d = t ? new Date(t) : null;
              const txt = d && !isNaN(d.getTime()) ? d.toLocaleTimeString() : '—';
              return <span className="text-xs">{txt}</span>;
            })()}
          </div>
        </div>
      ) : (
        <div className="text-muted-foreground text-sm">No health data</div>
      )}
    </div>
  );
};

const AutopilotCard: React.FC = () => {
  const { data: autopilotData, isLoading, error } = useQuery({
    queryKey: ['audit', 'autoloop', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/audit/autoloop/status');
      if (!response.ok) throw new Error('Failed to fetch autopilot status');
      return response.json();
    },
    refetchInterval: 30000,
    staleTime: 15000,
  });

  return (
    <div className="border rounded-lg p-4 bg-card">
      <div className="flex items-center gap-2 mb-3">
        <PlayCircle className="w-4 h-4 text-blue-600" />
        <h3 className="font-semibold">Autopilot</h3>
      </div>
      {isLoading ? (
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded"></div>
        </div>
      ) : error ? (
        <div className="text-red-500 text-sm">Error loading autopilot</div>
      ) : autopilotData ? (
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Mode:</span>
            <span className="font-medium">{autopilotData.mode}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Status:</span>
            <span className={autopilotData.running ? 'text-green-600' : 'text-red-600'}>
              {autopilotData.running ? 'Running' : 'Stopped'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Tick:</span>
            <span>{autopilotData.tick_ms}ms</span>
          </div>
        </div>
      ) : (
        <div className="text-muted-foreground text-sm">No autopilot data</div>
      )}
    </div>
  );
};

const StrategySpotlightCard: React.FC = () => {
  const { data: strategies, isLoading, error } = useQuery({
    queryKey: ['strategies'],
    queryFn: async () => {
      const response = await fetch('/api/strategies?limit=5');
      if (!response.ok) throw new Error('Failed to fetch strategies');
      const data = await response.json();
      return Array.isArray(data) ? data : data.items || [];
    },
    refetchInterval: 60000,
    staleTime: 30000,
  });

  const topStrategy = strategies?.[0];

  return (
    <div className="border rounded-lg p-4 bg-card">
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 className="w-4 h-4 text-purple-600" />
        <h3 className="font-semibold">Strategy Spotlight</h3>
      </div>
      {isLoading ? (
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded"></div>
        </div>
      ) : error ? (
        <div className="text-red-500 text-sm">Error loading strategies</div>
      ) : topStrategy ? (
        <div className="space-y-1 text-sm">
          <div className="font-medium">{topStrategy.id || topStrategy.name}</div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Win Rate:</span>
            <span className={(topStrategy.win_rate || 0) > 0.6 ? 'text-green-600' : 'text-yellow-600'}>
              {((topStrategy.win_rate || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Sharpe:</span>
            <span className={(topStrategy.sharpe_ratio || 0) > 0.5 ? 'text-green-600' : 'text-yellow-600'}>
              {(topStrategy.sharpe_ratio || 0).toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Trades:</span>
            <span>{topStrategy.trades_count || 0}</span>
          </div>
        </div>
      ) : (
        <div className="text-muted-foreground text-sm">No strategies found</div>
      )}
      <Link to="/decisions?tab=strategies" className="text-xs text-primary hover:text-primary/80 mt-2 inline-block">
        View all strategies →
      </Link>
    </div>
  );
};

const PipelineHealthCard: React.FC = () => {
  const { data: pipelineData, isLoading, error } = useQuery({
    queryKey: ['brain', 'flow', 'summary'],
    queryFn: async () => {
      const response = await fetch('/api/brain/flow/summary?window=15m');
      if (!response.ok) throw new Error('Failed to fetch pipeline health');
      return response.json();
    },
    refetchInterval: 30000,
    staleTime: 15000,
  });

  return (
    <div className="border rounded-lg p-4 bg-card">
      <div className="flex items-center gap-2 mb-3">
        <Activity className="w-4 h-4 text-orange-600" />
        <h3 className="font-semibold">Pipeline Health</h3>
      </div>
      {isLoading ? (
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded"></div>
        </div>
      ) : error ? (
        <div className="text-red-500 text-sm">Error loading pipeline</div>
      ) : pipelineData && pipelineData.counts && pipelineData.by_mode && pipelineData.latency_ms ? (
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Gates Passed:</span>
            <span className="text-green-600">{pipelineData.counts.gates_passed || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Gates Failed:</span>
            <span className="text-red-600">{pipelineData.counts.gates_failed || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Mode:</span>
            <span className="font-medium">{Object.keys(pipelineData.by_mode).find(mode => pipelineData.by_mode[mode] > 0) || 'mixed'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">P50 Latency:</span>
            <span>{pipelineData.latency_ms.p50 || 0}ms</span>
          </div>
        </div>
      ) : (
        <div className="text-muted-foreground text-sm">No pipeline data</div>
      )}
      <Link to="/decisions?tab=pipeline" className="text-xs text-primary hover:text-primary/80 mt-2 inline-block">
        Open Pipeline →
      </Link>
    </div>
  );
};

const DecisionsSummaryCard: React.FC = () => {
  const { data: decisionsData, isLoading, error } = useQuery({
    queryKey: ['decisions', 'summary'],
    queryFn: async () => {
      const response = await fetch('/api/decisions/summary?window=15m');
      if (!response.ok) throw new Error('Failed to fetch decisions summary');
      return response.json();
    },
    refetchInterval: 30000,
    staleTime: 15000,
  });

  return (
    <div className="border rounded-lg p-4 bg-card">
      <div className="flex items-center gap-2 mb-3">
        <Target className="w-4 h-4 text-cyan-600" />
        <h3 className="font-semibold">Decisions Summary</h3>
      </div>
      {isLoading ? (
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded"></div>
        </div>
      ) : error ? (
        <div className="text-red-500 text-sm">Error loading decisions</div>
      ) : decisionsData && typeof decisionsData.proposals_per_min === 'number' && decisionsData.by_stage ? (
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Proposals/min:</span>
            <span className="font-medium">{decisionsData.proposals_per_min.toFixed(1)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Unique Symbols:</span>
            <span>{decisionsData.unique_symbols || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Proposed:</span>
            <span>{decisionsData.by_stage.proposed || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Executed:</span>
            <span>{decisionsData.by_stage.executed || 0}</span>
          </div>
        </div>
      ) : (
        <div className="text-muted-foreground text-sm">No decisions data</div>
      )}
      <Link to="/decisions?tab=proposals" className="text-xs text-primary hover:text-primary/80 mt-2 inline-block">
        View all →
      </Link>
    </div>
  );
};

const OrdersSnapshotCard: React.FC = () => {
  const { data: ordersData, isLoading, error } = useQuery({
    queryKey: ['paper', 'orders'],
    queryFn: async () => {
      const response = await fetch('/api/paper/orders?limit=3');
      if (!response.ok) throw new Error('Failed to fetch orders');
      const data = await response.json();
      return Array.isArray(data) ? data : data.items || [];
    },
    refetchInterval: 15000,
    staleTime: 10000,
  });

  return (
    <div className="border rounded-lg p-4 bg-card">
      <div className="flex items-center gap-2 mb-3">
        <Zap className="w-4 h-4 text-yellow-600" />
        <h3 className="font-semibold">Orders Snapshot</h3>
      </div>
      {isLoading ? (
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded"></div>
        </div>
      ) : error ? (
        <div className="text-red-500 text-sm">Error loading orders</div>
      ) : ordersData ? (
        <div className="space-y-2">
          <div className="text-sm font-medium">
            Open Orders: {ordersData.length}
          </div>
          {ordersData.slice(0, 3).map((order: any, idx: number) => (
            <div key={order.id || idx} className="text-xs border-t pt-2">
              <div className="flex justify-between">
                <span className="font-medium">{order.symbol}</span>
                <span className={`px-2 py-1 rounded text-xs ${
                  order.side === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                }`}>
                  {order.side}
                </span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>{order.qty} @ {order.limit || order.price}</span>
                <span className={`px-2 py-1 rounded text-xs ${
                  order.status === 'FILLED' ? 'bg-green-500/20 text-green-400' :
                  order.status === 'PENDING' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-gray-500/20 text-gray-400'
                }`}>
                  {order.status}
                </span>
              </div>
            </div>
          ))}
          {ordersData.length === 0 && (
            <div className="text-muted-foreground text-sm text-center py-4">
              No open orders
            </div>
          )}
        </div>
      ) : (
        <div className="text-muted-foreground text-sm">No orders data</div>
      )}
      <Link to="/decisions?tab=executions" className="text-xs text-primary hover:text-primary/80 mt-2 inline-block">
        See all orders →
      </Link>
    </div>
  );
};

const LiveRnDCard: React.FC = () => {
  const { data: brainData, isLoading: brainLoading } = useQuery({
    queryKey: ['brain', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/brain/status');
      if (!response.ok) throw new Error('Failed to fetch brain status');
      return response.json();
    },
    refetchInterval: 30000,
    staleTime: 15000,
  });

  const { data: evoData, isLoading: evoLoading } = useQuery({
    queryKey: ['evo', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/evo/status');
      if (!response.ok) throw new Error('Failed to fetch evo status');
      return response.json();
    },
    refetchInterval: 30000,
    staleTime: 15000,
  });

  const isLoading = brainLoading || evoLoading;

  return (
    <div className="border rounded-lg p-4 bg-card">
      <div className="flex items-center gap-2 mb-3">
        <Brain className="w-4 h-4 text-indigo-600" />
        <h3 className="font-semibold">Live & R&D</h3>
      </div>
      {isLoading ? (
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-muted rounded"></div>
          <div className="h-4 bg-muted rounded"></div>
        </div>
      ) : (
        <div className="space-y-2 text-sm">
          {brainData && typeof brainData.recent_pf_after_costs === 'number' && typeof brainData.sharpe_30d === 'number' && (
            <div>
              <div className="font-medium text-xs text-muted-foreground mb-1">BRAIN (Live)</div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">PF:</span>
                <span className={brainData.recent_pf_after_costs >= 1 ? 'text-green-600' : 'text-red-600'}>
                  {brainData.recent_pf_after_costs.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sharpe:</span>
                <span className={brainData.sharpe_30d >= 0.5 ? 'text-green-600' : 'text-yellow-600'}>
                  {brainData.sharpe_30d.toFixed(2)}
                </span>
              </div>
            </div>
          )}
          {evoData && (
            <div className="border-t pt-2">
              <div className="font-medium text-xs text-muted-foreground mb-1">EVO (R&D)</div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Gen:</span>
                <span>{evoData.generation}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Best PF:</span>
                <span className={evoData.best?.metrics.pf_after_costs >= 1 ? 'text-green-600' : 'text-red-600'}>
                  {evoData.best?.metrics.pf_after_costs.toFixed(3)}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
      <Link to="/decisions?tab=evo" className="text-xs text-primary hover:text-primary/80 mt-2 inline-block">
        View candidates →
      </Link>
    </div>
  );
};

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

        {/* Main Dashboard Content - 8 Lean Cards */}
        <ErrorBoundary>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* 1. Health Card */}
            <HealthCard />

            {/* 2. Autopilot Card */}
            <AutopilotCard />

            {/* 3. Portfolio Card */}
            <PortfolioCard />

            {/* 4. Strategy Spotlight Card */}
            <StrategySpotlightCard />

            {/* 5. Pipeline Health Card */}
            <PipelineHealthCard />

            {/* 6. Decisions Summary Card */}
            <DecisionsSummaryCard />

            {/* 7. Orders Snapshot Card */}
            <OrdersSnapshotCard />

            {/* 8. Live & R&D Card */}
            <LiveRnDCard />
          </div>
        </ErrorBoundary>

        {/* Tables: Positions and Trades */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PositionsTable />
          <TradesTable />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
