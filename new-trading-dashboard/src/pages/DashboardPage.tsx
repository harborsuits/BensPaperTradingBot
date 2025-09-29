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
import { DATA_REFRESH_CONFIG } from '@/config/dataRefreshConfig';
import { ErrorBoundary } from '@/components/util/ErrorBoundary';
import { SimpleCard } from '@/components/ui/SimpleCard';
import PortfolioCard from '@/components/dashboard/PortfolioCard';
import BrainFlowNowCard from '@/components/dashboard/BrainFlowNowCard';
import TickerHighlightsCard from '@/components/dashboard/TickerHighlightsCard';
import { DataSyncDashboard } from '@/components/DataSyncDashboard';
import { useSyncedHealth, useSyncedAutoloop, useSyncedStrategies, useSyncedPipelineHealth, useSyncedDecisionsSummary, useSyncedOrders, useSyncedBrainStatus, useSyncedEvoStatus } from '@/hooks/useSyncedData';
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
import OptionsSummaryCard from '@/components/options/OptionsSummaryCard';

// Helpers to make rendering resilient to undefined data
const asArray = <T,>(v: T[] | undefined | null): T[] => (Array.isArray(v) ? v : []);
const numberOr = (v: unknown, d = 0): number => (typeof v === 'number' && !Number.isNaN(v) ? v : d);

// Dashboard Card Components

const HealthCard: React.FC = () => {
  const { data: healthData, isLoading, error } = useSyncedHealth();

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
            <span className="text-muted-foreground">WebSocket:</span>
            <span className={healthData?.wsConnected ? 'text-green-600' : 'text-red-600'}>
              {healthData?.wsConnected ? 'Connected' : 'Disconnected'}
            </span>
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
  const { data: autopilotData, isLoading, error } = useSyncedAutoloop();

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
            <span className="text-muted-foreground">Status:</span>
            <span className={`font-medium ${
              autopilotData.status === 'IDLE' || autopilotData.status === 'IDLE_NO_SIGNALS' 
                ? 'text-yellow-600' 
                : 'text-green-600'
            }`}>
              {autopilotData.status || 'UNKNOWN'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Running:</span>
            <span className={(autopilotData.is_running || autopilotData.isRunning || autopilotData.enabled) ? 'text-green-600' : 'text-red-600'}>
              {(autopilotData.is_running || autopilotData.isRunning || autopilotData.enabled) ? 'Yes' : 'No'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Interval:</span>
            <span>
              {autopilotData.interval_ms ? `${autopilotData.interval_ms/1000}s` : 
               autopilotData.interval ? `${autopilotData.interval/1000}s` : '—'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Last run:</span>
            <span className="text-xs">
              {(autopilotData.last_cycle || autopilotData.lastRun) 
                ? new Date(autopilotData.last_cycle || autopilotData.lastRun).toLocaleTimeString() 
                : '—'}
            </span>
          </div>
        </div>
      ) : (
        <div className="text-muted-foreground text-sm">No autopilot data</div>
      )}
    </div>
  );
};

const StrategySpotlightCard: React.FC = () => {
  const { data: strategiesData, isLoading, error } = useSyncedStrategies();
  
  // Handle both array and object with items property
  const strategies = Array.isArray(strategiesData) ? strategiesData : strategiesData?.items || [];
  
  // Find the best performing active strategy by profit factor
  const topStrategy = strategies
    .filter((s: any) => s.active)
    .sort((a: any, b: any) => {
      const aProfitFactor = a.performance?.profit_factor || 0;
      const bProfitFactor = b.performance?.profit_factor || 0;
      return bProfitFactor - aProfitFactor;
    })[0] || strategies[0]; // Fallback to first strategy if none are active

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
          <div className="font-medium">{topStrategy.name || topStrategy.id}</div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Status:</span>
            <span className={topStrategy.active ? 'text-green-600' : 'text-red-600'}>
              {topStrategy.active ? 'Active' : 'Inactive'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Sharpe:</span>
            <span className={(topStrategy.performance?.sharpe_ratio || 0) > 0.5 ? 'text-green-600' : 'text-yellow-600'}>
              {(topStrategy.performance?.sharpe_ratio || 0).toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Trades:</span>
            <span>{topStrategy.performance?.trades_count || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Profit Factor:</span>
            <span className={(topStrategy.performance?.profit_factor || 0) > 1 ? 'text-green-600' : 'text-red-600'}>
              {(topStrategy.performance?.profit_factor || 0).toFixed(2)}
            </span>
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
  const { data: pipelineData, isLoading, error } = useSyncedPipelineHealth('15m');

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
      ) : pipelineData ? (
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Total Scores:</span>
            <span className="font-medium">{pipelineData.total_scores || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Avg Score:</span>
            <span className={pipelineData.avg_score > 0.5 ? 'text-green-600' : 'text-yellow-600'}>
              {((pipelineData.avg_score || 0) * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">High Conf:</span>
            <span className="text-blue-600">{pipelineData.high_confidence || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Window:</span>
            <span className="text-xs">{pipelineData.window || '15m'}</span>
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
  const { data: decisionsData, isLoading, error } = useSyncedDecisionsSummary('15m');

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
            <span className="text-blue-600">{decisionsData.by_stage.proposed || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Intent:</span>
            <span className="text-yellow-600">{decisionsData.by_stage.intent || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Executed:</span>
            <span className="text-green-600">{decisionsData.by_stage.executed || 0}</span>
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
  const { data: ordersData, isLoading, error } = useSyncedOrders();

  // Ensure ordersData is an array
  const orders = Array.isArray(ordersData) ? ordersData : [];
  const openOrders = orders.filter((order: any) => order.status === 'pending' || order.status === 'open');

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
      ) : (
        <div className="space-y-2">
          <div className="text-sm font-medium">
            Open Orders: {openOrders.length}
          </div>
          {openOrders.slice(0, 3).map((order: any, idx: number) => (
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
          {openOrders.length === 0 && (
            <div className="text-muted-foreground text-sm text-center py-4">
              No open orders
            </div>
          )}
        </div>
      )}
      <Link to="/decisions?tab=executions" className="text-xs text-primary hover:text-primary/80 mt-2 inline-block">
        See all orders →
      </Link>
    </div>
  );
};

const LiveRnDCard: React.FC = () => {
  const { data: brainData, isLoading: brainLoading } = useSyncedBrainStatus();
  const { data: evoData, isLoading: evoLoading } = useSyncedEvoStatus();

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
          {brainData && (
            <div>
              <div className="font-medium text-xs text-muted-foreground mb-1">BRAIN (Live)</div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Status:</span>
                <span className={brainData.status === 'active' ? 'text-green-600' : 'text-yellow-600'}>
                  {brainData.status || 'unknown'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Symbols:</span>
                <span className="text-xs">{brainData.current_symbols?.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Python:</span>
                <span className={brainData.python_brain_config?.configured ? 'text-green-600' : 'text-red-600'}>
                  {brainData.python_brain_config?.configured ? 'Connected' : 'Disconnected'}
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
  
  // Initial data fetching with centralized refresh config
  useQuery({
    queryKey: qk.context,
    queryFn: () => contextApi.getMarketContext(),
    refetchInterval: DATA_REFRESH_CONFIG.context.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.context.staleTime,
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
    refetchInterval: DATA_REFRESH_CONFIG.metrics.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.metrics.staleTime,
  });

  useQuery({
    queryKey: qk.strategies,
    queryFn: () => strategyApi.getActiveStrategies(),
    refetchInterval: DATA_REFRESH_CONFIG.strategies.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.strategies.staleTime,
    onSuccess: (response) => {
      if (response.success && response.data) {
        setActiveStrategies(response.data);
      }
    },
  });
  
  useQuery({
    queryKey: qk.decisions(50),
    queryFn: () => decisionApi.getLatestDecisions(),
    refetchInterval: DATA_REFRESH_CONFIG.decisions.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.decisions.staleTime,
    onSuccess: (response) => {
      if (response.success && response.data) {
        setRecentDecisions(response.data);
      }
    },
  });
  
  const { data: portfolioData } = useQuery({
    queryKey: qk.portfolio('paper'),
    queryFn: () => portfolioApi.getPortfolio('paper'),
    refetchInterval: DATA_REFRESH_CONFIG.paperAccount.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.paperAccount.staleTime,
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

            {/* 8. Options Trading Card */}
            {/* <OptionsSummaryCard /> */}
          </div>
        </ErrorBoundary>

        {/* Tables: Positions and Trades */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PositionsTable />
          <TradesTable />
        </div>
        
        {/* Data Synchronization Status */}
        <div className="mt-8">
          <DataSyncDashboard />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
