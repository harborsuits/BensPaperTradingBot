import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  Activity,
  Target,
  BarChart3,
  Zap,
  PlayCircle,
  CheckCircle,
  Brain
} from 'lucide-react';

import { portfolioApi, ingestionApi } from '@/services/api';
import { qk } from '@/services/qk';
import { useWebSocketChannel, useWebSocketMessage } from '@/services/websocket';
import { TradeCandidate, DataStatusSummary, IngestionMetricsModel, SafetyStatus } from '@/types/api.types';
import { DATA_REFRESH_CONFIG } from '@/config/dataRefreshConfig';
import { ErrorBoundary } from '@/components/util/ErrorBoundary';
import PortfolioCard from '@/components/dashboard/PortfolioCard';
import { DataSyncDashboard } from '@/components/DataSyncDashboard';
import { useSyncedHealth, useSyncedAutoloop, useSyncedStrategies, useSyncedPipelineHealth, useSyncedDecisionsSummary, useSyncedOrders, useSyncedBrainStatus, useSyncedEvoStatus } from '@/hooks/useSyncedData';
import AutoRunnerStrip from '@/components/trading/AutoRunnerStrip';
import ActivityTicker from '@/components/trading/ActivityTicker';
import PriceTape from '@/components/cards/PriceTape';
import PositionsTable from '@/components/PositionsTable';
import TradesTable from '@/components/TradesTable';
import RealtimeBanner from '@/components/Banners/RealtimeBanner';


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
              const rttMs = (typeof healthData.broker?.rttMs === 'number' ? healthData.broker.rttMs : undefined) ?? 
                           (typeof healthData.broker_age_s === 'number' ? Math.round(healthData.broker_age_s * 1000) : undefined);
              // Cap display at 999ms for readability
              if (typeof rttMs === 'number') {
                return <span className={rttMs > 500 ? 'text-yellow-600' : ''}>{rttMs > 999 ? '>999ms' : `${rttMs}ms`}</span>;
              }
              return <span>—</span>;
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
            <span className={`font-medium ${autopilotData.status === 'IDLE' ? 'text-green-600' : 'text-yellow-600'}`}>
              {autopilotData.status}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Running:</span>
            <span className={(autopilotData.is_running || autopilotData.isRunning) ? 'text-green-600' : 'text-red-600'}>
              {(autopilotData.is_running || autopilotData.isRunning) ? 'Yes' : 'No'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Interval:</span>
            <span>{(autopilotData.interval_ms || autopilotData.interval) ? `${(autopilotData.interval_ms || autopilotData.interval)/1000}s` : '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Last run:</span>
            <span className="text-xs">
              {(autopilotData.last_cycle || autopilotData.lastRun) ? new Date(autopilotData.last_cycle || autopilotData.lastRun).toLocaleTimeString() : '—'}
            </span>
          </div>
        </div>
      ) : (
        <div className="text-muted-foreground text-sm">No autopilot data</div>
      )}
    </div>
  );
};

const StrategySpotlightCard: React.FC<{
  highlightedStrategy?: string | null;
  strategyDecisions?: Record<string, number>;
  realtimeActivity?: {symbol: string, strategy: string, action: string}[];
}> = ({ highlightedStrategy, strategyDecisions = {}, realtimeActivity = [] }) => {
  const { data: strategiesData, isLoading, error } = useSyncedStrategies();
  
  // Handle both array and object with items property
  const strategies = Array.isArray(strategiesData) ? strategiesData : strategiesData?.items || [];
  
  // Find News Momentum v2 strategy or use the top strategy
  const newsMomentumStrategy = strategies?.find((s: any) => s.id === 'news_momo_v2' || s.name?.includes('News Momentum'));
  const topStrategy = newsMomentumStrategy || strategies?.[0];
  
  // Check if this strategy is currently active
  const isHighlighted = topStrategy && highlightedStrategy === topStrategy.id;
  const recentDecisions = strategyDecisions[topStrategy?.id] || 0;
  const recentActivity = realtimeActivity.filter(a => a.strategy === topStrategy?.id);

  return (
    <div className={`border rounded-lg p-4 bg-card transition-all duration-300 ${
      isHighlighted ? 'ring-2 ring-purple-500 shadow-lg transform scale-105' : ''
    }`}>
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 className={`w-4 h-4 ${isHighlighted ? 'text-purple-500' : 'text-purple-600'}`} />
        <h3 className="font-semibold">Strategy Spotlight</h3>
        {isHighlighted && (
          <div className="ml-auto">
            <div className="w-3 h-3 rounded-full bg-purple-500 animate-pulse" />
          </div>
        )}
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
          <div className="font-medium flex items-center gap-2">
            {topStrategy.name || topStrategy.id}
            {topStrategy.id === 'news_momo_v2' && <Zap className="w-3 h-3 text-yellow-500" />}
          </div>
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
            <span className={recentDecisions > 0 ? 'text-purple-600 font-medium' : ''}>
              {topStrategy.performance?.trades_count || 0}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Profit Factor:</span>
            <span className={(topStrategy.performance?.profit_factor || 0) > 1 ? 'text-green-600' : 'text-red-600'}>
              {(topStrategy.performance?.profit_factor || 0).toFixed(2)}
            </span>
          </div>
          
          {/* Show real-time activity if any */}
          {recentActivity.length > 0 && (
            <div className="mt-2 pt-2 border-t border-muted">
              <div className="text-xs text-muted-foreground mb-1">Live Activity</div>
              <div className="space-y-1">
                {recentActivity.slice(0, 2).map((activity, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs bg-purple-50 dark:bg-purple-950/20 p-1 rounded">
                    <span className="font-medium">{activity.symbol}</span>
                    <span className={activity.action === 'BUY' ? 'text-green-600' : 'text-red-600'}>
                      {activity.action}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
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
            <span className="text-xs">15m</span>
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
      ) : decisionsData ? (
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Proposals/min:</span>
            <span className="font-medium">{
              decisionsData.proposalsPerMinute?.toFixed(1) || 
              decisionsData.proposals_per_min?.toFixed(1) || '0.0'
            }</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Unique Symbols:</span>
            <span>{decisionsData.uniqueSymbols || decisionsData.unique_symbols || 0}</span>
          </div>
          {decisionsData.byStrategy ? (
            <>
              <div className="flex justify-between">
                <span className="text-muted-foreground">News Momentum:</span>
                <span className="text-green-600">{decisionsData.byStrategy.news_momo_v2?.proposals || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mean Rev:</span>
                <span className="text-blue-600">{decisionsData.byStrategy.mean_rev_1?.proposals || 0}</span>
              </div>
            </>
          ) : decisionsData.by_stage ? (
            <>
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
            </>
          ) : null}
          {decisionsData.topSymbols && decisionsData.topSymbols.length > 0 && (
            <div className="mt-2 pt-2 border-t">
              <div className="text-xs text-muted-foreground">Top Symbols:</div>
              <div className="flex gap-2 mt-1">
                {decisionsData.topSymbols.slice(0, 3).map((item: any) => (
                  <span key={item.symbol} className="text-xs bg-muted px-1.5 py-0.5 rounded">
                    {item.symbol}
                  </span>
                ))}
              </div>
            </div>
          )}
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
                <span className="text-xs">{brainData.symbolsTracked || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Decisions:</span>
                <span className="text-xs">{brainData.decisionsToday || 0} today</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Latency:</span>
                <span className={brainData.health?.scoring?.latency_ms < 50 ? 'text-green-600' : 'text-yellow-600'}>
                  {brainData.health?.scoring?.latency_ms || 0}ms
                </span>
              </div>
            </div>
          )}
          {evoData && (
            <div className="border-t pt-2">
              <div className="font-medium text-xs text-muted-foreground mb-1">EVO (R&D)</div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Generations:</span>
                <span>{evoData.totalGenerations || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Best Fitness:</span>
                <span className={evoData.bestFitness >= 0.7 ? 'text-green-600' : 'text-yellow-600'}>
                  {(evoData.bestFitness || 0).toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Experiments:</span>
                <span className="text-xs">{evoData.activeExperiments || 0} / {evoData.resourceUsage?.maxExperiments || 5}</span>
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
  const [, setDataStatus] = useState<DataStatusSummary | null>(null);
  const [, setSafetyStatus] = useState<SafetyStatus | null>(null);
  
  // NEW: Track active strategy interactions
  const [highlightedStrategy, setHighlightedStrategy] = useState<string | null>(null);
  const [strategyDecisions, setStrategyDecisions] = useState<Record<string, number>>({});
  const [realtimeActivity, setRealtimeActivity] = useState<{symbol: string, strategy: string, action: string}[]>([]);
  
  // Connect to WebSocket channels
  useWebSocketChannel('context', true);
  useWebSocketChannel('strategies', true);
  useWebSocketChannel('trading', true);
  useWebSocketChannel('portfolio', true);
  useWebSocketChannel('logs', true);
  useWebSocketChannel('alerts', true);
  useWebSocketChannel('data', true);
  
  // Context data is managed by DataSyncDashboard
  
  useQuery({
    queryKey: ['dataStatus'],
    queryFn: () => ingestionApi.getDataStatus(),
    onSuccess: (response) => {
      if (response.success && response.data) setDataStatus(response.data);
    },
    refetchInterval: DATA_REFRESH_CONFIG.metrics?.refetchInterval || 60000,
    staleTime: DATA_REFRESH_CONFIG.metrics?.staleTime || 45000,
  });

  // Strategies are fetched via useSyncedStrategies hook instead
  
  // Decisions are fetched via useSyncedDecisions hook instead
  
  useQuery({
    queryKey: qk.portfolio('paper'),
    queryFn: () => portfolioApi.getPortfolio('paper'),
    refetchInterval: DATA_REFRESH_CONFIG.paperAccount?.refetchInterval || 15000,
    staleTime: DATA_REFRESH_CONFIG.paperAccount?.staleTime || 10000,
  });
  
  // Live portfolio disabled for now; focusing on paper trading only

  // Portfolio data is normalized in the synced hooks
  
  // Alerts are handled by DataSyncDashboard component
  
  // WebSocket message handlers
  useWebSocketMessage<TradeCandidate[]>(
    'decision_update',
    (message) => {
      const arr = Array.isArray(message?.data) ? (message.data as TradeCandidate[]) : [];
      
      // Track real-time activity for integration
      if (arr.length > 0) {
        arr.forEach(decision => {
          if (decision.strategy_id) {
            setStrategyDecisions(prev => ({
              ...prev,
              [decision.strategy_id]: (prev[decision.strategy_id] || 0) + 1
            }));
          }
        });
        
        // Add latest to realtime activity
        const latest = arr[0];
        if (latest) {
          setRealtimeActivity(prev => [{
            symbol: latest.symbol,
            strategy: latest.strategy_id || 'unknown',
            action: latest.direction?.toUpperCase() || 'UNKNOWN'
          }, ...prev].slice(0, 10));
          
          // Highlight active strategy briefly
          if (latest.strategy_id) {
            setHighlightedStrategy(latest.strategy_id);
            setTimeout(() => setHighlightedStrategy(null), 3000);
          }
        }
      }
    },
    []
  );
  
  // Log and alert handling is managed by DataSyncDashboard

  useWebSocketMessage<DataStatusSummary>(
    'ingestion_sources_status',
    (message) => { setDataStatus(message.data); },
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

            {/* 4. Strategy Spotlight Card - Enhanced with real-time */}
            <StrategySpotlightCard 
              highlightedStrategy={highlightedStrategy}
              strategyDecisions={strategyDecisions}
              realtimeActivity={realtimeActivity}
            />

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
        
        {/* Data Synchronization Status */}
        <div className="mt-8">
          <DataSyncDashboard />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
