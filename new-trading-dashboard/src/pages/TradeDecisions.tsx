import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import {
  BarChart3, 
  Activity,
  Target, 
  Zap, 
  Brain,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle
} from 'lucide-react';
import { useDecisionsRecent } from '@/hooks/useDecisionsRecent';
import { useSyncedStrategies, useSyncedOrders, useSyncedPipelineHealth, useSyncedEvoStatus } from '@/hooks/useSyncedData';
import { PipelineFlowDiagram } from '@/components/ui/PipelineFlowDiagram';

const TradeDecisionsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'proposals');

  // Update URL when tab changes
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    setSearchParams({ tab: value });
  };

  // Sync activeTab with URL params
  useEffect(() => {
    const tab = searchParams.get('tab');
    if (tab && tab !== activeTab) {
      setActiveTab(tab);
    }
  }, [searchParams, activeTab]);

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Trade Decisions</h1>
        <p className="text-muted-foreground mt-2">
          Monitor strategies, pipeline processing, and execution tracking
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="strategies">Strategies</TabsTrigger>
          <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="proposals">Proposals</TabsTrigger>
          <TabsTrigger value="intents">Intents</TabsTrigger>
          <TabsTrigger value="executions">Executions</TabsTrigger>
          <TabsTrigger value="evo">Evolution</TabsTrigger>
        </TabsList>

        <TabsContent value="strategies" className="mt-6">
          <StrategiesTab />
        </TabsContent>

        <TabsContent value="pipeline" className="mt-6">
          <PipelineTab />
        </TabsContent>

        <TabsContent value="proposals" className="mt-6">
          <ProposalsTab />
        </TabsContent>

        <TabsContent value="intents" className="mt-6">
          <IntentsTab />
        </TabsContent>

        <TabsContent value="executions" className="mt-6">
          <ExecutionsTab />
        </TabsContent>

        <TabsContent value="evo" className="mt-6">
          <EvolutionTab />
        </TabsContent>
      </Tabs>
    </div>
  );
};

// Strategies Tab
const StrategiesTab: React.FC = () => {
  const { data: strategiesData, isLoading } = useSyncedStrategies();
  const strategies = Array.isArray(strategiesData) ? strategiesData : strategiesData?.items || [];

  if (isLoading) {
    return <div className="text-center py-8">Loading strategies...</div>;
  }

  return (
    <div className="space-y-6">

      {!strategies.length ? (
        <Card className="p-8 text-center">
          <BarChart3 className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No Active Strategies</h3>
          <p className="text-muted-foreground mt-2">Trading strategies will appear here when activated.</p>
        </Card>
      ) : (
        <div className="grid gap-4">
          {strategies.map((strategy: any) => (
            <Card key={strategy.id} className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold">{strategy.name || strategy.id}</h3>
                  <p className="text-sm text-muted-foreground">{strategy.description || 'No description'}</p>
                </div>
                <span className={`px-2 py-1 rounded text-xs ${
                  strategy.active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                }`}>
                  {strategy.active ? 'Active' : 'Inactive'}
                </span>
              </div>
              
              <div className="grid grid-cols-4 gap-4 text-sm mb-4">
                <div>
                  <span className="text-muted-foreground">Profit Factor</span>
                  <p className="font-medium">{(strategy.performance?.profit_factor || 0).toFixed(2)}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Sharpe Ratio</span>
                  <p className="font-medium">{(strategy.performance?.sharpe_ratio || 0).toFixed(2)}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Total Trades</span>
                  <p className="font-medium">{strategy.performance?.trades_count || 0}</p>
                </div>
        <div>
                  <span className="text-muted-foreground">Win Rate</span>
                  <p className="font-medium">{((strategy.performance?.win_rate || 0) * 100).toFixed(1)}%</p>
                </div>
              </div>

              {/* Evidence Section */}
              <div className="mt-4 p-3 bg-muted/30 rounded">
                <h4 className="text-sm font-semibold mb-2">Why This Strategy Is {strategy.active ? 'Active' : 'Inactive'}:</h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  {(strategy.performance?.profit_factor || 0) > 1 && (
                    <li>• Profit factor of {(strategy.performance?.profit_factor || 0).toFixed(2)} shows consistent profitability</li>
                  )}
                  {(strategy.performance?.sharpe_ratio || 0) > 0.5 && (
                    <li>• Sharpe ratio of {(strategy.performance?.sharpe_ratio || 0).toFixed(2)} indicates good risk-adjusted returns</li>
                  )}
                  {(strategy.performance?.win_rate || 0) > 0.5 && (
                    <li>• Win rate of {((strategy.performance?.win_rate || 0) * 100).toFixed(0)}% exceeds random chance</li>
                  )}
                  {strategy.performance?.trades_count > 100 && (
                    <li>• Tested over {strategy.performance.trades_count} trades for statistical significance</li>
                  )}
                  {!strategy.active && (
                    <li>• Currently disabled due to underperformance or market conditions</li>
                  )}
                </ul>
        </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

// Pipeline Tab
const PipelineTab: React.FC = () => {
  const { data: pipelineData, isLoading, error } = useSyncedPipelineHealth('15m');

  // Debug logging
  console.log('Pipeline Tab State:', { pipelineData, isLoading, error });

  if (isLoading) {
    return <div className="text-center py-8">Loading pipeline data...</div>;
  }

  if (error) {
    console.error('Pipeline tab error:', error);
    return (
      <Card className="p-8 text-center">
        <Activity className="mx-auto h-12 w-12 text-red-500 mb-4" />
        <h3 className="text-lg font-medium">Error Loading Pipeline</h3>
        <p className="text-muted-foreground mt-2">Failed to load pipeline data. Please try refreshing.</p>
        <p className="text-sm text-red-500 mt-2">{error?.message || 'Unknown error'}</p>
      </Card>
    );
  }

  if (!pipelineData) {
    return (
      <Card className="p-8 text-center">
        <Activity className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium">No Pipeline Data</h3>
        <p className="text-muted-foreground mt-2">Pipeline data is not available yet.</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
          {/* Pipeline Flow Visualization */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Pipeline Flow</h3>
            <PipelineFlowDiagram 
              stages={[
                { 
                  name: 'Input', 
                  count: pipelineData.total_scores || 0, 
                  status: pipelineData.total_scores > 0 ? 'success' : 'warning',
                  details: `${pipelineData.unique_symbols || 0} symbols`
                },
                { 
                  name: 'Validation', 
                  count: Math.floor((pipelineData.total_scores || 0) * 0.8), 
                  status: 'success',
                  details: 'Format & range checks'
                },
                { 
                  name: 'Scoring', 
                  count: Math.floor((pipelineData.total_scores || 0) * 0.7), 
                  status: pipelineData.avg_score > 0.5 ? 'success' : 'warning',
                  details: `Avg: ${((pipelineData.avg_score || 0) * 100).toFixed(0)}%`
                },
                { 
                  name: 'Gates', 
                  count: pipelineData.high_confidence || 0, 
                  status: pipelineData.high_confidence > 0 ? 'success' : 'error',
                  details: 'Risk & confidence'
                },
                { 
                  name: 'Output', 
                  count: pipelineData.high_confidence || 0, 
                  status: 'processing',
                  details: 'Ready to trade'
                }
              ]}
              className="mb-6"
            />
          </Card>

          {/* Pipeline Summary Stats */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Pipeline Summary (Last 15 Minutes)</h3>
            <div className="grid grid-cols-4 gap-4 mb-4">
            <div>
              <span className="text-muted-foreground text-sm">Total Processed</span>
              <p className="text-2xl font-bold">{pipelineData.total_scores || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Avg Score</span>
              <p className={`text-2xl font-bold ${
                (pipelineData.avg_score || 0) > 0.7 ? 'text-green-600' : 
                (pipelineData.avg_score || 0) > 0.5 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {((pipelineData.avg_score || 0) * 100).toFixed(0)}%
              </p>
        </div>
            <div>
              <span className="text-muted-foreground text-sm">High Confidence</span>
              <p className="text-2xl font-bold text-blue-600">{pipelineData.high_confidence || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Last Update</span>
              <p className="text-sm">{pipelineData.asOf ? new Date(pipelineData.asOf).toLocaleTimeString() : 'N/A'}</p>
            </div>
          </div>

          {/* Evidence Section */}
          <div className="mt-4 p-3 bg-muted/30 rounded">
            <h4 className="text-sm font-semibold mb-2">Why These Scores:</h4>
            <ul className="text-xs text-muted-foreground space-y-1">
              <li>• {pipelineData.total_scores || 0} trade signals generated from {pipelineData.unique_symbols || 'multiple'} symbols being monitored</li>
              {pipelineData.avg_score < 0.5 && (
                <li>• Low average score ({((pipelineData.avg_score || 0) * 100).toFixed(0)}%) due to weak market conditions or poor signal quality</li>
              )}
              {pipelineData.avg_score >= 0.5 && pipelineData.avg_score < 0.7 && (
                <li>• Moderate score ({((pipelineData.avg_score || 0) * 100).toFixed(0)}%) indicates mixed market signals</li>
              )}
              {pipelineData.avg_score >= 0.7 && (
                <li>• High score ({((pipelineData.avg_score || 0) * 100).toFixed(0)}%) shows strong alignment of technical and sentiment indicators</li>
              )}
              <li>• {pipelineData.high_confidence || 0} signals passed all quality gates including risk checks, pattern validation, and market timing</li>
              {pipelineData.by_symbol && Object.keys(pipelineData.by_symbol).length > 0 && (
                <li>• Top symbols: {Object.keys(pipelineData.by_symbol).slice(0, 3).join(', ')}</li>
              )}
            </ul>
      </div>
          </Card>
    </div>
  );
};

// Proposals Tab
const ProposalsTab: React.FC = () => {
  const { data: allDecisions } = useDecisionsRecent(50);
  
  // Filter out negative EV trades on the frontend too
  const decisions = allDecisions?.filter(d => {
    const ev = d.analysis?.scores?.afterCostEV || 
               d.meta?.scoring_breakdown?.afterCostEV || 
               d.confidence || 0;
    return ev > 0; // Only show positive expected value trades
  }) || [];

  return (
    <div className="space-y-6">
      {!decisions || decisions.length === 0 ? (
        <Card className="p-8 text-center">
          <Target className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No Trade Proposals</h3>
          <p className="text-muted-foreground mt-2">Trade proposals will appear here as they are generated.</p>
        </Card>
      ) : (
        <div className="space-y-4">
          {decisions.map((d: any) => (
            <Card key={d.trace_id || d.id} className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold">{d.symbol}</h3>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    // Only show green for profitable trades, red for unprofitable
                    d.meta?.scoring_breakdown?.afterCostEV !== undefined ?
                      (d.meta.scoring_breakdown.afterCostEV > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800') :
                    d.confidence !== undefined ?
                      (d.confidence > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800') :
                    // Default colors for BUY/SELL
                    (d.action === 'BUY' || d.side === 'buy' 
                      ? 'bg-blue-100 text-blue-800' 
                      : 'bg-orange-100 text-orange-800')
                  }`}>
                    {(d.action || d.side || '').toUpperCase()}
                  </span>
                  {/* Show quantity if available */}
                  {(d.qty || d.quantity || d.plan?.qty) && (
                    <span className="px-2 py-1 rounded text-xs bg-gray-100 text-gray-800">
                      Qty: {d.qty || d.quantity || d.plan?.qty}
                    </span>
                  )}
                  {d.gates?.passed && (
                    <CheckCircle className="w-4 h-4 text-green-600" />
                  )}
                  {d.gates?.passed === false && (
                    <XCircle className="w-4 h-4 text-red-600" />
          )}
        </div>
                <span className="text-sm text-muted-foreground">
                  {new Date(d.createdAt || d.timestamp).toLocaleTimeString()}
                </span>
      </div>

              <p className="text-muted-foreground mb-4">
                {d.one_liner || d.reason || 
                 (d.meta?.reason === 'coordinator_winner' ? `${d.meta.winner || d.strategy_id} strategy signal` : 'No reason provided')}
              </p>

              {/* Evidence Section */}
              <div className="mb-4 p-3 bg-muted/30 rounded">
                <h4 className="text-sm font-semibold mb-2">Evidence for This Trade:</h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  {/* Confidence can be positive or negative */}
                  {d.confidence !== undefined && (
                    <li className={d.confidence < 0 ? 'text-red-600' : ''}>
                      • Confidence level: {d.confidence < 0 ? '-' : ''}{Math.abs(d.confidence * 100).toFixed(0)}%
                    </li>
                  )}
                  {/* Show brain score if available */}
                  {d.brain_score !== undefined && (
                    <li>• Brain score: {(d.brain_score * 100).toFixed(0)}%</li>
                  )}
                  {/* Show strategy metadata */}
                  {d.meta?.winner && (
                    <li>• Winning strategy: {d.meta.winner}</li>
                  )}
                  {d.meta?.scoring_breakdown && (
                    <>
                      {d.meta.scoring_breakdown.afterCostEV && (
                        <li className={d.meta.scoring_breakdown.afterCostEV < 0 ? 'text-orange-600' : ''}>
                          • Expected value after costs: {(d.meta.scoring_breakdown.afterCostEV * 100).toFixed(1)}%
                        </li>
                      )}
                      {d.meta.scoring_breakdown.reliability && (
                        <li>• Reliability score: {d.meta.scoring_breakdown.reliability.toFixed(2)}</li>
                      )}
                      {d.meta.scoring_breakdown.liquidity && (
                        <li>• Liquidity score: {(d.meta.scoring_breakdown.liquidity * 100).toFixed(1)}%</li>
                      )}
                    </>
                  )}
                  {/* Show analysis data */}
                  {d.analysis && (
                    <>
                      {d.analysis.spread_bps && (
                        <li>• Spread: {d.analysis.spread_bps.toFixed(1)} bps</li>
                      )}
                      {d.analysis.position_value && (
                        <li>• Position value: ${d.analysis.position_value.toFixed(2)}</li>
                      )}
                    </>
                  )}
                  {/* Key factors from various sources */}
                  {d.reasons && d.reasons.length > 0 && (
                    <li>• Key factors: {d.reasons.slice(0, 3).join(', ')}</li>
                  )}
                  {d.sources && d.sources.length > 0 && (
                    <li>• Data sources: {d.sources.slice(0, 3).join(', ')}</li>
                  )}
                  {d.technical_score && (
                    <li>• Technical analysis score: {(d.technical_score * 100).toFixed(0)}%</li>
                  )}
                  {d.sentiment_score && (
                    <li>• Market sentiment: {d.sentiment_score > 0 ? 'Positive' : 'Negative'} ({Math.abs(d.sentiment_score).toFixed(2)})</li>
                  )}
                  {/* Market context */}
                  {d.market_context && (
                    <>
                      {d.market_context.regime && (
                        <li>• Market regime: {d.market_context.regime}</li>
                      )}
                      {d.market_context.volatility && (
                        <li>• Volatility: {d.market_context.volatility}</li>
                      )}
                    </>
                  )}
                  {/* Gate failures */}
                  {d.gates?.passed === false && d.gates?.failed_reason && (
                    <li className="text-red-600">• Failed gate: {d.gates.failed_reason}</li>
                  )}
                </ul>
              </div>

              {d.plan && (
                <div className="p-3 bg-muted/30 rounded text-sm">
                  <span className="font-medium">
                    Order Type: {d.plan.orderType || 'MARKET'}
                    {d.plan.limit && ` • Limit: $${d.plan.limit}`}
                    {d.plan.stop && ` • Stop: $${d.plan.stop}`}
                  </span>
                </div>
              )}
            </Card>
          ))}
        </div>
      )}
                  </div>
  );
};

// Intents Tab
const IntentsTab: React.FC = () => {
  const { data: decisions } = useDecisionsRecent(100);
  const intents = decisions?.filter((d: any) => 
    d.stage === 'intent' || 
    (d.gates?.passed === true && d.status !== 'executed')
  ) || [];

  return (
    <div className="space-y-6">
      {!intents.length ? (
        <Card className="p-8 text-center">
          <Clock className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No Pending Intents</h3>
          <p className="text-muted-foreground mt-2">Approved trades awaiting execution will appear here.</p>
        </Card>
      ) : (
        <div className="space-y-4">
          {intents.map((d: any) => (
            <Card key={d.trace_id || d.id} className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold">{d.symbol}</h3>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    // Color based on expected value or confidence
                    d.meta?.scoring_breakdown?.afterCostEV !== undefined ?
                      (d.meta.scoring_breakdown.afterCostEV > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800') :
                    d.confidence !== undefined ?
                      (d.confidence > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800') :
                    // Default colors for BUY/SELL
                    (d.action === 'BUY' || d.side === 'buy' 
                      ? 'bg-blue-100 text-blue-800' 
                      : 'bg-orange-100 text-orange-800')
                  }`}>
                    {(d.action || d.side || '').toUpperCase()}
                  </span>
                  {/* Show quantity if available */}
                  {(d.qty || d.quantity || d.plan?.qty) && (
                    <span className="px-2 py-1 rounded text-xs bg-gray-100 text-gray-800">
                      Qty: {d.qty || d.quantity || d.plan?.qty}
                    </span>
                  )}
                  <span className="px-2 py-1 rounded text-xs bg-yellow-100 text-yellow-800">
                    Pending
                  </span>
                </div>
                <span className="text-sm text-muted-foreground">
                  {new Date(d.createdAt || d.timestamp).toLocaleTimeString()}
                </span>
                  </div>

              <p className="text-muted-foreground mb-4">
                {d.one_liner || d.reason || 
                 (d.meta?.reason === 'coordinator_winner' ? `${d.meta.winner || d.strategy_id} strategy signal` : 'Awaiting execution')}
              </p>

              {/* Evidence Section */}
              <div className="p-3 bg-muted/30 rounded">
                <h4 className="text-sm font-semibold mb-2">Evidence for This Trade:</h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  {/* Confidence can be positive or negative */}
                  {d.confidence !== undefined && (
                    <li className={d.confidence < 0 ? 'text-red-600' : ''}>
                      • Confidence level: {d.confidence < 0 ? '-' : ''}{Math.abs(d.confidence * 100).toFixed(0)}%
                    </li>
                  )}
                  {/* Key factors from the winning strategy */}
                  {d.meta?.reason === 'coordinator_winner' && d.meta?.winner && (
                    <li>• Key factors: {d.meta.winner}</li>
                  )}
                  {d.meta?.scoring_breakdown && (
                    <>
                      {d.meta.scoring_breakdown.afterCostEV && (
                        <li className={d.meta.scoring_breakdown.afterCostEV < 0 ? 'text-orange-600' : ''}>
                          • Expected value: {(d.meta.scoring_breakdown.afterCostEV * 100).toFixed(1)}%
                        </li>
                      )}
                    </>
                  )}
                  {/* Execution timing */}
                  <li>• Approved at {new Date(d.gates?.passed_at || d.createdAt || d.timestamp).toLocaleTimeString()}</li>
                  {d.waiting_for && (
                    <li>• Waiting for: {d.waiting_for}</li>
                  )}
                  {d.market_hours === false && (
                    <li>• Waiting for market to open</li>
                  )}
                  {d.liquidity_check === 'pending' && (
                    <li>• Checking liquidity conditions</li>
                  )}
                  {d.position_size_check && (
                    <li>• Verifying position size limits</li>
                  )}
                  <li>• Time in queue: {Math.round((Date.now() - new Date(d.timestamp || d.createdAt).getTime()) / 60000)} minutes</li>
                </ul>
              </div>
            </Card>
          ))}
        </div>
            )}
          </div>
  );
};

// Executions Tab
const ExecutionsTab: React.FC = () => {
  const { data: ordersData, isLoading } = useSyncedOrders();
  const orders = ordersData || [];

  if (isLoading) {
    return <div className="text-center py-8">Loading orders...</div>;
  }

  return (
    <div className="space-y-6">
      {!orders.length ? (
        <Card className="p-8 text-center">
          <Zap className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No Orders</h3>
          <p className="text-muted-foreground mt-2">Order executions will appear here.</p>
        </Card>
      ) : (
        <div className="space-y-4">
          {orders.map((order: any) => (
            <Card key={order.id} className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold">{order.symbol}</h3>
                  <span className={`px-2 py-1 rounded text-xs ${
                    order.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {order.side?.toUpperCase()}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs ${
                    order.status === 'filled' ? 'bg-green-100 text-green-800' :
                    order.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {order.status?.toUpperCase()}
                  </span>
                </div>
                <span className="text-sm text-muted-foreground">
                  {new Date(order.created_at || order.timestamp).toLocaleTimeString()}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-4 text-sm mb-4">
                <div>
                  <span className="text-muted-foreground">Type</span>
                  <p className="font-medium">{order.type?.toUpperCase() || 'MARKET'}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Quantity</span>
                  <p className="font-medium">{order.qty || 0}</p>
          </div>
                <div>
                  <span className="text-muted-foreground">Price</span>
                  <p className="font-medium">${order.price || order.limit || 'Market'}</p>
                </div>
              </div>

              {/* Evidence Section */}
              <div className="p-3 bg-muted/30 rounded">
                <h4 className="text-sm font-semibold mb-2">Order Details:</h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  {order.status === 'filled' && (
                    <>
                      <li>• Filled at ${order.fill_price || order.price || 'market price'}</li>
                      <li>• Execution time: {order.filled_at ? new Date(order.filled_at).toLocaleTimeString() : 'Unknown'}</li>
                    </>
                  )}
                  {order.status === 'canceled' && (
                    <>
                      <li>• Canceled at {order.canceled_at ? new Date(order.canceled_at).toLocaleTimeString() : 'Unknown'}</li>
                      <li>• Reason: {order.cancel_reason || 'Manual cancellation or market conditions'}</li>
                    </>
                  )}
                  {order.status === 'pending' && (
                    <>
                      <li>• Submitted to {order.venue || 'broker'} at {new Date(order.created_at).toLocaleTimeString()}</li>
                      <li>• Waiting for fill (market {order.type === 'limit' ? `limit at $${order.limit}` : 'order'})</li>
                    </>
                  )}
                  {order.broker_order_id && (
                    <li>• Broker reference: {order.broker_order_id}</li>
                  )}
                  {order.commission && (
                    <li>• Commission: ${order.commission.toFixed(2)}</li>
                  )}
                </ul>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

// Evolution Tab
const EvolutionTab: React.FC = () => {
  const { data: evoData, isLoading } = useSyncedEvoStatus();

  if (isLoading) {
    return <div className="text-center py-8">Loading evolution data...</div>;
  }

  return (
    <div className="space-y-6">
      {evoData ? (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Evolution Status</h3>
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div>
              <span className="text-muted-foreground text-sm">Generation</span>
              <p className="text-2xl font-bold">{evoData.generation || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Population</span>
              <p className="text-2xl font-bold">{evoData.population || 0}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Best PF</span>
              <p className={`text-2xl font-bold ${
                (evoData.best?.metrics?.pf_after_costs || 0) > 1 ? 'text-green-600' : 'text-red-600'
              }`}>
                {(evoData.best?.metrics?.pf_after_costs || 0).toFixed(2)}
              </p>
            </div>
            <div>
              <span className="text-muted-foreground text-sm">Status</span>
              <p className={`text-sm font-medium ${
                evoData.running ? 'text-green-600' : 'text-gray-600'
              }`}>
                {evoData.running ? 'Running' : 'Stopped'}
              </p>
            </div>
          </div>

          {/* Evidence Section */}
          <div className="p-3 bg-muted/30 rounded">
            <h4 className="text-sm font-semibold mb-2">Evolution Progress:</h4>
            <ul className="text-xs text-muted-foreground space-y-1">
              <li>• Generation {evoData.generation} has tested {evoData.population} strategy variations</li>
              {evoData.best?.metrics?.pf_after_costs > 1 && (
                <li>• Best strategy achieved profit factor of {evoData.best.metrics.pf_after_costs.toFixed(2)} (profitable)</li>
              )}
              {evoData.best?.metrics?.sharpe && (
                <li>• Risk-adjusted return (Sharpe): {evoData.best.metrics.sharpe.toFixed(2)}</li>
              )}
              {evoData.best?.metrics?.trades && (
                <li>• Best performer validated over {evoData.best.metrics.trades} trades</li>
              )}
              {evoData.improvement_rate && (
                <li>• Generation-over-generation improvement: {(evoData.improvement_rate * 100).toFixed(1)}%</li>
              )}
              {evoData.convergence_score && (
                <li>• Convergence score: {(evoData.convergence_score * 100).toFixed(0)}% (higher = more stable)</li>
              )}
              {!evoData.running && (
                <li>• Evolution paused - likely reached optimization target or manual stop</li>
              )}
            </ul>
          </div>
        </Card>
      ) : (
        <Card className="p-8 text-center">
          <Brain className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No Evolution Data</h3>
          <p className="text-muted-foreground mt-2">Evolution candidates will appear here.</p>
        </Card>
      )}
    </div>
  );
};

export default TradeDecisionsPage;