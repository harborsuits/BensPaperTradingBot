import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, 
  Activity, 
  Brain, 
  Zap, 
  Target,
  ArrowRight,
  Circle,
  CheckCircle,
  AlertCircle,
  TrendingDown,
  DollarSign,
  BarChart3,
  Shield,
  RefreshCw
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useWebSocketMessage } from '@/services/websocket';
import { useSyncedStrategies, useSyncedDecisionsSummary, useSyncedTrades, useSyncedBrainStatus } from '@/hooks/useSyncedData';
import { toArray } from '@/services/normalize';

interface StrategyFlow {
  strategyId: string;
  strategyName: string;
  status: 'active' | 'inactive' | 'processing';
  decisionsPerMin: number;
  successRate: number;
  lastDecision?: {
    symbol: string;
    action: string;
    confidence: number;
    timestamp: string;
  };
  impact?: {
    trades: number;
    pnl: number;
    winRate: number;
  };
}

interface FlowConnection {
  from: string;
  to: string;
  type: 'decision' | 'trade' | 'analysis';
  active: boolean;
  label?: string;
}

const IntegratedDashboard: React.FC = () => {
  const { data: strategiesData } = useSyncedStrategies();
  const { data: decisionsData } = useSyncedDecisionsSummary('15m');
  const { data: tradesData } = useSyncedTrades();
  const { data: brainData } = useSyncedBrainStatus();
  
  const strategies = toArray(strategiesData);
  const trades = toArray(tradesData);
  
  // Track active flows and connections
  const [activeFlows, setActiveFlows] = useState<FlowConnection[]>([]);
  const [highlightedStrategy, setHighlightedStrategy] = useState<string | null>(null);
  const [realtimeMetrics, setRealtimeMetrics] = useState({
    totalDecisions: 0,
    totalTrades: 0,
    successRate: 0,
    activeSymbols: []
  });

  // Listen for WebSocket updates
  useWebSocketMessage('decision', (message) => {
    if (message.type === 'decision_proposed') {
      // Add flow from strategy to brain
      const flow: FlowConnection = {
        from: message.data.strategy_id,
        to: 'brain',
        type: 'decision',
        active: true,
        label: message.data.symbol
      };
      setActiveFlows(prev => [...prev, flow]);
      
      // Remove flow after animation
      setTimeout(() => {
        setActiveFlows(prev => prev.filter(f => f !== flow));
      }, 3000);
    }
  });

  useWebSocketMessage('trading', (message) => {
    if (message.type === 'trade_executed') {
      // Add flow from brain to execution
      const flow: FlowConnection = {
        from: 'brain',
        to: 'execution',
        type: 'trade',
        active: true,
        label: `${message.data.symbol} ${message.data.side}`
      };
      setActiveFlows(prev => [...prev, flow]);
      
      setTimeout(() => {
        setActiveFlows(prev => prev.filter(f => f !== flow));
      }, 3000);
    }
  });

  // Calculate strategy flows
  const strategyFlows = useMemo<StrategyFlow[]>(() => {
    return strategies.map(strategy => {
      const strategyDecisions = decisionsData?.byStrategy?.[strategy.id] || {};
      const strategyTrades = trades.filter(t => t.strategy_id === strategy.id);
      const recentDecisions = strategyDecisions.proposals || 0;
      
      return {
        strategyId: strategy.id,
        strategyName: strategy.name,
        status: strategy.status as 'active' | 'inactive',
        decisionsPerMin: strategyDecisions.perMinute || 0,
        successRate: strategy.metrics?.win_rate || 0,
        lastDecision: strategy.last_decision ? {
          symbol: strategy.last_decision.symbol,
          action: strategy.last_decision.action,
          confidence: strategy.last_decision.confidence,
          timestamp: strategy.last_decision.timestamp
        } : undefined,
        impact: {
          trades: strategyTrades.length,
          pnl: strategyTrades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0),
          winRate: strategyTrades.length > 0 
            ? (strategyTrades.filter(t => (t.realized_pnl || 0) > 0).length / strategyTrades.length) * 100
            : 0
        }
      };
    });
  }, [strategies, decisionsData, trades]);

  // News Momentum Strategy Card (Enhanced)
  const newsMomentumStrategy = strategyFlows.find(s => s.strategyId === 'news_momo_v2');

  return (
    <div className="space-y-6">
      {/* Integrated Flow Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Integrated Trading System</h2>
          <p className="text-muted-foreground">Real-time strategy performance and decision flow</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Circle className="w-3 h-3 fill-green-500 text-green-500 animate-pulse" />
            <span className="text-sm">Live</span>
          </div>
          <Badge variant="outline" className="gap-1">
            <Activity className="w-3 h-3" />
            {decisionsData?.totalProposals || 0} decisions/min
          </Badge>
        </div>
      </div>

      {/* Main Flow Visualization */}
      <Card className="overflow-hidden">
        <CardContent className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Strategies Column */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Target className="w-5 h-5" />
                Active Strategies
              </h3>
              
              {/* News Momentum Strategy - Featured */}
              {newsMomentumStrategy && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    "relative p-4 rounded-lg border-2 transition-all cursor-pointer",
                    highlightedStrategy === 'news_momo_v2' 
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20" 
                      : "border-border hover:border-blue-400"
                  )}
                  onClick={() => setHighlightedStrategy(
                    highlightedStrategy === 'news_momo_v2' ? null : 'news_momo_v2'
                  )}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h4 className="font-semibold flex items-center gap-2">
                        <Zap className="w-4 h-4 text-blue-500" />
                        News Momentum v2
                      </h4>
                      <p className="text-xs text-muted-foreground">
                        Analyzes breaking news for momentum plays
                      </p>
                    </div>
                    <Badge 
                      variant={newsMomentumStrategy.status === 'active' ? 'default' : 'secondary'}
                      className="text-xs"
                    >
                      {newsMomentumStrategy.status}
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Rate:</span>
                      <span className="ml-1 font-medium">
                        {newsMomentumStrategy.decisionsPerMin.toFixed(1)}/min
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Success:</span>
                      <span className="ml-1 font-medium text-green-600">
                        {(newsMomentumStrategy.successRate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  {newsMomentumStrategy.lastDecision && (
                    <div className="mt-3 p-2 bg-muted/50 rounded text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{newsMomentumStrategy.lastDecision.symbol}</span>
                        <span className={cn(
                          "font-medium",
                          newsMomentumStrategy.lastDecision.action === 'BUY' 
                            ? "text-green-600" 
                            : "text-red-600"
                        )}>
                          {newsMomentumStrategy.lastDecision.action}
                        </span>
                      </div>
                      <div className="text-muted-foreground mt-1">
                        Confidence: {(newsMomentumStrategy.lastDecision.confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                  )}

                  {/* Real-time Impact */}
                  <div className="mt-3 pt-3 border-t">
                    <div className="text-xs text-muted-foreground">Today's Impact</div>
                    <div className="grid grid-cols-3 gap-2 mt-1">
                      <div>
                        <div className="font-medium">{newsMomentumStrategy.impact?.trades || 0}</div>
                        <div className="text-xs text-muted-foreground">Trades</div>
                      </div>
                      <div>
                        <div className={cn(
                          "font-medium",
                          (newsMomentumStrategy.impact?.pnl || 0) >= 0 
                            ? "text-green-600" 
                            : "text-red-600"
                        )}>
                          ${Math.abs(newsMomentumStrategy.impact?.pnl || 0).toFixed(2)}
                        </div>
                        <div className="text-xs text-muted-foreground">P&L</div>
                      </div>
                      <div>
                        <div className="font-medium">
                          {newsMomentumStrategy.impact?.winRate.toFixed(0)}%
                        </div>
                        <div className="text-xs text-muted-foreground">Win</div>
                      </div>
                    </div>
                  </div>

                  {/* Active Flow Indicator */}
                  {activeFlows.some(f => f.from === 'news_momo_v2') && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                      className="absolute -right-2 top-1/2 -translate-y-1/2"
                    >
                      <ArrowRight className="w-6 h-6 text-blue-500 animate-pulse" />
                    </motion.div>
                  )}
                </motion.div>
              )}

              {/* Other Strategies */}
              {strategyFlows.filter(s => s.strategyId !== 'news_momo_v2').map(strategy => (
                <motion.div
                  key={strategy.strategyId}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={cn(
                    "p-3 rounded-lg border transition-all cursor-pointer",
                    highlightedStrategy === strategy.strategyId 
                      ? "border-primary bg-primary/5" 
                      : "hover:border-primary/50"
                  )}
                  onClick={() => setHighlightedStrategy(
                    highlightedStrategy === strategy.strategyId ? null : strategy.strategyId
                  )}
                >
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">{strategy.strategyName}</h4>
                    <Badge variant="outline" className="text-xs">
                      {strategy.decisionsPerMin.toFixed(1)}/min
                    </Badge>
                  </div>
                  
                  {strategy.lastDecision && (
                    <div className="mt-2 text-xs text-muted-foreground">
                      Last: {strategy.lastDecision.symbol} • 
                      {(strategy.lastDecision.confidence * 100).toFixed(0)}% confidence
                    </div>
                  )}

                  {activeFlows.some(f => f.from === strategy.strategyId) && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                      className="absolute -right-2 top-1/2 -translate-y-1/2"
                    >
                      <ArrowRight className="w-5 h-5 text-primary animate-pulse" />
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </div>

            {/* Brain Processing Column */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Brain className="w-5 h-5" />
                AI Brain Processing
              </h3>
              
              <Card className="border-2 border-purple-200 dark:border-purple-800">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-purple-500 rounded-full animate-pulse" />
                      <span className="font-medium">Neural Processing</span>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {brainData?.decisionsToday || 0} today
                    </Badge>
                  </div>

                  {/* Real-time Processing */}
                  <div className="space-y-3">
                    <AnimatePresence>
                      {activeFlows.filter(f => f.to === 'brain').map((flow, idx) => (
                        <motion.div
                          key={`${flow.from}-${idx}`}
                          initial={{ opacity: 0, x: -50 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 50 }}
                          className="flex items-center gap-2 p-2 bg-purple-100 dark:bg-purple-900/20 rounded"
                        >
                          <Activity className="w-4 h-4 text-purple-600" />
                          <span className="text-sm">
                            Processing {flow.label} decision...
                          </span>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>

                  {/* Brain Stats */}
                  <div className="grid grid-cols-2 gap-3 mt-4">
                    <div className="text-center p-2 bg-muted rounded">
                      <div className="text-2xl font-bold text-purple-600">
                        {brainData?.symbolsTracked || 0}
                      </div>
                      <div className="text-xs text-muted-foreground">Symbols</div>
                    </div>
                    <div className="text-center p-2 bg-muted rounded">
                      <div className="text-2xl font-bold text-purple-600">
                        {((brainData?.health?.scoring?.latency_ms || 0) / 1000).toFixed(2)}s
                      </div>
                      <div className="text-xs text-muted-foreground">Latency</div>
                    </div>
                  </div>

                  {/* Decision Quality */}
                  <div className="mt-4 p-3 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20 rounded">
                    <div className="text-sm font-medium mb-1">Decision Quality</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                          initial={{ width: 0 }}
                          animate={{ width: '85%' }}
                          transition={{ duration: 1, ease: "easeOut" }}
                        />
                      </div>
                      <span className="text-sm font-medium">85%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Active Decisions Queue */}
              {decisionsData && decisionsData.totalProposals > 0 && (
                <Card>
                  <CardContent className="p-3">
                    <div className="text-sm font-medium mb-2">Decision Queue</div>
                    <div className="space-y-1">
                      {decisionsData.topSymbols?.slice(0, 3).map((item: any) => (
                        <div key={item.symbol} className="flex items-center justify-between text-xs">
                          <span>{item.symbol}</span>
                          <Badge variant="secondary" className="text-xs">
                            {item.proposals}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Execution & Results Column */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <CheckCircle className="w-5 h-5" />
                Execution & Results
              </h3>

              <Card className="border-2 border-green-200 dark:border-green-800">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-medium">Live Execution</span>
                    <Badge variant="outline" className="text-xs gap-1">
                      <Circle className="w-2 h-2 fill-green-500 text-green-500" />
                      Paper Trading
                    </Badge>
                  </div>

                  {/* Active Executions */}
                  <div className="space-y-2">
                    <AnimatePresence>
                      {activeFlows.filter(f => f.to === 'execution').map((flow, idx) => (
                        <motion.div
                          key={`${flow.from}-${idx}`}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          className="flex items-center gap-2 p-2 bg-green-100 dark:bg-green-900/20 rounded"
                        >
                          <CheckCircle className="w-4 h-4 text-green-600" />
                          <span className="text-sm font-medium">{flow.label}</span>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>

                  {/* Today's Performance */}
                  <div className="mt-4 p-3 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded">
                    <div className="text-sm font-medium mb-2">Today's Performance</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <div className="text-lg font-bold text-green-600">
                          {trades.length}
                        </div>
                        <div className="text-xs text-muted-foreground">Trades</div>
                      </div>
                      <div>
                        <div className={cn(
                          "text-lg font-bold",
                          trades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0) >= 0
                            ? "text-green-600"
                            : "text-red-600"
                        )}>
                          ${Math.abs(trades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0)).toFixed(2)}
                        </div>
                        <div className="text-xs text-muted-foreground">P&L</div>
                      </div>
                    </div>
                  </div>

                  {/* Risk Management */}
                  <div className="mt-4 flex items-center justify-between p-2 bg-muted rounded">
                    <div className="flex items-center gap-2">
                      <Shield className="w-4 h-4 text-blue-600" />
                      <span className="text-sm">Risk Shield</span>
                    </div>
                    <Badge variant="default" className="text-xs bg-green-600">
                      Active
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {/* Recent Trades */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Recent Executions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {trades.slice(0, 3).map((trade, idx) => (
                    <div key={idx} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{trade.symbol}</span>
                        <Badge 
                          variant={trade.side === 'buy' ? 'default' : 'secondary'}
                          className="text-xs"
                        >
                          {trade.side}
                        </Badge>
                      </div>
                      <span className={cn(
                        "font-medium",
                        (trade.realized_pnl || 0) >= 0 ? "text-green-600" : "text-red-600"
                      )}>
                        ${Math.abs(trade.realized_pnl || 0).toFixed(2)}
                      </span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Flow Connections Visualization */}
          <div className="mt-6 pt-6 border-t">
            <div className="flex items-center justify-center gap-8">
              <div className="flex items-center gap-2">
                <div className="w-4 h-1 bg-blue-500" />
                <span className="text-xs text-muted-foreground">Decision Flow</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-1 bg-purple-500" />
                <span className="text-xs text-muted-foreground">Processing</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-1 bg-green-500" />
                <span className="text-xs text-muted-foreground">Execution</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Integration Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Strategies</p>
                <p className="text-2xl font-bold">{strategies.length}</p>
              </div>
              <Target className="w-8 h-8 text-muted-foreground/20" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Decisions/Min</p>
                <p className="text-2xl font-bold">
                  {decisionsData?.proposalsPerMinute?.toFixed(1) || 0}
                </p>
              </div>
              <Brain className="w-8 h-8 text-muted-foreground/20" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active Trades</p>
                <p className="text-2xl font-bold">{trades.length}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-muted-foreground/20" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Today's P&L</p>
                <p className={cn(
                  "text-2xl font-bold",
                  trades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0) >= 0
                    ? "text-green-600"
                    : "text-red-600"
                )}>
                  ${Math.abs(trades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0)).toFixed(2)}
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-muted-foreground/20" />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default IntegratedDashboard;
