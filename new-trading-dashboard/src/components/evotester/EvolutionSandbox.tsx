/**
 * ============================================
 * [CARD: EVOLUTION SANDBOX]
 * Auto-triggers, capital management, automated experiments, risk controls
 * ============================================
 */

import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import {
  FlaskConical,
  Beaker,
  Zap,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Target,
  Activity,
  BarChart3,
  Settings,
  Play,
  Pause,
  Square,
  RefreshCw,
  Bell,
  BellOff,
  Loader2
} from 'lucide-react';
import TimeSeriesChart from '@/components/ui/TimeSeriesChart';
import { useAutoTrigger } from '@/hooks/useAutoTrigger';
import { useSegregatedCapital } from '@/hooks/useSegregatedCapital';
import ModeLabel from '@/components/ui/ModeLabel';
import { evoTesterApi, strategyApi } from '@/services/api';

interface SandboxExperiment {
  id: string;
  name: string;
  status: 'running' | 'paused' | 'completed' | 'failed';
  triggerCondition: string;
  startTime: string;
  allocatedCapital: number;
  currentPnl: number;
  bestStrategy: string;
  generations: number;
  successRate: number;
  riskLevel: 'low' | 'medium' | 'high';
  marketCondition: string;
}


interface CapitalAllocation {
  totalResearchCapital: number;
  allocatedCapital: number;
  availableCapital: number;
  maxPerExperiment: number;
  riskMultiplier: number;
}

interface EvolutionSandboxProps {
  className?: string;
}

const EvolutionSandbox: React.FC<EvolutionSandboxProps> = ({ className = '' }) => {
  const [activeExperiments, setActiveExperiments] = useState<SandboxExperiment[]>([]);
  const [capitalAllocation, setCapitalAllocation] = useState<CapitalAllocation>({
    totalResearchCapital: 10000,
    allocatedCapital: 2500,
    availableCapital: 7500,
    maxPerExperiment: 1000,
    riskMultiplier: 0.1
  });

  // Use auto-trigger service
  const {
    rules: triggerRules,
    config: triggerConfig,
    isMonitoring,
    activeTriggers,
    triggerStats,
    startMonitoring,
    stopMonitoring,
    updateRule,
    updateConfig,
    resetTriggerCooldown,
    getTriggerEvents
  } = useAutoTrigger();

  // Use segregated capital service
  const {
    pools: capitalPools,
    allocations: capitalAllocations,
    transactions: capitalTransactions,
    limits: capitalLimits,
    allocateCapital,
    releaseCapital,
    updatePnl,
    transferCapital,
    getPoolAnalytics,
    getActiveAllocations,
    getPoolUtilization
  } = useSegregatedCapital();

  // Real-time data connections - using available API functions
  // Use existing evoTesterApi.getEvoHistory for active experiments
  const { data: liveExperiments, isLoading: experimentsLoading } = useQuery({
    queryKey: ['evoTester', 'experiments', 'active'],
    queryFn: () => evoTesterApi.getEvoHistory(),
    refetchInterval: 20000, // Refresh every 20 seconds
    staleTime: 10000,
  });

  // Mock capital data since the API function doesn't exist yet
  const liveCapitalData = { capital: { total: 10000, allocated: 2500, available: 7500 } };
  const capitalLoading = false;

  // Mock trigger data since the API function doesn't exist yet
  const liveTriggerData = { triggers: [] };
  const triggerLoading = false;

  // Mock data for demonstration
  useEffect(() => {
    const mockExperiments: SandboxExperiment[] = [
      {
        id: 'evo_sandbox_001',
        name: 'Volatility Spike Research',
        status: 'running',
        triggerCondition: 'Market Volatility > 2.5σ',
        startTime: '2024-04-30T09:30:00Z',
        allocatedCapital: 500,
        currentPnl: 47.23,
        bestStrategy: 'adaptive_volatility_v7',
        generations: 12,
        successRate: 73.2,
        riskLevel: 'low',
        marketCondition: 'High Volatility'
      },
      {
        id: 'evo_sandbox_002',
        name: 'Bear Market Adaptation',
        status: 'running',
        triggerCondition: 'Market Regime Change',
        startTime: '2024-04-30T08:15:00Z',
        allocatedCapital: 750,
        currentPnl: -23.45,
        bestStrategy: 'defensive_ma_crossover_v3',
        generations: 8,
        successRate: 65.8,
        riskLevel: 'medium',
        marketCondition: 'Bear Trend'
      },
      {
        id: 'evo_sandbox_003',
        name: 'News Impact Study',
        status: 'completed',
        triggerCondition: 'High News Sentiment',
        startTime: '2024-04-30T07:00:00Z',
        allocatedCapital: 300,
        currentPnl: 89.12,
        bestStrategy: 'sentiment_aware_v5',
        generations: 15,
        successRate: 81.4,
        riskLevel: 'low',
        marketCondition: 'News Driven'
      }
    ];

    setActiveExperiments(mockExperiments);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />;
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-600" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'failed':
        return <AlertTriangle className="w-4 h-4 text-red-600" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'high':
        return 'bg-red-100 text-red-800 border-red-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const handleExperimentAction = (experimentId: string, action: 'pause' | 'resume' | 'stop') => {
    setActiveExperiments(prev =>
      prev.map(exp =>
        exp.id === experimentId
          ? { ...exp, status: action === 'pause' ? 'paused' : action === 'resume' ? 'running' : 'completed' }
          : exp
      )
    );
  };


  const renderExperimentsView = () => (
    <div className="space-y-4">
      {/* Active Experiments */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <FlaskConical className="w-5 h-5 mr-2 text-purple-500" />
          Active Experiments
          {experimentsLoading && <Loader2 className="w-4 h-4 ml-2 animate-spin text-purple-500" />}
        </h3>
        <Button size="sm" variant="outline">
          <RefreshCw className="w-3 h-3 mr-1" />
          Refresh
        </Button>
      </div>
      <div className="grid gap-4">
        {(liveExperiments?.experiments || activeExperiments).map((experiment) => (
          <Card key={experiment.id} className="border-l-4 border-l-purple-500">
            <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <FlaskConical className="w-5 h-5 text-purple-500" />
                    <div>
                      <h3 className="font-semibold text-foreground">{experiment.name}</h3>
                      <p className="text-sm text-foreground">{experiment.triggerCondition}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <ModeLabel mode="research" size="sm" status={experiment.status} />
                      <Badge className={getRiskColor(experiment.riskLevel)}>
                        {experiment.riskLevel.toUpperCase()} RISK
                      </Badge>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(experiment.status)}
                    <Badge variant={
                      experiment.status === 'running' ? 'default' :
                      experiment.status === 'paused' ? 'secondary' :
                      experiment.status === 'completed' ? 'outline' : 'destructive'
                    }>
                      {experiment.status}
                    </Badge>
                  </div>
                </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                <div>
                  <span className="text-xs text-gray-500">Allocated Capital</span>
                  <div className="font-medium text-green-600">${experiment.allocatedCapital}</div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Current P&L</span>
                  <div className={`font-medium ${experiment.currentPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ${experiment.currentPnl.toFixed(2)}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Best Strategy</span>
                  <div className="font-medium text-blue-600">{experiment.bestStrategy}</div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Success Rate</span>
                  <div className="font-medium text-purple-600">{experiment.successRate}%</div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-600">
                  Started: {new Date(experiment.startTime).toLocaleString()} |
                  Generations: {experiment.generations} |
                  Market: {experiment.marketCondition}
                </div>
                <div className="flex space-x-2">
                  {experiment.status === 'running' && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleExperimentAction(experiment.id, 'pause')}
                    >
                      <Pause className="w-3 h-3 mr-1" /> Pause
                    </Button>
                  )}
                  {experiment.status === 'paused' && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleExperimentAction(experiment.id, 'resume')}
                    >
                      <Play className="w-3 h-3 mr-1" /> Resume
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => handleExperimentAction(experiment.id, 'stop')}
                  >
                    <Square className="w-3 h-3 mr-1" /> Stop
                  </Button>
                  <Button size="sm" variant="outline">
                    <BarChart3 className="w-3 h-3 mr-1" /> Details
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600 mb-1">
              ${capitalAllocation.availableCapital.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Available Research Capital</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600 mb-1">
              {activeExperiments.filter(e => e.status === 'running').length}
            </div>
            <div className="text-sm text-gray-600">Active Experiments</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-purple-600 mb-1">
              {activeExperiments.reduce((sum, exp) => sum + exp.currentPnl, 0).toFixed(2)}
            </div>
            <div className="text-sm text-gray-600">Total Research P&L</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-orange-600 mb-1">
              {triggerRules.filter(t => t.active).length}
            </div>
            <div className="text-sm text-gray-600">Active Triggers</div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  const renderTriggersView = () => {
    const triggerEvents = getTriggerEvents();

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">Auto-Trigger Conditions</h3>
            <p className="text-sm text-gray-600">Conditions that automatically launch research experiments</p>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className={isMonitoring ? 'border-green-300 text-green-700' : 'border-gray-300 text-gray-600'}>
              <Activity className={`w-3 h-3 mr-1 ${isMonitoring ? 'text-green-500' : 'text-gray-400'}`} />
              {isMonitoring ? 'Monitoring' : 'Stopped'}
            </Badge>
            <Button
              variant={triggerConfig.enabled ? "default" : "outline"}
              onClick={triggerConfig.enabled ? stopMonitoring : startMonitoring}
            >
              {triggerConfig.enabled ? (
                <>
                  <BellOff className="w-4 h-4 mr-2" />
                  Stop Monitoring
                </>
              ) : (
                <>
                  <Bell className="w-4 h-4 mr-2" />
                  Start Monitoring
                </>
              )}
            </Button>
          </div>
        </div>

        <div className="grid gap-4">
          {triggerRules.map((trigger) => (
            <Card key={trigger.id} className={trigger.active ? 'border-green-300' : 'border-gray-200'}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${trigger.active ? 'bg-green-500' : 'bg-gray-400'}`} />
                    <div>
                      <h4 className="font-medium text-foreground">{trigger.name}</h4>
                      <p className="text-sm text-foreground">{trigger.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-xs">
                      Priority {trigger.priority}
                    </Badge>
                    <Button
                      size="sm"
                      variant={trigger.active ? "default" : "outline"}
                      onClick={() => updateRule(trigger.id, { active: !trigger.active })}
                    >
                      {trigger.active ? 'Active' : 'Inactive'}
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Trigger Count</span>
                    <div className="font-medium">{trigger.triggerCount}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Success Rate</span>
                    <div className="font-medium text-green-600">{trigger.successRate.toFixed(1)}%</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Last Triggered</span>
                    <div className="font-medium">
                      {trigger.lastTriggered
                        ? new Date(trigger.lastTriggered).toLocaleString()
                        : 'Never'
                      }
                    </div>
                  </div>
                </div>

                {trigger.lastTriggered && (
                  <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
                    <span>Cooldown: {trigger.cooldownMinutes} minutes</span>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-xs h-6"
                      onClick={() => resetTriggerCooldown(trigger.id)}
                    >
                      Reset Cooldown
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Trigger Events History */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Recent Trigger Events</CardTitle>
          </CardHeader>
          <CardContent>
            {triggerEvents.length > 0 ? (
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {triggerEvents.slice(0, 10).map((event) => (
                  <div key={event.id} className="flex items-center justify-between p-2 bg-gray-50 rounded text-sm">
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className="text-xs">
                        {event.type.replace('_', ' ')}
                      </Badge>
                      <span className="text-gray-700">
                        {triggerRules.find(r => r.id === event.ruleId)?.name || event.ruleId}
                      </span>
                    </div>
                    <span className="text-gray-500 text-xs">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">
                No trigger events yet
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Trigger Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {triggerRules.map((trigger) => (
                <div key={trigger.id} className="flex items-center justify-between">
                  <span className="text-sm font-medium">{trigger.name}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-500 h-2 rounded-full"
                        style={{ width: `${Math.min(trigger.successRate, 100)}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-600 w-12">{trigger.successRate.toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderCapitalView = () => {
    const researchPool = capitalPools.find(p => p.id === 'research_pool');
    const competitionPool = capitalPools.find(p => p.id === 'competition_pool');
    const validationPool = capitalPools.find(p => p.id === 'validation_pool');

    return (
      <div className="space-y-6">
        {/* Pool Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {capitalPools.map((pool) => {
            const analytics = getPoolAnalytics(pool.id);
            const utilization = getPoolUtilization(pool.id);

            return (
              <Card key={pool.id} className={`border-l-4 ${
                pool.purpose === 'research' ? 'border-l-purple-500' :
                pool.purpose === 'competition' ? 'border-l-blue-500' :
                'border-l-green-500'
              }`}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-foreground">{pool.name}</h3>
                    <Badge variant="outline" className="text-xs">
                      {pool.riskLevel.toUpperCase()}
                    </Badge>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total:</span>
                      <span className="font-medium">${pool.totalCapital.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Available:</span>
                      <span className="font-medium text-green-600">${pool.availableCapital.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Allocated:</span>
                      <span className="font-medium text-blue-600">${pool.allocatedCapital.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Utilization:</span>
                      <span className="font-medium">{(utilization * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="mt-3">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          utilization > 0.8 ? 'bg-red-500' :
                          utilization > 0.6 ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${utilization * 100}%` }}
                      />
                    </div>
                  </div>

                  {analytics && (
                    <div className="mt-3 pt-3 border-t text-xs">
                      <div className="flex justify-between">
                        <span>Active Expts:</span>
                        <span className="font-medium">{analytics.activeExperiments}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Win Rate:</span>
                        <span className="font-medium text-green-600">{(analytics.winRate * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Risk Parameters */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Settings className="w-5 h-5 mr-2" />
              Risk Parameters & Limits
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="font-medium text-foreground">Per-Experiment Limits</h4>
                <div className="space-y-2">
                  {Object.entries(capitalLimits.maxPerExperiment).map(([risk, amount]) => (
                    <div key={risk} className="flex justify-between items-center">
                      <span className={`text-sm capitalize ${
                        risk === 'low' ? 'text-green-600' :
                        risk === 'medium' ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {risk} Risk:
                      </span>
                      <span className="font-medium">${amount.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-medium text-foreground">Global Limits</h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Max Concurrent:</span>
                    <span className="font-medium">{capitalLimits.maxConcurrentExperiments}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Max Drawdown:</span>
                    <span className="font-medium">{(capitalLimits.maxTotalDrawdown * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Emergency Stop:</span>
                    <span className="font-medium text-red-600">{(capitalLimits.emergencyStopLoss * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Transactions */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Capital Transactions</CardTitle>
          </CardHeader>
          <CardContent>
            {capitalTransactions.length > 0 ? (
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {capitalTransactions.slice(-10).reverse().map((transaction) => (
                  <div key={transaction.id} className="flex items-center justify-between p-2 bg-gray-50 rounded text-sm">
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className="text-xs">
                        {transaction.type}
                      </Badge>
                      <span className="text-foreground">{transaction.description}</span>
                    </div>
                    <div className="text-right">
                      <div className={`font-medium ${
                        transaction.amount >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {transaction.amount >= 0 ? '+' : ''}${transaction.amount.toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(transaction.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">
                No transactions yet
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center">
              <FlaskConical className="w-5 h-5 mr-2 text-purple-500" />
              Evolution Sandbox
            </CardTitle>
            <p className="text-sm text-gray-600 mt-1">
              Autonomous research experiments with segregated capital
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <ModeLabel mode="research" size="sm" />
            <Badge variant="outline">
              <Activity className="w-3 h-3 mr-1" />
              {activeExperiments.filter(e => e.status === 'running').length} Active
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="experiments">
          <TabsList className="mb-6">
            <TabsTrigger value="experiments">Active Experiments</TabsTrigger>
            <TabsTrigger value="triggers">Auto-Triggers</TabsTrigger>
            <TabsTrigger value="capital">Capital Management</TabsTrigger>
          </TabsList>

          <TabsContent value="experiments">
            {renderExperimentsView()}
          </TabsContent>

          <TabsContent value="triggers">
            {renderTriggersView()}
          </TabsContent>

          <TabsContent value="capital">
            {renderCapitalView()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default EvolutionSandbox;
