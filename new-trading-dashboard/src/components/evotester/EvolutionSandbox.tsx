/**
 * ============================================
 * [CARD: EVOLUTION SANDBOX]
 * Auto-triggers, capital management, automated experiments, risk controls
 * ============================================
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
  Loader2,
  RotateCcw,
  Shield
} from 'lucide-react';
import ModeLabel from '@/components/ui/ModeLabel';
import { evoTesterApi } from '@/services/api';
import {
  useEvoLedger,
  useEvoPoolStatus,
  useEvoWebSocketHandlers,
  EvoStatus,
  EvoAllocation,
  canTransition,
  getValidActions
} from '@/hooks/useEvoQueries';

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
  // Phase 3: Connect to EVO APIs
  const { data: evoLedger, isLoading: ledgerLoading } = useEvoLedger();
  const { data: evoPoolStatus, isLoading: poolLoading } = useEvoPoolStatus();

  // Enable WebSocket handlers for real-time updates
  useEvoWebSocketHandlers();

  // Legacy experiment data (will be replaced with EVO allocations)
  const [activeExperiments, setActiveExperiments] = useState<SandboxExperiment[]>([]);
  const [capitalAllocation, setCapitalAllocation] = useState<CapitalAllocation>({
    totalResearchCapital: 10000,
    allocatedCapital: 2500,
    availableCapital: 7500,
    maxPerExperiment: 1000,
    riskMultiplier: 0.1
  });

  // Trigger system state
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [triggerConfig, setTriggerConfig] = useState({
    enabled: false,
    lastCheck: new Date(),
    checkInterval: 30000
  });

  // Mock trigger rules data
  const [triggerRules] = useState([
    {
      id: 'volatility_spike',
      name: 'Volatility Spike',
      description: 'Market volatility exceeds 2.5σ threshold',
      active: true,
      priority: 1,
      triggerCount: 12,
      successRate: 73.2,
      lastTriggered: '2024-04-30T14:32:00Z',
      cooldownMinutes: 30
    },
    {
      id: 'volume_anomaly',
      name: 'Volume Anomaly',
      description: 'Unusual trading volume detected',
      active: true,
      priority: 2,
      triggerCount: 8,
      successRate: 65.4,
      lastTriggered: '2024-04-29T10:15:00Z',
      cooldownMinutes: 15
    },
    {
      id: 'sentiment_shift',
      name: 'Sentiment Shift',
      description: 'Market sentiment changes significantly',
      active: false,
      priority: 3,
      triggerCount: 5,
      successRate: 58.7,
      lastTriggered: '2024-04-28T16:45:00Z',
      cooldownMinutes: 60
    }
  ]);

  // Capital pools state
  const [capitalPools] = useState([
    {
      id: 'research_pool',
      name: 'Research Pool',
      type: 'research',
      totalCapital: 50000,
      allocatedCapital: 12500,
      availableCapital: 37500,
      maxAllocation: 5000,
      riskLevel: 'medium',
      activeStrategies: 8,
      pnl: 1250.75,
      pnlPercent: 10.0,
      lastUpdate: new Date().toISOString()
    },
    {
      id: 'competition_pool',
      name: 'Competition Pool',
      type: 'competition',
      totalCapital: 25000,
      allocatedCapital: 18750,
      availableCapital: 6250,
      maxAllocation: 3000,
      riskLevel: 'high',
      activeStrategies: 12,
      pnl: -750.25,
      pnlPercent: -4.0,
      lastUpdate: new Date().toISOString()
    },
    {
      id: 'validation_pool',
      name: 'Validation Pool',
      type: 'validation',
      totalCapital: 15000,
      allocatedCapital: 7500,
      availableCapital: 7500,
      maxAllocation: 2000,
      riskLevel: 'low',
      activeStrategies: 5,
      pnl: 375.50,
      pnlPercent: 5.0,
      lastUpdate: new Date().toISOString()
    }
  ]);

  // Capital limits state
  const [capitalLimits] = useState({
    maxPerExperiment: {
      low: 2000,
      medium: 5000,
      high: 10000
    },
    maxPerStrategy: 2500,
    totalDailyLimit: 25000,
    maxConcurrentExperiments: 15,
    maxTotalDrawdown: 0.15,
    emergencyStopLoss: 0.20
  });

  // Capital transactions state
  const [capitalTransactions] = useState([
    {
      id: 'txn_001',
      type: 'allocation',
      description: 'Strategy S1 allocated $2,500',
      amount: -2500,
      timestamp: '2024-04-30T14:32:00Z'
    },
    {
      id: 'txn_002',
      type: 'return',
      description: 'Strategy S1 returned $2,750 (+10%)',
      amount: 2750,
      timestamp: '2024-04-30T16:15:00Z'
    },
    {
      id: 'txn_003',
      type: 'allocation',
      description: 'Strategy S2 allocated $1,800',
      amount: -1800,
      timestamp: '2024-04-29T10:45:00Z'
    }
  ]);

  // Capital pool utility functions
  const getPoolAnalytics = (poolId: string) => {
    const pool = capitalPools.find(p => p.id === poolId);
    if (!pool) return null;

    return {
      totalReturn: pool.pnl,
      returnPercent: pool.pnlPercent,
      sharpeRatio: pool.riskLevel === 'low' ? 1.8 : pool.riskLevel === 'medium' ? 1.2 : 0.8,
      maxDrawdown: pool.riskLevel === 'low' ? -5.2 : pool.riskLevel === 'medium' ? -12.5 : -18.7,
      winRate: pool.riskLevel === 'low' ? 68.5 : pool.riskLevel === 'medium' ? 54.2 : 43.8
    };
  };

  const getPoolUtilization = (poolId: string) => {
    const pool = capitalPools.find(p => p.id === poolId);
    if (!pool) return 0;

    return (pool.allocatedCapital / pool.totalCapital) * 100;
  };

  // Use existing evoTesterApi for non-EVO experiments
  const { data: liveExperiments, isLoading: experimentsLoading } = useQuery({
    queryKey: ['evoTester', 'experiments', 'active'],
    queryFn: () => evoTesterApi.getEvoHistory(),
    refetchInterval: 20000,
    staleTime: 10000,
  });

  // Debounced state updates to prevent excessive re-renders
  const debouncedSetTriggerConfig = useCallback(
    (updater: any) => {
      // Simple debounce - only update if not recently updated
      setTriggerConfig(prev => {
        const newConfig = typeof updater === 'function' ? updater(prev) : updater;
        // Prevent unnecessary updates if config hasn't actually changed
        return JSON.stringify(prev) === JSON.stringify(newConfig) ? prev : newConfig;
      });
    },
    []
  );

  // Trigger system functions
  const startMonitoring = useCallback(() => {
    setIsMonitoring(true);
    debouncedSetTriggerConfig(prev => ({ ...prev, enabled: true }));
  }, [debouncedSetTriggerConfig]);

  const stopMonitoring = useCallback(() => {
    setIsMonitoring(false);
    debouncedSetTriggerConfig(prev => ({ ...prev, enabled: false }));
  }, [debouncedSetTriggerConfig]);

  const updateRule = useCallback((ruleId: string, updates: Partial<typeof triggerRules[0]>) => {
    // This would normally update the rule in state or backend
    console.log('Updating rule:', ruleId, updates);
  }, []);

  const resetTriggerCooldown = useCallback((ruleId: string) => {
    // This would normally reset the cooldown for a rule
    console.log('Resetting cooldown for rule:', ruleId);
  }, []);

  const getTriggerEvents = () => {
    // Mock trigger events data (will be replaced with real API call)
    return [
      {
        id: 'evt_001',
        type: 'volatility_trigger',
        ruleId: 'volatility_spike',
        timestamp: '2024-04-30T14:32:00Z',
        triggeredBy: 'Market volatility exceeded threshold',
        result: 'Experiment launched successfully'
      },
      {
        id: 'evt_002',
        type: 'volume_trigger',
        ruleId: 'volume_anomaly',
        timestamp: '2024-04-29T10:15:00Z',
        triggeredBy: 'Volume spike detected',
        result: 'Experiment queued'
      }
    ];
  };

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

  // Phase 3: EVO Allocation Action Handlers
  const handleAllocationAction = async (allocationId: string, action: string) => {
    // This will be implemented with proper EVO API calls
    console.log(`EVO Allocation ${allocationId}: ${action}`);
  };

  // Get status color for EVO allocations
  const getEvoStatusColor = (status: EvoStatus) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 border-green-300';
      case 'staged': return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'activating': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'halting': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'halted': return 'bg-gray-100 text-gray-800 border-gray-300';
      case 'expired': return 'bg-red-100 text-red-800 border-red-300';
      case 'failed': return 'bg-red-200 text-red-900 border-red-400';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  // Get status icon for EVO allocations
  const getEvoStatusIcon = (status: EvoStatus) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'staged': return <Clock className="w-4 h-4 text-blue-600" />;
      case 'activating': return <Loader2 className="w-4 h-4 text-yellow-600 animate-spin" />;
      case 'halting': return <Pause className="w-4 h-4 text-orange-600" />;
      case 'halted': return <Square className="w-4 h-4 text-gray-600" />;
      case 'expired': return <AlertTriangle className="w-4 h-4 text-red-600" />;
      case 'failed': return <AlertTriangle className="w-4 h-4 text-red-700" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };


  // Phase 3: EVO Allocations View
  const renderEvoAllocationsView = () => (
    <div className="space-y-4">
      {/* EVO Pool Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold flex items-center">
            <Shield className="w-5 h-5 mr-2 text-green-600" />
            EVO Allocations
            {ledgerLoading && <Loader2 className="w-4 h-4 ml-2 animate-spin text-green-600" />}
          </h3>
          <p className="text-sm text-gray-600">Paper-only EVO strategy allocations with segregated capital</p>
        </div>
        <div className="flex items-center space-x-2">
          {evoPoolStatus && (
            <div className="text-sm text-gray-600">
              Pool: {evoPoolStatus.capPct * 100}% cap, {evoPoolStatus.utilizationPct * 100}% used
            </div>
          )}
          <Button size="sm" variant="outline">
            <RefreshCw className="w-3 h-3 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      {/* EVO Allocations List */}
      <div className="space-y-4">
        {evoLedger?.rows?.map((allocation: EvoAllocation) => (
          <Card key={allocation.id} className="border-l-4 border-l-green-500">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <Shield className="w-5 h-5 text-green-600" />
                  <div>
                    <h4 className="font-semibold text-foreground">{allocation.strategyRef}</h4>
                    <p className="text-sm text-gray-600">Session: {allocation.sessionId}</p>
                  </div>
                  <Badge className={getEvoStatusColor(allocation.status)}>
                    {allocation.status.toUpperCase()}
                  </Badge>
                </div>
                <div className="flex items-center space-x-2">
                  {getEvoStatusIcon(allocation.status)}
                  <Badge variant="outline" className="text-xs">
                    EVO PAPER ONLY
                  </Badge>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                <div>
                  <span className="text-xs text-gray-500">Allocation</span>
                  <div className="font-medium text-green-600">${allocation.allocation}</div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Realized P&L</span>
                  <div className={`font-medium ${allocation.realizedPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ${allocation.realizedPnl.toFixed(2)}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">TTL Until</span>
                  <div className="font-medium text-blue-600">
                    {new Date(allocation.ttlUntil).toLocaleDateString()}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Created</span>
                  <div className="font-medium text-gray-600">
                    {new Date(allocation.createdAt).toLocaleDateString()}
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-600">
                  ID: {allocation.id} |
                  Status: {allocation.status} |
                  {allocation.symbolHint && allocation.symbolHint.length > 0 &&
                    `Symbols: ${allocation.symbolHint.join(', ')}`
                  }
                </div>
                <div className="flex space-x-2">
                  {getValidActions(allocation.status).map(action => (
                    <Button
                      key={action}
                      size="sm"
                      variant="outline"
                      onClick={() => handleAllocationAction(allocation.id, action)}
                      disabled={allocation.status === 'activating' || allocation.status === 'halting'}
                    >
                      {action === 'cancel' && 'Cancel'}
                      {action === 'halt' && 'Halt'}
                      {action === 'renew' && 'Renew'}
                      {action === 'remove' && 'Remove'}
                    </Button>
                  ))}
                  <Button size="sm" variant="outline">
                    <BarChart3 className="w-3 h-3 mr-1" /> Details
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )) || []}

        {(!evoLedger?.rows || evoLedger.rows.length === 0) && !ledgerLoading && (
          <div className="text-center py-8 text-gray-500">
            <Shield className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No EVO Allocations</h3>
            <p className="text-gray-600">Strategies promoted to EVO will appear here</p>
          </div>
        )}
      </div>

      {/* EVO Pool Stats */}
      {evoPoolStatus && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-green-600 mb-1">
                {evoPoolStatus.capPct * 100}%
              </div>
              <div className="text-sm text-gray-600">Pool Cap</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-blue-600 mb-1">
                {evoPoolStatus.utilizationPct * 100}%
              </div>
              <div className="text-sm text-gray-600">Utilization</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-purple-600 mb-1">
                {evoPoolStatus.activeCount}
              </div>
              <div className="text-sm text-gray-600">Active Allocs</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-orange-600 mb-1">
                ${evoPoolStatus.poolPnl.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Total P&L</div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );

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

  // Use React Query for trigger events with proper caching
  const { data: triggerEvents = [], isLoading: triggersLoading } = useQuery({
    queryKey: ['evo', 'trigger-events'],
    queryFn: getTriggerEvents,
    refetchInterval: 30000, // Refetch every 30 seconds
    refetchOnWindowFocus: false,
    staleTime: 25000, // Consider data stale after 25 seconds
  });

  const renderTriggersView = () => {
    if (triggersLoading) {
      return (
        <div className="flex items-center justify-center p-8">
          <div className="text-gray-500">Loading trigger events...</div>
        </div>
      );
    }

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
                        {event.type ? event.type.replace('_', ' ') : 'Unknown'}
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

    // Use the same triggerEvents from React Query above

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
              EVO: Paper-only strategy allocations with segregated capital
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <ModeLabel mode="research" size="sm" />
            <Badge variant="outline">
              <Shield className="w-3 h-3 mr-1" />
              {evoLedger?.rows?.length || 0} EVO Active
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="evo">
          <TabsList className="mb-6">
            <TabsTrigger value="evo">EVO Allocations</TabsTrigger>
            <TabsTrigger value="experiments">Active Experiments</TabsTrigger>
            <TabsTrigger value="triggers">Auto-Triggers</TabsTrigger>
            <TabsTrigger value="capital">Capital Management</TabsTrigger>
          </TabsList>

          <TabsContent value="evo">
            {renderEvoAllocationsView()}
          </TabsContent>

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
