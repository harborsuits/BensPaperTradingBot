import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Brain, Zap, AlertTriangle, CheckCircle, Clock, TrendingUp, Users, DollarSign, Activity } from 'lucide-react';
import { SimpleCard } from '@/components/ui/SimpleCard';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { apiClient } from '@/services/apiClient';

interface AIStatus {
  is_active: boolean;
  last_run: string;
  total_cycles: number;
  current_regime: string;
  recent_decisions: any[];
  policy_version: string;
  timestamp: string;
  circuit_breakers?: any[];
}

interface AIContext {
  regime: string;
  volatility: string;
  sentiment: string;
  vix_level: number;
  calendar_events: any[];
  roster_metrics: {
    total_strategies: number;
    by_stage: Record<string, number>;
    by_status: Record<string, number>;
    avg_sharpe: number;
    avg_pf: number;
    underperformers: number;
  };
  capacity: {
    paper_budget_used: number;
    paper_budget_max: number;
    paper_budget_available: number;
    slots_r1_available: number;
    slots_r2_available: number;
    slots_r3_available: number;
  };
  timestamp: string;
}

const AIOrchestratorStatus: React.FC = () => {
  const [aiStatus, setAiStatus] = useState<AIStatus | null>(null);
  const [aiContext, setAiContext] = useState<AIContext | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Fetch AI status
  const { data: statusData, isLoading: statusLoading, error: statusError, refetch: refetchStatus } = useQuery({
    queryKey: ['ai-status'],
    queryFn: async () => {
      const response = await apiClient.get('/live/ai/status');
      return response.data;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch AI context
  const { data: contextData, isLoading: contextLoading, refetch: refetchContext } = useQuery({
    queryKey: ['ai-context'],
    queryFn: async () => {
      const response = await apiClient.get('/live/ai/context');
      return response.data;
    },
    refetchInterval: 60000, // Refresh every minute
  });

  useEffect(() => {
    if (statusData) {
      setAiStatus(statusData);
      setLastUpdate(new Date());
    }
  }, [statusData]);

  useEffect(() => {
    if (contextData) {
      setAiContext(contextData);
    }
  }, [contextData]);

  const handleManualTrigger = async () => {
    try {
      await apiClient.post('/api/ai/trigger-cycle');
      // Refetch status after manual trigger
      setTimeout(() => {
        refetchStatus();
        refetchContext();
      }, 2000);
    } catch (error) {
      console.error('Failed to trigger AI cycle:', error);
    }
  };

  const getRegimeColor = (regime: string) => {
    switch (regime?.toLowerCase()) {
      case 'trending':
      case 'bull':
        return 'text-green-400';
      case 'bear':
        return 'text-red-400';
      case 'choppy':
      case 'neutral':
        return 'text-yellow-400';
      case 'volatile':
        return 'text-orange-400';
      default:
        return 'text-gray-400';
    }
  };

  const getVolatilityColor = (volatility: string) => {
    switch (volatility?.toLowerCase()) {
      case 'low':
        return 'text-green-400';
      case 'medium':
        return 'text-yellow-400';
      case 'high':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    if (!timestamp) return 'Never';
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${Math.floor(diffHours / 24)}d ago`;
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  if (statusLoading || contextLoading) {
    return (
      <SimpleCard title="AI Orchestrator Status">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <span className="ml-2 text-muted-foreground">Loading AI status...</span>
        </div>
      </SimpleCard>
    );
  }

  if (statusError) {
    return (
      <SimpleCard title="AI Orchestrator Status">
        <div className="flex items-center justify-center py-8 text-red-500">
          <AlertTriangle className="h-8 w-8 mr-2" />
          <span>Failed to load AI status</span>
        </div>
      </SimpleCard>
    );
  }

  return (
    <SimpleCard
      title={
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" />
          AI Orchestrator Status
          {aiStatus?.is_active && (
            <div className="flex items-center gap-1">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-green-400 font-medium">ACTIVE</span>
            </div>
          )}
        </div>
      }
      action={
        <button
          onClick={handleManualTrigger}
          className="text-sm text-primary hover:text-primary/80 transition-colors flex items-center gap-1"
        >
          <Zap className="h-4 w-4" />
          Trigger Cycle
        </button>
      }
    >
      <div className="space-y-6">
        {/* AI Status Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="flex items-center justify-center mb-2">
              <Activity className="h-5 w-5 text-blue-400" />
            </div>
            <div className="text-2xl font-bold text-foreground">{aiStatus?.total_cycles || 0}</div>
            <div className="text-xs text-muted-foreground">Cycles Run</div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center mb-2">
              <Clock className="h-5 w-5 text-green-400" />
            </div>
            <div className="text-lg font-semibold text-foreground">
              {formatTimeAgo(aiStatus?.last_run || '')}
            </div>
            <div className="text-xs text-muted-foreground">Last Run</div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center mb-2">
              <Users className="h-5 w-5 text-purple-400" />
            </div>
            <div className="text-2xl font-bold text-foreground">
              {aiContext?.roster_metrics?.total_strategies || 0}
            </div>
            <div className="text-xs text-muted-foreground">Total Strategies</div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center mb-2">
              <DollarSign className="h-5 w-5 text-yellow-400" />
            </div>
            <div className="text-lg font-semibold text-foreground">
              {formatCurrency(aiContext?.capacity?.paper_budget_available || 0)}
            </div>
            <div className="text-xs text-muted-foreground">Budget Available</div>
          </div>
        </div>

        {/* Market Context */}
        <div className="border-t border-border pt-4">
          <h4 className="text-sm font-medium text-foreground mb-3">Market Context</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-muted-foreground mb-1">Regime</div>
              <div className={`text-sm font-medium ${getRegimeColor(aiContext?.regime)}`}>
                {aiContext?.regime || 'Unknown'}
              </div>
            </div>

            <div>
              <div className="text-xs text-muted-foreground mb-1">Volatility</div>
              <div className={`text-sm font-medium ${getVolatilityColor(aiContext?.volatility)}`}>
                {aiContext?.volatility || 'Unknown'}
              </div>
            </div>

            <div>
              <div className="text-xs text-muted-foreground mb-1">VIX Level</div>
              <div className="text-sm font-medium text-foreground">
                {aiContext?.vix_level ? aiContext.vix_level.toFixed(1) : 'N/A'}
              </div>
            </div>
          </div>
        </div>

        {/* Tournament Stages */}
        <div className="border-t border-border pt-4">
          <h4 className="text-sm font-medium text-foreground mb-3">Tournament Stages</h4>
          <div className="grid grid-cols-4 gap-2">
            {['R1', 'R2', 'R3', 'LIVE'].map((stage) => {
              const count = aiContext?.roster_metrics?.by_stage?.[stage] || 0;
              const max = stage === 'R1' ? 50 : stage === 'R2' ? 20 : stage === 'R3' ? 8 : 10;
              const utilization = (count / max) * 100;

              return (
                <div key={stage} className="text-center">
                  <div className="text-xs text-muted-foreground mb-1">{stage}</div>
                  <div className="text-lg font-bold text-foreground">{count}</div>
                  <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                    <div
                      className={`h-1 rounded-full ${
                        utilization > 80 ? 'bg-red-500' :
                        utilization > 60 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${Math.min(utilization, 100)}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Recent Decisions */}
        {aiStatus?.recent_decisions && aiStatus.recent_decisions.length > 0 && (
          <div className="border-t border-border pt-4">
            <h4 className="text-sm font-medium text-foreground mb-3">Recent AI Decisions</h4>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {aiStatus.recent_decisions.map((decision: any, idx: number) => (
                <div key={idx} className="text-xs bg-muted/50 rounded p-2">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-foreground">
                      {decision.type}: {decision.strategyId?.slice(-8) || 'Unknown'}
                    </span>
                    <span className="text-muted-foreground">
                      {formatTimeAgo(decision.timestamp)}
                    </span>
                  </div>
                  <div className="text-muted-foreground mt-1">
                    {decision.reason || 'No reason provided'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Semantic Similarity Status */}
        {aiStatus?.semantic_data && aiStatus.semantic_data.enabled && (
          <div className="border-t border-border pt-4">
            <h4 className="text-sm font-medium text-foreground mb-3 flex items-center gap-2">
              <Brain className="h-4 w-4 text-purple-500" />
              Semantic Memory Active
            </h4>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-muted-foreground">Prior Contribution:</span>
                  <span className="ml-2 font-medium text-foreground">
                    {(aiStatus.semantic_data.prior_contribution * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Confidence:</span>
                  <span className="ml-2 font-medium text-foreground">
                    {(aiStatus.semantic_data.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {aiStatus.semantic_data.similar_cases && aiStatus.semantic_data.similar_cases.length > 0 && (
                <div>
                  <div className="text-xs text-muted-foreground mb-2">Similar Historical Cases:</div>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {aiStatus.semantic_data.similar_cases.map((case_: any, idx: number) => (
                      <div key={idx} className="bg-muted/30 rounded p-2 text-xs">
                        <div className="flex justify-between items-start mb-1">
                          <span className="text-muted-foreground">Similarity:</span>
                          <span className="font-medium text-foreground">
                            {(case_.similarity * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-foreground line-clamp-2 mb-1">
                          {case_.text}
                        </div>
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Return: {(case_.historical_return * 100).toFixed(1)}%</span>
                          <span>N={case_.sample_size}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Circuit Breaker Status */}
        {aiStatus?.circuit_breakers && aiStatus.circuit_breakers.length > 0 && (
          <div className="border-t border-border pt-4">
            <h4 className="text-sm font-medium text-foreground mb-3 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              Circuit Breakers Active
            </h4>
            <div className="space-y-2">
              {aiStatus.circuit_breakers.map((cb: any, idx: number) => (
                <div key={idx} className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded p-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs px-2 py-1 rounded font-medium ${
                        cb.severity === 'critical' ? 'bg-red-500 text-white' :
                        cb.severity === 'high' ? 'bg-orange-500 text-white' :
                        'bg-yellow-500 text-black'
                      }`}>
                        {cb.severity.toUpperCase()}
                      </span>
                      <span className="text-sm font-medium text-foreground">{cb.type.replace('_', ' ')}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">{cb.action.replace('_', ' ')}</span>
                  </div>
                  <p className="text-sm text-red-700 dark:text-red-300 mt-1">{cb.reason}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Status Footer */}
        <div className="border-t border-border pt-4 flex justify-between items-center text-xs text-muted-foreground">
          <span>Policy: {aiStatus?.policy_version || 'latest'}</span>
          <span>Last updated: {formatTimeAgo(lastUpdate.toISOString())}</span>
        </div>
      </div>
    </SimpleCard>
  );
};

export default AIOrchestratorStatus;
