import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Brain,
  Dna,
  Trophy,
  TrendingUp,
  TrendingDown,
  Target,
  Zap,
  Users,
  Award,
  AlertTriangle,
  CheckCircle,
  Clock,
  PlayCircle,
  Activity,
  BarChart3
} from 'lucide-react';

interface TournamentRound {
  stage: string;
  name: string;
  active_strategies: number;
  criteria: {
    minSharpe?: number;
    minPf?: number;
    maxDd?: number;
    maxBreaches?: number;
    maxSlippage?: number;
  };
}

interface TournamentData {
  current_generation: number;
  rounds: TournamentRound[];
  stats: {
    totalPromotions: number;
    totalDemotions: number;
    roundPassRates: Record<string, { promoted: number; demoted: number }>;
  };
  recent_decisions?: Array<{
    strategyId: string;
    decision: string;
    fromStage: string;
    toStage: string;
    reason: string;
    timestamp: string;
  }>;
}

interface BrainEvoFlowProps {
  className?: string;
}

export const BrainEvoFlow: React.FC<BrainEvoFlowProps> = ({ className = '' }) => {
  // Brain status query
  const { data: brainStatus, isLoading: brainLoading, error: brainError, refetch: refetchBrain } = useQuery({
    queryKey: ['brain', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/brain/status');
      if (!response.ok) {
        throw new Error('Failed to fetch brain status');
      }
      return response.json();
    },
    refetchInterval: 30000, // Update every 30 seconds
    staleTime: 15000,
  });

  // Evo status query
  const { data: evoStatus, isLoading: evoLoading, error: evoError, refetch: refetchEvo } = useQuery({
    queryKey: ['evo', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/evo/status');
      if (!response.ok) {
        throw new Error('Failed to fetch evo status');
      }
      return response.json();
    },
    refetchInterval: 30000, // Update every 30 seconds
    staleTime: 15000,
  });

  // Evo candidates query
  const { data: evoCandidates, isLoading: candidatesLoading, refetch: refetchCandidates } = useQuery({
    queryKey: ['evo', 'candidates'],
    queryFn: async () => {
      const response = await fetch('/api/evo/candidates?limit=20');
      if (!response.ok) {
        throw new Error('Failed to fetch evo candidates');
      }
      return response.json();
    },
    refetchInterval: 60000, // Update every minute
    staleTime: 30000,
  });

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'R1': return 'bg-blue-500/10 text-blue-600 border-blue-500/20';
      case 'R2': return 'bg-green-500/10 text-green-600 border-green-500/20';
      case 'R3': return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      case 'LIVE': return 'bg-red-500/10 text-red-600 border-red-500/20';
      default: return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
    }
  };

  const getDecisionIcon = (decision: string) => {
    switch (decision) {
      case 'promote': return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'demote': return <TrendingDown className="w-4 h-4 text-red-600" />;
      default: return <Activity className="w-4 h-4 text-blue-600" />;
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffMs = now.getTime() - time.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    return `${diffHours}h ago`;
  };

  const [schedulingValidation, setSchedulingValidation] = useState(false);

  const handleScheduleValidation = async (configId: string, days: number = 14) => {
    setSchedulingValidation(true);
    try {
      const response = await fetch('/api/evo/schedule-paper-validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ config_id: configId, days }),
      });

      if (!response.ok) {
        throw new Error('Failed to schedule validation');
      }

      const result = await response.json();
      console.log('Validation scheduled:', result);
      // Could add toast notification here
      refetchCandidates(); // Refresh candidates list
    } catch (error) {
      console.error('Error scheduling validation:', error);
      // Could add error toast here
    } finally {
      setSchedulingValidation(false);
    }
  };

  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

  const isLoading = brainLoading || evoLoading || candidatesLoading;
  const hasError = brainError || evoError;

  if (isLoading) {
    return (
      <div className="border rounded-2xl p-4">
        <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Brain & EvoFlow
          </h3>
        </div>
        <div className="mt-4 space-y-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="animate-pulse">
              <div className="h-16 bg-muted rounded"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (hasError) {
    return (
      <div className="border rounded-2xl p-4">
        <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Brain & EvoFlow
          </h3>
          <div className="flex items-center gap-1 text-xs text-red-600">
            <AlertTriangle className="w-3 h-3" />
            Connection Error
          </div>
        </div>
        <div className="mt-4 text-center py-8 text-muted-foreground">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Brain/EvoFlow data unavailable</p>
          <p className="text-xs mt-1">Check API connections</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`border rounded-2xl p-4 w-full max-w-full ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Brain & EvoFlow
        </h3>
        <div className="flex items-center gap-2">
          <div className="text-xs text-muted-foreground">
            Live pilot + Offline R&D
          </div>
          <button
            onClick={() => { refetchBrain(); refetchEvo(); refetchCandidates(); }}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Two-panel layout */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Brain Panel (Live Pilot) */}
        <div className="border rounded-lg p-4 bg-card">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-4 h-4 text-blue-600" />
            <h4 className="font-medium">Brain (Live Pilot)</h4>
            <div className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-full ${brainStatus?.running ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-xs text-muted-foreground">
                {brainStatus?.running ? 'Running' : 'Stopped'}
              </span>
            </div>
          </div>

          {brainStatus && (
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mode:</span>
                <span className="font-medium">{brainStatus.mode}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Tick Rate:</span>
                <span>{brainStatus.tick_ms}ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">PF (after costs):</span>
                <span className={brainStatus.recent_pf_after_costs >= 1 ? 'text-green-600' : 'text-red-600'}>
                  {formatPercent(brainStatus.recent_pf_after_costs)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sharpe 30d:</span>
                <span className={brainStatus.sharpe_30d >= 0.5 ? 'text-green-600' : brainStatus.sharpe_30d >= 0 ? 'text-yellow-600' : 'text-red-600'}>
                  {brainStatus.sharpe_30d.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sharpe 90d:</span>
                <span className={brainStatus.sharpe_90d >= 0.5 ? 'text-green-600' : brainStatus.sharpe_90d >= 0 ? 'text-yellow-600' : 'text-red-600'}>
                  {brainStatus.sharpe_90d.toFixed(2)}
                </span>
              </div>
              {brainStatus.breaker && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Circuit Breaker:</span>
                  <span className="text-red-600 text-xs">{brainStatus.breaker}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* EvoFlow Panel (Offline R&D) */}
        <div className="border rounded-lg p-4 bg-card">
          <div className="flex items-center gap-2 mb-3">
            <Dna className="w-4 h-4 text-purple-600" />
            <h4 className="font-medium">EvoFlow (Offline R&D)</h4>
            <div className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-full ${evoStatus?.running ? 'bg-green-500' : 'bg-yellow-500'}`} />
              <span className="text-xs text-muted-foreground">
                {evoStatus?.running ? 'Running' : 'Idle'}
              </span>
            </div>
          </div>

          {evoStatus && (
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Generation:</span>
                <span className="font-medium">{evoStatus.generation}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Population:</span>
                <span>{evoStatus.population}</span>
              </div>
              {evoStatus.best && (
                <>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Best Config:</span>
                    <span className="font-medium text-xs">{evoStatus.best.config_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Best PF:</span>
                    <span className={evoStatus.best.metrics.pf_after_costs >= 1 ? 'text-green-600' : 'text-red-600'}>
                      {formatPercent(evoStatus.best.metrics.pf_after_costs)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Best Sharpe:</span>
                    <span className={evoStatus.best.metrics.sharpe >= 0.5 ? 'text-green-600' : 'text-yellow-600'}>
                      {evoStatus.best.metrics.sharpe.toFixed(2)}
                    </span>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* EvoFlow Candidates Table */}
      {evoCandidates && evoCandidates.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center gap-2 mb-3">
            <Trophy className="w-4 h-4" />
            <span className="text-sm font-medium">Recent Candidates</span>
            <Badge variant="outline" className="text-xs">
              Ready for Paper Validation
            </Badge>
          </div>

          <div className="border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left py-2 px-3">Strategy</th>
                  <th className="text-left py-2 px-3">Config ID</th>
                  <th className="text-right py-2 px-3">PF (after costs)</th>
                  <th className="text-right py-2 px-3">Sharpe</th>
                  <th className="text-right py-2 px-3">Trades</th>
                  <th className="text-center py-2 px-3">Action</th>
                </tr>
              </thead>
              <tbody>
                {evoCandidates.map((candidate: any, index: number) => (
                  <tr key={candidate.config_id} className="border-t">
                    <td className="py-2 px-3 font-medium">{candidate.strategy_id}</td>
                    <td className="py-2 px-3 text-xs font-mono">{candidate.config_id}</td>
                    <td className={`py-2 px-3 text-right font-mono ${candidate.backtest.pf_after_costs >= 1 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatPercent(candidate.backtest.pf_after_costs)}
                    </td>
                    <td className={`py-2 px-3 text-right font-mono ${candidate.backtest.sharpe >= 0.5 ? 'text-green-600' : 'text-yellow-600'}`}>
                      {candidate.backtest.sharpe.toFixed(2)}
                    </td>
                    <td className="py-2 px-3 text-right font-mono">{candidate.backtest.trades}</td>
                    <td className="py-2 px-3 text-center">
                      <button
                        onClick={() => handleScheduleValidation(candidate.config_id)}
                        disabled={schedulingValidation}
                        className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded disabled:opacity-50"
                      >
                        {schedulingValidation ? '...' : 'Validate'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
};

export default BrainEvoFlow;
