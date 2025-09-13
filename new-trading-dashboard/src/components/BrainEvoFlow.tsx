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
  const { data: tournamentData, isLoading, error } = useQuery<TournamentData>({
    queryKey: ['tournament'],
    queryFn: async () => {
      const response = await fetch('/api/live/tournament');
      if (!response.ok) {
        throw new Error('Failed to fetch tournament data');
      }
      return response.json();
    },
    refetchInterval: 15000, // Update every 15 seconds
    staleTime: 5000,
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

  if (isLoading) {
    return (
      <div className="border rounded-2xl p-4">
        <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Brain & Evo Flow
          </h3>
        </div>
        <div className="mt-4 space-y-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="animate-pulse">
              <div className="h-12 bg-muted rounded"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !tournamentData) {
    return (
      <div className="border rounded-2xl p-4">
        <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Brain & Evo Flow
          </h3>
          <div className="flex items-center gap-1 text-xs text-red-600">
            <AlertTriangle className="w-3 h-3" />
            Disconnected
          </div>
        </div>
        <div className="mt-4 text-center py-8 text-muted-foreground">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Tournament data unavailable</p>
          <p className="text-xs mt-1">Strategy manager connection required</p>
        </div>
      </div>
    );
  }

  const totalActive = tournamentData.rounds.reduce((sum, r) => sum + r.active_strategies, 0);

  return (
    <div className={`border rounded-2xl p-4 w-full max-w-full overflow-x-auto ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Brain & Evo Flow
        </h3>
        <div className="text-xs text-muted-foreground">
          {totalActive} active • Gen {tournamentData.current_generation} • auto-refreshing
        </div>
      </div>

      {/* Tournament Ladder */}
      <div className="mt-4">
        <div className="flex items-center gap-2 mb-3">
          <Trophy className="w-4 h-4" />
          <span className="text-sm font-medium">Tournament Ladder</span>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {tournamentData.rounds.map((round) => (
            <div key={round.stage} className="border rounded-xl p-3 bg-card">
              <div className="flex items-center justify-between mb-2">
                <div className={`px-2 py-1 rounded text-xs font-medium border ${getStageColor(round.stage)}`}>
                  {round.stage}
                </div>
                <span className="text-lg font-bold">{round.active_strategies}</span>
              </div>

              <h4 className="font-medium text-sm mb-2">{round.name}</h4>

              <div className="space-y-1 text-xs text-muted-foreground">
                {round.criteria.minSharpe && (
                  <div>Sharpe ≥ {round.criteria.minSharpe}</div>
                )}
                {round.criteria.minPf && (
                  <div>P/F ≥ {round.criteria.minPf}</div>
                )}
                {round.criteria.maxDd && (
                  <div>DD ≤ {(round.criteria.maxDd * 100).toFixed(1)}%</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Tournament Decisions */}
      {tournamentData.recent_decisions && tournamentData.recent_decisions.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4" />
            <span className="text-sm font-medium">Recent Decisions</span>
          </div>

          <div className="space-y-2 max-h-40 overflow-y-auto scrollbar-thin scrollbar-thumb-muted scrollbar-track-muted/30">
            {tournamentData.recent_decisions.slice(0, 5).map((decision, index) => (
              <div key={index} className="border rounded-lg p-2 bg-card/50">
                <div className="flex items-center gap-2">
                  {getDecisionIcon(decision.decision)}
                  <div className="flex-grow min-w-0">
                    <p className="text-sm font-medium truncate">
                      {decision.strategyId}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {decision.fromStage} → {decision.toStage} • {decision.reason}
                    </p>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {formatTimeAgo(decision.timestamp)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Evolution Stats */}
      <div className="mt-6">
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 className="w-4 h-4" />
          <span className="text-sm font-medium">Evolution Stats</span>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="border rounded-xl p-3 bg-card text-center">
            <div className="text-xl font-bold text-green-600">
              {tournamentData.stats.totalPromotions}
            </div>
            <div className="text-xs text-muted-foreground">Promotions</div>
          </div>

          <div className="border rounded-xl p-3 bg-card text-center">
            <div className="text-xl font-bold text-red-600">
              {tournamentData.stats.totalDemotions}
            </div>
            <div className="text-xs text-muted-foreground">Demotions</div>
          </div>

          <div className="border rounded-xl p-3 bg-card text-center">
            <div className="text-xl font-bold text-blue-600">
              {totalActive}
            </div>
            <div className="text-xs text-muted-foreground">Active</div>
          </div>

          <div className="border rounded-xl p-3 bg-card text-center">
            <div className="text-xl font-bold text-purple-600">
              {tournamentData.current_generation}
            </div>
            <div className="text-xs text-muted-foreground">Generation</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BrainEvoFlow;
