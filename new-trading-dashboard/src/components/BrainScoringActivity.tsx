import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Target,
  Trophy,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw
} from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

interface ScoringCandidate {
  strategy_id: string;
  raw_score: number;
  ev_after_costs: number;
  reliability: number;
  liquidity: number;
  total: number;
  selected: boolean;
  reason: string;
}

interface ScoringActivity {
  symbol: string;
  ts: string;
  candidates: ScoringCandidate[];
  weights: {
    ev: number;
    reliability: number;
    liquidity: number;
  };
  trace_id: string;
}

interface BrainScoringActivityProps {
  symbol?: string;
  timestamp?: string;
  className?: string;
}

export const BrainScoringActivity: React.FC<BrainScoringActivityProps> = ({
  symbol: initialSymbol,
  timestamp,
  className = ''
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>(initialSymbol || 'SPY');

  const { data: scoringData, isLoading, error, refetch } = useQuery<ScoringActivity>({
    queryKey: ['brain', 'scoring', 'activity', selectedSymbol, timestamp],
    queryFn: async () => {
      const params = new URLSearchParams({
        symbol: selectedSymbol,
        ...(timestamp && { ts: timestamp })
      });
      const response = await fetch(`/api/brain/scoring/activity?${params}`);
      if (!response.ok) {
        throw new Error('Failed to fetch scoring activity');
      }
      return response.json();
    },
    enabled: !!selectedSymbol,
    refetchInterval: 30000, // Update every 30 seconds
    staleTime: 15000,
  });

  const formatScore = (score: number, decimals: number = 3) => {
    return score.toFixed(decimals);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const getScoreColor = (score: number, type: 'ev' | 'reliability' | 'liquidity') => {
    if (type === 'ev') {
      return score > 0 ? 'text-green-600' : 'text-red-600';
    }
    // For reliability and liquidity, higher is better
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getTotalScoreColor = (total: number) => {
    if (total > 0.5) return 'text-green-600';
    if (total > 0.2) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (isLoading) {
    return (
      <div className={`border rounded-2xl p-4 ${className}`}>
        <div className="flex items-center gap-2 mb-4">
          <Target className="w-5 h-5" />
          <span className="font-semibold">Brain Scoring Activity</span>
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="animate-pulse">
              <div className="h-12 bg-muted rounded"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !scoringData) {
    return (
      <div className={`border rounded-2xl p-4 ${className}`}>
        <div className="flex items-center gap-2 mb-4">
          <Target className="w-5 h-5" />
          <span className="font-semibold">Brain Scoring Activity</span>
        </div>
        <div className="text-center py-8 text-muted-foreground">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Scoring activity unavailable</p>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            className="mt-2"
          >
            <RefreshCw className="w-3 h-3 mr-1" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Sort candidates by total score descending
  const sortedCandidates = [...scoringData.candidates].sort((a, b) => b.total - a.total);

  return (
    <div className={`border rounded-2xl p-4 w-full ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5" />
          <span className="font-semibold">Brain Scoring Activity</span>
          <Badge variant="outline" className="text-xs">
            {scoringData.symbol}
          </Badge>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          className="text-xs"
        >
          <RefreshCw className="w-3 h-3 mr-1" />
          Refresh
        </Button>
      </div>

      {/* Scoring weights */}
      <div className="mb-4 p-3 bg-muted/50 rounded-lg">
        <div className="text-sm font-medium mb-2">Scoring Weights</div>
        <div className="flex gap-4 text-xs">
          <span>EV: {formatPercent(scoringData.weights.ev)}</span>
          <span>Reliability: {formatPercent(scoringData.weights.reliability)}</span>
          <span>Liquidity: {formatPercent(scoringData.weights.liquidity)}</span>
        </div>
      </div>

      {/* Scoring table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2 px-2">Strategy</th>
              <th className="text-right py-2 px-2">Raw Score</th>
              <th className="text-right py-2 px-2">EV After Costs</th>
              <th className="text-right py-2 px-2">Reliability</th>
              <th className="text-right py-2 px-2">Liquidity</th>
              <th className="text-right py-2 px-2">Total</th>
              <th className="text-center py-2 px-2">Selected?</th>
              <th className="text-left py-2 px-2">Why</th>
            </tr>
          </thead>
          <tbody>
            {sortedCandidates.map((candidate, index) => (
              <tr key={candidate.strategy_id} className={`border-b ${candidate.selected ? 'bg-blue-50/50' : ''}`}>
                <td className="py-3 px-2">
                  <div className="flex items-center gap-2">
                    {candidate.selected && <Trophy className="w-4 h-4 text-yellow-600" />}
                    <span className="font-medium">{candidate.strategy_id}</span>
                    {index === 0 && !candidate.selected && (
                      <Badge variant="outline" className="text-xs">Highest Score</Badge>
                    )}
                  </div>
                </td>
                <td className="text-right py-3 px-2 font-mono">
                  {formatScore(candidate.raw_score)}
                </td>
                <td className={`text-right py-3 px-2 font-mono ${getScoreColor(candidate.ev_after_costs, 'ev')}`}>
                  <div className="flex items-center justify-end gap-1">
                    {candidate.ev_after_costs > 0 ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : (
                      <TrendingDown className="w-3 h-3" />
                    )}
                    {formatScore(candidate.ev_after_costs, 4)}
                  </div>
                </td>
                <td className={`text-right py-3 px-2 font-mono ${getScoreColor(candidate.reliability, 'reliability')}`}>
                  {formatPercent(candidate.reliability)}
                </td>
                <td className={`text-right py-3 px-2 font-mono ${getScoreColor(candidate.liquidity, 'liquidity')}`}>
                  {formatPercent(candidate.liquidity)}
                </td>
                <td className={`text-right py-3 px-2 font-mono font-semibold ${getTotalScoreColor(candidate.total)}`}>
                  {formatScore(candidate.total)}
                </td>
                <td className="text-center py-3 px-2">
                  {candidate.selected ? (
                    <CheckCircle className="w-5 h-5 text-green-600 mx-auto" />
                  ) : (
                    <XCircle className="w-5 h-5 text-gray-400 mx-auto" />
                  )}
                </td>
                <td className="py-3 px-2 max-w-xs">
                  <div className="text-xs text-muted-foreground truncate" title={candidate.reason}>
                    {candidate.reason}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sortedCandidates.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No scoring candidates found</p>
          <p className="text-xs mt-1">Check if strategies are generating candidates</p>
        </div>
      )}

      {/* Summary */}
      {sortedCandidates.length > 0 && (
        <div className="mt-4 p-3 bg-muted/30 rounded-lg">
          <div className="text-sm">
            <strong>Summary:</strong> {sortedCandidates.filter(c => c.selected).length} of {sortedCandidates.length} candidates selected.
            {scoringData.candidates.find(c => c.selected) && (
              <span className="ml-2">
                Winner: <span className="font-medium">{scoringData.candidates.find(c => c.selected)?.strategy_id}</span>
                {' '}with total score of {formatScore(scoringData.candidates.find(c => c.selected)?.total || 0)}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default BrainScoringActivity;
