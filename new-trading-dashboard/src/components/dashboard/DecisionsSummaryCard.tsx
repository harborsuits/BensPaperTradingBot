import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Brain, TrendingUp, Target } from 'lucide-react';
import { j } from '@/lib/api';
import { EvidenceButton } from '@/components/EvidenceButton';
import { useCardHeartbeat } from '@/hooks/useCardHeartbeat';

interface DecisionsSummaryData {
  window: string;
  total_decisions: number;
  proposals_per_min: number;
  unique_symbols: number;
  by_stage: {
    proposed: number;
    intents: number;
    executed: number;
  };
  avg_confidence: number;
  last_ts: string | null;
  asOf: string;
}

export function DecisionsSummaryCard() {
  useCardHeartbeat('dashboard:decisions-summary');

  const { data, isLoading, error } = useQuery({
    queryKey: ['decisions:summary', '15m'],
    queryFn: () => j<DecisionsSummaryData>('/api/decisions/summary?window=15m'),
    refetchInterval: 5000
  });

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-medium flex items-center gap-2">
          <Brain className="w-4 h-4 text-purple-400" />
          Decision Flow
        </h3>
        <EvidenceButton
          endpoint="/api/decisions/summary?window=15m"
          title="Decision Flow Evidence"
        />
      </div>

      {isLoading ? (
        <div className="animate-pulse">
          <div className="h-4 bg-gray-700 rounded mb-2"></div>
          <div className="h-4 bg-gray-700 rounded w-3/4"></div>
        </div>
      ) : error ? (
        <div className="text-red-400 text-sm">
          Failed to load decision data
        </div>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <div className="text-xl font-bold text-blue-400">
                {data?.by_stage.proposed ?? 0}
              </div>
              <div className="text-xs text-gray-400">Proposed</div>
            </div>
            <div>
              <div className="text-xl font-bold text-yellow-400">
                {data?.by_stage.intents ?? 0}
              </div>
              <div className="text-xs text-gray-400">Intents</div>
            </div>
            <div>
              <div className="text-xl font-bold text-green-400">
                {data?.by_stage.executed ?? 0}
              </div>
              <div className="text-xs text-gray-400">Executed</div>
            </div>
          </div>

          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1">
              <Target className="w-3 h-3 text-purple-400" />
              <span className="text-gray-300">
                {data?.unique_symbols ?? 0} Symbols
              </span>
            </div>
            <div className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3 text-green-400" />
              <span className="text-gray-300">
                {data?.proposals_per_min?.toFixed(1) ?? '0.0'}/min
              </span>
            </div>
          </div>

          {data?.avg_confidence && (
            <div className="text-center">
              <div className="text-sm font-medium text-gray-300">
                Avg Confidence: {(data.avg_confidence * 100).toFixed(0)}%
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
