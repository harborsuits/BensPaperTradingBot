import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Activity, Zap, Clock } from 'lucide-react';
import { j } from '@/lib/api';
import { EvidenceButton } from '@/components/EvidenceButton';
import { useCardHeartbeat } from '@/hooks/useCardHeartbeat';

interface PipelineSummaryData {
  counts: {
    gates_passed: number;
    gates_failed: number;
    total: number;
  };
  by_mode: {
    discovery: number;
    shadow: number;
    live: number;
  };
  latency_ms: {
    p50: number | null;
    p95: number | null;
    avg: number | null;
  };
  asOf: string;
}

export function PipelineHealthCard() {
  useCardHeartbeat('dashboard:pipeline-health');

  const { data, isLoading, error } = useQuery({
    queryKey: ['pipeline:summary', '15m'],
    queryFn: () => j<PipelineSummaryData>('/api/brain/flow/summary?window=15m'),
    refetchInterval: 5000
  });

  const totalProcessed = data?.counts.total ?? 0;
  const successRate = totalProcessed > 0
    ? ((data?.counts.gates_passed ?? 0) / totalProcessed) * 100
    : 0;

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-medium flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400" />
          Pipeline Health
        </h3>
        <EvidenceButton
          endpoint="/api/brain/flow/summary?window=15m"
          title="Pipeline Health Evidence"
        />
      </div>

      {isLoading ? (
        <div className="animate-pulse">
          <div className="h-4 bg-gray-700 rounded mb-2"></div>
          <div className="h-4 bg-gray-700 rounded w-3/4"></div>
        </div>
      ) : error ? (
        <div className="text-red-400 text-sm">
          Failed to load pipeline data
        </div>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">
                {data?.counts.gates_passed ?? 0}
              </div>
              <div className="text-xs text-gray-400">Passed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-400">
                {data?.counts.gates_failed ?? 0}
              </div>
              <div className="text-xs text-gray-400">Failed</div>
            </div>
          </div>

          <div className="text-center">
            <div className="text-lg font-semibold text-blue-400">
              {successRate.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400">Success Rate</div>
          </div>

          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3 text-yellow-400" />
              <span className="text-gray-300">
                {data?.by_mode.live ?? 0} Live
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3 text-blue-400" />
              <span className="text-gray-300">
                p50: {data?.latency_ms.p50 ?? '—'}ms
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
