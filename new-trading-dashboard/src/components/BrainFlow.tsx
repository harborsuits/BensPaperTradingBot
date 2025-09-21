import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Brain,
  CheckCircle,
  XCircle,
  Minus,
  AlertTriangle,
  ExternalLink,
  Clock,
  Filter
} from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

interface BrainFlowTick {
  symbol: string;
  ts: string;
  stages: {
    ingest: { ok: boolean; quote_age_s?: number };
    context: { ok: boolean; vol_rank?: number; atr?: number };
    candidates: { ok: boolean; count?: number; winner?: { strategy_id: string; confidence: number } };
    gates: { ok: boolean; passed?: string[]; rejected?: string[] };
    plan: { ok: boolean; reason?: string };
    route: { ok: boolean; skipped?: boolean };
    manage: { ok: boolean; skipped?: boolean };
    learn: { ok: boolean; skipped?: boolean };
  };
  mode: string;
  trace_id: string;
}

interface BrainFlowProps {
  className?: string;
}

const stageOrder = ['ingest', 'context', 'candidates', 'gates', 'plan', 'route', 'manage', 'learn'];

const stageLabels = {
  ingest: 'INGEST',
  context: 'CONTEXT',
  candidates: 'CANDIDATES',
  gates: 'GATES',
  plan: 'PLAN',
  route: 'ROUTE',
  manage: 'MANAGE',
  learn: 'LEARN'
};

export const BrainFlow: React.FC<BrainFlowProps> = ({ className = '' }) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  const { data: brainFlowData, isLoading, error, refetch } = useQuery<BrainFlowTick[]>({
    queryKey: ['brain', 'flow', 'recent'],
    queryFn: async () => {
      const response = await fetch('/api/brain/flow/recent?limit=100');
      if (!response.ok) {
        throw new Error('Failed to fetch brain flow data');
      }
      return response.json();
    },
    refetchInterval: 15000, // Update every 15 seconds
    staleTime: 5000,
  });

  const getStageIcon = (stage: any) => {
    if (stage.ok) return <CheckCircle className="w-4 h-4 text-green-600" />;
    if (stage.skipped) return <Minus className="w-4 h-4 text-gray-500" />;
    return <XCircle className="w-4 h-4 text-red-600" />;
  };

  const getStageColor = (stage: any) => {
    if (stage.ok) return 'bg-green-500/10 text-green-700 border-green-500/20';
    if (stage.skipped) return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
    return 'bg-red-500/10 text-red-700 border-red-500/20';
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

  // Get unique symbols for filtering
  const symbols = brainFlowData ? Array.from(new Set(brainFlowData.map(tick => tick.symbol))) : [];

  // Filter data by selected symbol
  const filteredData = selectedSymbol
    ? brainFlowData?.filter(tick => tick.symbol === selectedSymbol) || []
    : brainFlowData || [];

  if (collapsed) {
    return (
      <div className={`border rounded-2xl p-4 ${className}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            <span className="font-semibold">Brain Flow</span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCollapsed(false)}
            className="text-xs"
          >
            Expand
          </Button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className={`border rounded-2xl p-4 ${className}`}>
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5" />
          <span className="font-semibold">Brain Flow</span>
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="animate-pulse">
              <div className="h-16 bg-muted rounded"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !brainFlowData) {
    return (
      <div className={`border rounded-2xl p-4 ${className}`}>
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5" />
          <span className="font-semibold">Brain Flow</span>
        </div>
        <div className="text-center py-8 text-muted-foreground">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Brain flow data unavailable</p>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            className="mt-2"
          >
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className={`border rounded-2xl p-4 w-full max-w-full overflow-x-auto ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          <span className="font-semibold">Brain Flow</span>
          <span className="text-sm text-muted-foreground">
            {filteredData.length} recent ticks
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCollapsed(true)}
            className="text-xs"
          >
            Collapse
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            className="text-xs"
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* Symbol filter */}
      {symbols.length > 1 && (
        <div className="flex items-center gap-2 mb-4">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <div className="flex flex-wrap gap-2">
            <Badge
              className="cursor-pointer bg-blue-800 hover:bg-blue-700 text-white"
              variant={selectedSymbol === null ? "default" : "outline"}
              onClick={() => setSelectedSymbol(null)}
            >
              All
            </Badge>
            {symbols.map(symbol => (
              <Badge
                key={symbol}
                className="cursor-pointer bg-blue-800 hover:bg-blue-700 text-white"
                variant={selectedSymbol === symbol ? "default" : "outline"}
                onClick={() => setSelectedSymbol(symbol)}
              >
                {symbol}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Pipeline flow visualization */}
      <div className="space-y-4">
        {filteredData.slice(0, 10).map((tick, index) => (
          <div key={tick.trace_id} className="border rounded-lg p-4 bg-card">
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="font-medium">{tick.symbol}</span>
                <Badge variant="outline" className="text-xs">
                  {tick.mode}
                </Badge>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Clock className="w-3 h-3" />
                  {formatTimeAgo(tick.ts)}
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs h-6 px-2"
                onClick={() => {
                  // Open evidence for this tick
                  window.open(`/api/brain/flow/recent?symbol=${tick.symbol}&ts=${tick.ts}`, '_blank');
                }}
              >
                <ExternalLink className="w-3 h-3 mr-1" />
                Evidence
              </Button>
            </div>

            {/* Pipeline stages */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">
              {stageOrder.map(stageKey => {
                const stage = tick.stages[stageKey];
                const stageName = stageLabels[stageKey];

                return (
                  <div
                    key={stageKey}
                    className={`border rounded-md p-2 text-center ${getStageColor(stage)}`}
                  >
                    <div className="flex items-center justify-center gap-1 mb-1">
                      {getStageIcon(stage)}
                      <span className="text-xs font-medium">{stageName}</span>
                    </div>
                    {stageKey === 'candidates' && stage.count && (
                      <div className="text-xs text-muted-foreground">
                        {stage.count} candidates
                      </div>
                    )}
                    {stageKey === 'gates' && stage.passed && (
                      <div className="text-xs text-muted-foreground">
                        {stage.passed.length}/{stage.passed.length + (stage.rejected?.length || 0)} passed
                      </div>
                    )}
                    {!stage.ok && !stage.skipped && stage.reason && (
                      <div className="text-xs text-muted-foreground truncate" title={stage.reason}>
                        {stage.reason.length > 15 ? `${stage.reason.substring(0, 15)}...` : stage.reason}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {filteredData.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No brain flow data found</p>
          <p className="text-xs mt-1">
            {selectedSymbol ? `No data for ${selectedSymbol}` : 'Check if the brain loop is running'}
          </p>
        </div>
      )}
    </div>
  );
};

export default BrainFlow;
