import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';

interface BrainActivityItem {
  ts: string;
  symbol: string;
  final_score: number;
  confidence: number;
  regime: string;
  news_delta: number;
  latency_ms: number;
  fallback: boolean;
  decision_id: string;
  experts: Array<{ name: string; score: number; conf: number }>;
  gates: {
    earnings_window: boolean;
    dd_ok: boolean;
    fresh_ok: boolean;
  };
}

const BrainScoringActivityCard: React.FC = () => {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['brain-activity'],
    queryFn: async () => {
      const response = await fetch('/api/brain/activity?limit=50');
      if (!response.ok) throw new Error('Failed to fetch brain activity');
      return response.json();
    },
    refetchInterval: 5000, // Refresh every 5 seconds
    staleTime: 2000,
  });

  const activities: BrainActivityItem[] = data?.items || [];

  const toggleRow = (decisionId: string) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(decisionId)) {
      newExpanded.delete(decisionId);
    } else {
      newExpanded.add(decisionId);
    }
    setExpandedRows(newExpanded);
  };

  const formatTime = (ts: string) => {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'bull_strong': return 'bg-green-100 text-green-800';
      case 'bull_medium': return 'bg-green-50 text-green-700';
      case 'neutral': return 'bg-gray-100 text-gray-800';
      case 'bear_medium': return 'bg-orange-50 text-orange-700';
      case 'bear_strong': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (isLoading) {
    return (
      <div className="border rounded-xl p-4 w-full max-w-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">🧠 Brain Scoring Activity</h3>
          <div className="text-xs text-muted-foreground">Loading...</div>
        </div>
        <div className="animate-pulse space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-gray-200 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="border rounded-xl p-4 w-full max-w-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">🧠 Brain Scoring Activity</h3>
          <div className="text-xs text-red-500">Error loading data</div>
        </div>
        <div className="text-center py-8 text-muted-foreground">
          <AlertTriangle className="mx-auto h-8 w-8 mb-2" />
          <p>Unable to load brain activity data</p>
          <button
            onClick={() => refetch()}
            className="mt-2 px-3 py-1 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="border rounded-xl p-4 w-full max-w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">🧠 Brain Scoring Activity</h3>
        <div className="text-xs text-muted-foreground">
          {activities.length} recent scores • auto-refresh
        </div>
      </div>

      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {activities.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Clock className="mx-auto h-8 w-8 mb-2" />
            <p>No recent brain activity</p>
            <p className="text-sm mt-1">Brain scoring will appear here as decisions are made</p>
          </div>
        ) : (
          activities.map((activity) => (
            <div key={activity.decision_id} className="border rounded-lg overflow-hidden">
              {/* Main Row */}
              <div
                className="flex items-center justify-between p-3 bg-card hover:bg-card/80 cursor-pointer"
                onClick={() => toggleRow(activity.decision_id)}
              >
                <div className="flex items-center gap-4">
                  {expandedRows.has(activity.decision_id) ? (
                    <ChevronDown size={16} className="text-muted-foreground" />
                  ) : (
                    <ChevronRight size={16} className="text-muted-foreground" />
                  )}

                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-lg">{activity.symbol}</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getRegimeColor(activity.regime)}`}>
                      {activity.regime.replace('_', ' ')}
                    </span>
                    {activity.fallback && (
                      <span className="px-2 py-1 rounded text-xs font-medium bg-red-100 text-red-800">
                        FALLBACK
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-4 text-sm">
                  <div className="text-center">
                    <div className={`font-bold text-lg ${getScoreColor(activity.final_score)}`}>
                      {(activity.final_score * 100).toFixed(0)}
                    </div>
                    <div className="text-xs text-muted-foreground">score</div>
                  </div>

                  <div className="text-center">
                    <div className="font-medium">{(activity.confidence * 100).toFixed(0)}%</div>
                    <div className="text-xs text-muted-foreground">conf</div>
                  </div>

                  <div className="text-center">
                    <div className="font-medium">{activity.news_delta.toFixed(2)}</div>
                    <div className="text-xs text-muted-foreground">news Δ</div>
                  </div>

                  <div className="text-center">
                    <div className="font-medium">{activity.latency_ms}ms</div>
                    <div className="text-xs text-muted-foreground">latency</div>
                  </div>

                  <div className="text-muted-foreground text-sm">
                    {formatTime(activity.ts)}
                  </div>
                </div>
              </div>

              {/* Expanded Details */}
              {expandedRows.has(activity.decision_id) && (
                <div className="border-t bg-muted/30 p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Expert Scores */}
                    <div>
                      <h4 className="font-medium mb-2">Expert Scores</h4>
                      <div className="space-y-1">
                        {activity.experts.map((expert, idx) => (
                          <div key={idx} className="flex justify-between text-sm">
                            <span className="capitalize">{expert.name}</span>
                            <span className="font-medium">
                              {(expert.score * 100).toFixed(0)} ({(expert.conf * 100).toFixed(0)}%)
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Gate Status */}
                    <div>
                      <h4 className="font-medium mb-2">Gate Status</h4>
                      <div className="space-y-1">
                        {Object.entries(activity.gates).map(([gate, status]) => (
                          <div key={gate} className="flex items-center justify-between text-sm">
                            <span className="capitalize">{gate.replace('_', ' ')}</span>
                            {status ? (
                              <CheckCircle size={14} className="text-green-600" />
                            ) : (
                              <AlertTriangle size={14} className="text-red-600" />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 pt-4 border-t">
                    <div className="text-xs text-muted-foreground">
                      Decision ID: {activity.decision_id}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default BrainScoringActivityCard;
