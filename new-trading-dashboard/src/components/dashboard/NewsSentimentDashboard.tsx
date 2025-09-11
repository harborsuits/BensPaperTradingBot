import React from 'react';
import { TrendingUp, TrendingDown, Clock, BarChart3, AlertTriangle } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';

interface OutletData {
  count: number;
  avg_conf: number;
  impact: number;
}

interface TopImpact {
  symbol: string;
  delta: number;
  event: string;
  conf: number;
}

interface NewsInsightsData {
  summary: {
    market_sentiment: string;
    last_event_s: number;
    queue_lag_ms: number;
  };
  by_outlet: Record<string, OutletData>;
  top_impacts: TopImpact[];
}

const NewsSentimentDashboard: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['news-insights'],
    queryFn: async () => {
      const response = await fetch('/api/news/insights?window=24');
      if (!response.ok) throw new Error('Failed to fetch news insights');
      return response.json() as Promise<NewsInsightsData>;
    },
    refetchInterval: 10000, // Refresh every 10 seconds
    staleTime: 5000,
  });

  const getSentimentIcon = (sentiment: string) => {
    if (sentiment.includes('bullish')) return <TrendingUp className="w-4 h-4 text-green-500" />;
    if (sentiment.includes('bearish')) return <TrendingDown className="w-4 h-4 text-red-500" />;
    return <BarChart3 className="w-4 h-4 text-gray-500" />;
  };

  const getSentimentColor = (sentiment: string) => {
    if (sentiment.includes('bullish')) return 'text-green-600 bg-green-50 border-green-200';
    if (sentiment.includes('bearish')) return 'text-red-600 bg-red-50 border-red-200';
    return 'text-gray-600 bg-gray-50 border-gray-200';
  };

  const formatTimeAgo = (seconds: number) => {
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  const formatLag = (ms: number) => {
    return `${ms.toFixed(0)}ms`;
  };

  if (isLoading) {
    return (
      <div className="border rounded-xl p-4 w-full max-w-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">📰 News Sentiment Dashboard</h3>
          <div className="text-xs text-muted-foreground">Loading...</div>
        </div>
        <div className="animate-pulse space-y-4">
          <div className="h-20 bg-gray-200 rounded"></div>
          <div className="grid grid-cols-2 gap-4">
            <div className="h-32 bg-gray-200 rounded"></div>
            <div className="h-32 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="border rounded-xl p-4 w-full max-w-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">📰 News Sentiment Dashboard</h3>
          <div className="text-xs text-red-500">Error loading data</div>
        </div>
        <div className="text-center py-8 text-muted-foreground">
          <AlertTriangle className="mx-auto h-8 w-8 mb-2" />
          <p>Unable to load news sentiment data</p>
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

  const summary = data?.summary;
  const outlets = data?.by_outlet || {};
  const topImpacts = data?.top_impacts || [];

  const sortedOutlets = Object.entries(outlets)
    .sort(([, a], [, b]) => b.impact - a.impact)
    .slice(0, 8);

  return (
    <div className="border rounded-xl p-4 w-full max-w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">📰 News Sentiment Dashboard</h3>
        <div className="text-xs text-muted-foreground">Real-time • 24h window</div>
      </div>

      {/* Market Sentiment Summary */}
      <div className="mb-6">
        <div className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg border ${getSentimentColor(summary?.market_sentiment || 'neutral')}`}>
          {getSentimentIcon(summary?.market_sentiment || 'neutral')}
          <span className="font-medium capitalize">
            {summary?.market_sentiment?.replace('_', ' ') || 'Neutral'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Freshness & Performance */}
        <div className="space-y-4">
          <div className="border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Clock className="w-4 h-4 text-muted-foreground" />
              <h4 className="font-medium">Data Freshness</h4>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Last Event</span>
                <span className="text-sm font-medium">
                  {formatTimeAgo(summary?.last_event_s || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Queue Lag</span>
                <span className="text-sm font-medium">
                  {formatLag(summary?.queue_lag_ms || 0)}
                </span>
              </div>
            </div>
          </div>

          {/* Top Sentiment Impacts */}
          <div className="border rounded-lg p-4">
            <h4 className="font-medium mb-3">Top Sentiment Impacts</h4>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {topImpacts.length > 0 ? (
                topImpacts.map((impact, idx) => (
                  <div key={idx} className="flex items-center justify-between py-2 border-b border-border last:border-b-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{impact.symbol}</span>
                      <span className="text-xs text-muted-foreground">{impact.event}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`text-sm font-medium ${impact.delta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {(impact.delta * 100).toFixed(1)}%
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {(impact.conf * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-4 text-muted-foreground text-sm">
                  No significant sentiment impacts in the last 24h
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Outlet Analysis */}
        <div className="border rounded-lg p-4">
          <h4 className="font-medium mb-3">Outlet Impact Analysis</h4>
          <div className="space-y-3 max-h-60 overflow-y-auto">
            {sortedOutlets.length > 0 ? (
              sortedOutlets.map(([outlet, data]) => (
                <div key={outlet} className="border rounded p-3 bg-muted/30">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">{outlet}</span>
                    <span className="text-xs text-muted-foreground">
                      {data.count} articles
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="ml-1 font-medium">
                        {(data.avg_conf * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Impact:</span>
                      <span className="ml-1 font-medium">
                        {(data.impact * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="mt-2 bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(data.impact * 100, 100)}%` }}
                    />
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-muted-foreground text-sm">
                <BarChart3 className="mx-auto h-8 w-8 mb-2 opacity-50" />
                <p>No outlet activity in the selected window</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewsSentimentDashboard;
