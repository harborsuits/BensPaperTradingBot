import { useState } from "react";
import { useDecisions, useDecisionsStream } from "@/hooks/useDecisions";
import { Decision } from "@/schemas/trading";
import { TrendingUp, TrendingDown, Zap, Target, AlertCircle } from "lucide-react";

export default function RecentTradeDecisionsCard() {
  const { data: decisions, isLoading } = useDecisions(10);
  const streamingDecisions = useDecisionsStream();

  // Combine REST API data with streaming data
  const allDecisions = streamingDecisions.length > 0 ? streamingDecisions : (decisions || []);
  const displayDecisions = allDecisions.slice(0, 5); // Show top 5

  const formatTime = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  const getDecisionIcon = (side: string) => {
    return side === 'buy' ? (
      <TrendingUp className="w-4 h-4 text-green-500" />
    ) : (
      <TrendingDown className="w-4 h-4 text-red-500" />
    );
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-50';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  if (isLoading && displayDecisions.length === 0) {
    return (
      <div className="border rounded-2xl p-4">
        <div className="flex items-baseline justify-between">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Recent Trade Decisions
          </h3>
        </div>
        <div className="mt-4 space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="animate-pulse">
              <div className="h-16 bg-gray-200 rounded-xl"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="border rounded-2xl p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Zap className="w-5 h-5" />
          Recent Trade Decisions
        </h3>
        <div className="flex items-center gap-2">
          {streamingDecisions.length > 0 && (
            <div className="flex items-center gap-1 text-xs text-green-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              Live
            </div>
          )}
          <span className="text-xs text-muted-foreground">
            {displayDecisions.length} recent
          </span>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {displayDecisions.map((decision: Decision) => (
          <div
            key={decision.id}
            className="border rounded-xl p-3 hover:bg-muted/50 transition-colors"
          >
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getDecisionIcon(decision.side)}
                <span className="font-semibold">{decision.symbol}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  decision.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {decision.side.toUpperCase()}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-xs px-2 py-1 rounded ${getConfidenceColor(decision.confidence)}`}>
                  {(decision.confidence * 100).toFixed(0)}%
                </span>
                <span className="text-xs text-muted-foreground">
                  {formatTime(decision.ts)}
                </span>
              </div>
            </div>

            {/* Score and Strategy */}
            <div className="flex items-center justify-between mt-2">
              <div className="flex items-center gap-2">
                <Target className="w-3 h-3 text-muted-foreground" />
                <span className="text-sm font-medium">
                  Score: {decision.score.toFixed(1)}
                </span>
              </div>
              <span className="text-xs text-muted-foreground">
                {decision.strategy}
              </span>
            </div>

            {/* Thesis */}
            <div className="mt-2 text-xs text-muted-foreground line-clamp-2">
              {decision.thesis}
            </div>

            {/* Risk and Order Details */}
            <div className="flex items-center justify-between mt-2 pt-2 border-t">
              <div className="text-xs text-muted-foreground">
                Qty: {decision.order.qty} • Risk: ${(decision.risk.max_loss / decision.order.qty).toFixed(0)}/share
              </div>
              <div className="flex items-center gap-1 text-xs">
                {decision.checks.slippage_ok ? (
                  <span className="text-green-600">✓ Slippage OK</span>
                ) : (
                  <span className="text-yellow-600">⚠ High Slippage</span>
                )}
              </div>
            </div>
          </div>
        ))}

        {displayDecisions.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No recent decisions</p>
            <p className="text-xs mt-1">Brain is analyzing market signals...</p>
          </div>
        )}
      </div>
    </div>
  );
}


