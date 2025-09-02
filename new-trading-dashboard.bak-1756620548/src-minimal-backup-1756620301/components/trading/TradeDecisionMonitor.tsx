import React, { useState, useEffect } from 'react';
import { decisionApi } from '@/services/api';
import { TradeCandidate } from '@/types/api.types';
import { useTradeDecisionUpdates } from '@/hooks/useWebSocketSubscriptions';
import { FiArrowUp, FiArrowDown, FiCheckCircle, FiXCircle } from 'react-icons/fi';

interface TradeDecisionMonitorProps {
  maxDecisions?: number;
}

type DecisionCycle = {
  id: string;
  timestamp: string;
  market_conditions: {
    regime: string;
    volatility: number;
  };
  candidates: TradeCandidate[];
  executed_trades: string[];
  active_strategies: string[];
};

const TradeDecisionMonitor: React.FC<TradeDecisionMonitorProps> = ({
  maxDecisions = 10
}) => {
  const [decisionCycle, setDecisionCycle] = useState<DecisionCycle | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch initial decision data
  useEffect(() => {
    const fetchLatestDecisions = async () => {
      setLoading(true);
      try {
        const response = await decisionApi.getLatestDecisions();
        if (response.success && response.data) {
          setDecisionCycle(response.data);
        } else {
          setError(response.error || 'Failed to fetch trade decisions');
        }
      } catch (err) {
        setError('Error fetching trade decisions');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchLatestDecisions();
  }, []);

  // Listen for WebSocket decision updates
  const { isConnected } = useTradeDecisionUpdates((updatedDecision) => {
    setDecisionCycle(updatedDecision);
  });

  // Format scores and confidence levels
  const formatScore = (score: number): string => {
    return score.toFixed(2);
  };

  const getScoreBadgeColor = (score: number): string => {
    if (score >= 0.7) return 'bg-green-100 text-green-800';
    if (score >= 0.5) return 'bg-blue-100 text-blue-800';
    if (score >= 0.3) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  // Check if a candidate was executed
  const wasExecuted = (candidate: TradeCandidate): boolean => {
    if (!decisionCycle?.executed_trades) return false;
    return decisionCycle.executed_trades.includes(candidate.id);
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string): string => {
    return new Date(timestamp).toLocaleTimeString();
  };

  if (loading) {
    return (
      <div className="p-4 rounded-lg bg-navy-700 shadow-lg animate-pulse h-64">
        <div className="h-6 bg-navy-600 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-10 bg-navy-600 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 rounded-lg bg-red-900 text-white shadow-lg">
        <h3 className="text-lg font-semibold mb-2">Error Loading Trade Decisions</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!decisionCycle) {
    return (
      <div className="p-4 rounded-lg bg-navy-700 shadow-lg">
        <h3 className="text-lg font-semibold text-white mb-2">No Trade Decisions</h3>
        <p className="text-gray-300">No recent trade decision data available.</p>
      </div>
    );
  }

  return (
    <div className="p-4 rounded-lg bg-navy-700 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-white">Trade Decisions</h2>
        <div className="flex space-x-1 items-center">
          <span className="text-xs text-gray-400 mr-2">
            Updated: {formatTimestamp(decisionCycle.timestamp)}
          </span>
          {isConnected ? (
            <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
              Live
            </span>
          ) : (
            <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">
              Offline
            </span>
          )}
        </div>
      </div>

      {/* Market conditions summary */}
      <div className="mb-4 p-3 rounded bg-navy-600">
        <h3 className="text-sm font-medium text-gray-300 mb-2">Market Conditions</h3>
        <div className="flex flex-wrap gap-2">
          <div className="px-2 py-1 bg-navy-500 rounded text-xs text-white">
            Regime: {decisionCycle.market_conditions.regime}
          </div>
          <div className="px-2 py-1 bg-navy-500 rounded text-xs text-white">
            Volatility: {decisionCycle.market_conditions.volatility.toFixed(2)}
          </div>
          <div className="px-2 py-1 bg-navy-500 rounded text-xs text-white">
            Active Strategies: {decisionCycle.active_strategies.length}
          </div>
        </div>
      </div>

      {/* Trade candidates */}
      <div>
        <h3 className="text-sm font-medium text-gray-300 mb-2">Trade Candidates</h3>
        
        {decisionCycle.candidates.length === 0 ? (
          <p className="text-gray-400 text-sm">No trade candidates in this cycle.</p>
        ) : (
          <div className="space-y-3">
            {decisionCycle.candidates
              .slice(0, maxDecisions)
              .map((candidate) => (
                <div 
                  key={candidate.id} 
                  className={`p-3 rounded border ${
                    wasExecuted(candidate) 
                      ? 'border-green-700 bg-green-900 bg-opacity-20' 
                      : 'border-navy-600 bg-navy-600'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex items-center">
                      {candidate.action === 'buy' ? (
                        <FiArrowUp className="h-5 w-5 text-green-500 mr-2" />
                      ) : (
                        <FiArrowDown className="h-5 w-5 text-red-500 mr-2" />
                      )}
                      <div>
                        <p className="text-white font-medium">
                          {candidate.symbol} ({candidate.action.toUpperCase()})
                        </p>
                        <p className="text-xs text-gray-400">
                          Strategy: {candidate.strategy_name}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <span 
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium mr-2 ${
                          getScoreBadgeColor(candidate.score)
                        }`}
                      >
                        Score: {formatScore(candidate.score)}
                      </span>
                      {wasExecuted(candidate) ? (
                        <FiCheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <FiXCircle className="h-5 w-5 text-gray-500" />
                      )}
                    </div>
                  </div>
                  
                  <div className="mt-2">
                    <p className="text-xs text-gray-300">Reason: {candidate.reason}</p>
                    <div className="mt-1 flex justify-between items-center">
                      <p className="text-xs text-gray-400">
                        {formatTimestamp(candidate.timestamp)}
                      </p>
                      <span 
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                          getScoreBadgeColor(candidate.confidence)
                        }`}
                      >
                        Confidence: {formatScore(candidate.confidence)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default TradeDecisionMonitor;
