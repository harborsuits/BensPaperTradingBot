import React, { useState, useEffect } from 'react';
import { strategyApi } from '@/services/api';
import { Strategy } from '@/types/api.types';
import { useStrategyUpdates } from '@/hooks/useWebSocketSubscriptions';
import { FiArrowUp, FiArrowDown } from 'react-icons/fi';

const RANKING_METRICS = [
  { id: 'overall', label: 'Overall' },
  { id: 'marketFit', label: 'Market Fit' },
  { id: 'risk', label: 'Risk-Adjusted' },
  { id: 'performance', label: 'Performance' },
  { id: 'trend', label: 'Trend Following' },
  { id: 'volatility', label: 'Volatility Response' },
];

interface StrategyRankingTableProps {
  defaultMetric?: string;
  limit?: number;
}

const StrategyRankingTable: React.FC<StrategyRankingTableProps> = ({ 
  defaultMetric = 'overall',
  limit = 10
}) => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState(defaultMetric);

  // Fetch initial ranked strategies
  useEffect(() => {
    const fetchRankedStrategies = async () => {
      setLoading(true);
      try {
        const response = await strategyApi.getRankedStrategies(selectedMetric, limit);
        if (response.success && response.data) {
          setStrategies(response.data);
        } else {
          setError(response.error || 'Failed to fetch strategies');
        }
      } catch (err) {
        setError('Error fetching ranked strategies');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchRankedStrategies();
  }, [selectedMetric, limit]);

  // Listen for WebSocket strategy updates
  const { isConnected } = useStrategyUpdates((updatedStrategy) => {
    setStrategies(prevStrategies => {
      // Update the strategy in the existing array if it exists
      const exists = prevStrategies.some(s => s.id === updatedStrategy.id);
      if (exists) {
        return prevStrategies.map(s => 
          s.id === updatedStrategy.id ? { ...s, ...updatedStrategy } : s
        );
      }
      // If we have fewer than limit strategies or this strategy should be in the top,
      // we need to refetch the ranked list
      if (prevStrategies.length < limit) {
        strategyApi.getRankedStrategies(selectedMetric, limit)
          .then(response => {
            if (response.success && response.data) {
              setStrategies(response.data);
            }
          });
        return prevStrategies;
      }
      return prevStrategies;
    });
  });

  // Format score for display
  const formatScore = (score: number): string => {
    if (score === undefined || score === null) return 'N/A';
    return score.toFixed(2);
  };

  // Get score based on selected metric
  const getScore = (strategy: Strategy): number => {
    if (selectedMetric === 'marketFit') {
      return strategy.marketFit || 0;
    }
    if (selectedMetric === 'risk') {
      return strategy.riskScore || 0;
    }
    if (selectedMetric === 'performance') {
      return strategy.performanceScore || 0;
    }
    if (selectedMetric === 'trend') {
      return strategy.trendScore || 0;
    }
    if (selectedMetric === 'volatility') {
      return strategy.volatilityScore || 0;
    }
    
    // Default to getting the rank from the rankings object
    return strategy.rankings?.overall || 0;
  };

  // Determine badge color based on strategy status
  const getStatusBadgeColor = (status: string): string => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'inactive':
        return 'bg-gray-100 text-gray-800';
      case 'testing':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Toggle strategy status handler
  const handleToggleStatus = async (strategy: Strategy) => {
    try {
      if (strategy.status === 'active') {
        await strategyApi.disableStrategy(strategy.id);
      } else {
        await strategyApi.enableStrategy(strategy.id);
      }
      // No need to update state here as we'll get the WebSocket update
    } catch (err) {
      console.error('Error toggling strategy status:', err);
      setError('Failed to update strategy status');
    }
  };

  if (loading) {
    return (
      <div className="p-4 rounded-lg bg-navy-700 shadow-lg animate-pulse h-64">
        <div className="h-6 bg-navy-600 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-10 bg-navy-600 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 rounded-lg bg-red-900 text-white shadow-lg">
        <h3 className="text-lg font-semibold mb-2">Error Loading Strategies</h3>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="p-4 rounded-lg bg-navy-700 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-white">Strategy Rankings</h2>
        <div className="flex space-x-1">
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

      {/* Metric selection tabs */}
      <div className="flex mb-4 overflow-x-auto pb-2">
        {RANKING_METRICS.map((metric) => (
          <button
            key={metric.id}
            className={`px-3 py-1 text-sm font-medium rounded-full mr-2 transition ${
              selectedMetric === metric.id
                ? 'bg-blue-600 text-white'
                : 'bg-navy-600 text-gray-300 hover:bg-navy-500'
            }`}
            onClick={() => setSelectedMetric(metric.id)}
          >
            {metric.label}
          </button>
        ))}
      </div>

      {/* Strategy ranking table */}
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead className="border-b border-navy-600">
            <tr>
              <th className="py-2 px-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Rank
              </th>
              <th className="py-2 px-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Strategy
              </th>
              <th className="py-2 px-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Score
              </th>
              <th className="py-2 px-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Status
              </th>
              <th className="py-2 px-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-navy-600">
            {strategies.map((strategy, index) => (
              <tr key={strategy.id} className="hover:bg-navy-600">
                <td className="py-2 px-3 whitespace-nowrap text-sm font-medium text-white">
                  {index + 1}
                </td>
                <td className="py-2 px-3 whitespace-nowrap">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-white">{strategy.name}</span>
                    <span className="text-xs text-gray-400">{strategy.description}</span>
                  </div>
                </td>
                <td className="py-2 px-3 whitespace-nowrap text-sm text-white">
                  {formatScore(getScore(strategy))}
                </td>
                <td className="py-2 px-3 whitespace-nowrap">
                  <span className={`px-2 py-1 text-xs rounded-full ${getStatusBadgeColor(strategy.status)}`}>
                    {strategy.status}
                  </span>
                </td>
                <td className="py-2 px-3 whitespace-nowrap text-sm">
                  <button
                    onClick={() => handleToggleStatus(strategy)}
                    className={`px-3 py-1 rounded text-xs font-medium ${
                      strategy.status === 'active'
                        ? 'bg-red-100 text-red-800 hover:bg-red-200'
                        : 'bg-green-100 text-green-800 hover:bg-green-200'
                    }`}
                  >
                    {strategy.status === 'active' ? 'Disable' : 'Enable'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StrategyRankingTable;
