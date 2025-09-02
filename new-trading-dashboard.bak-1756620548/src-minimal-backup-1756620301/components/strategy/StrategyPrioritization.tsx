import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { useWebSocketChannel, useWebSocketChannelMessage } from '@/services/websocketManager';
import { Check, Play, Pause, AlertTriangle, Activity } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Progress } from '@/components/ui/Progress';

// Types for strategy data
interface Strategy {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'idle' | 'testing' | 'disabled';
  priority_score: number;
  last_signal?: {
    symbol: string;
    action: 'buy' | 'sell' | 'hold';
    strength: number;
    timestamp: string;
  };
  reasoning?: string;
  performance_metrics?: {
    sharpe_ratio?: number;
    max_drawdown?: number;
    win_rate?: number;
    profit_factor?: number;
  };
}

const StrategyPrioritization: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);
  const [showOnlyActive, setShowOnlyActive] = useState(false);

  // Subscribe to the strategies channel
  useWebSocketChannel('strategies');

  // Fetch initial strategies data
  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const response = await fetch('/api/strategies');
        if (response.ok) {
          const data = await response.json();
          setStrategies(data);
          setLastUpdateTime(new Date());
        }
      } catch (error) {
        console.error('Error fetching strategies data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStrategies();
  }, []);

  // Listen for strategy updates via WebSocket
  useWebSocketChannelMessage('strategies', (message) => {
    if (message.event === 'strategy_priority_update') {
      // Update the strategies list with new priorities
      setStrategies(message.payload.strategies);
      setLastUpdateTime(new Date());
    } else if (message.event === 'strategy_status_change') {
      // Update a specific strategy's status
      const { strategy_id, new_status, reason } = message.payload;
      setStrategies(prevStrategies => 
        prevStrategies.map(strategy => 
          strategy.id === strategy_id 
            ? { ...strategy, status: new_status, reasoning: reason } 
            : strategy
        )
      );
      setLastUpdateTime(new Date());
    } else if (message.event === 'strategy_signal') {
      // Update a strategy's last signal
      const { strategy_id, signal } = message.payload;
      setStrategies(prevStrategies => 
        prevStrategies.map(strategy => 
          strategy.id === strategy_id 
            ? { ...strategy, last_signal: signal } 
            : strategy
        )
      );
      setLastUpdateTime(new Date());
    }
  });

  // Get sorted strategies (by priority score)
  const getSortedStrategies = () => {
    const sorted = [...strategies].sort((a, b) => b.priority_score - a.priority_score);
    return showOnlyActive ? sorted.filter(s => s.status === 'active') : sorted;
  };

  // Get status badge
  const getStatusBadge = (status: Strategy['status']) => {
    switch (status) {
      case 'active':
        return <Badge variant="outline" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">Active</Badge>;
      case 'idle':
        return <Badge variant="outline" className="bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300">Idle</Badge>;
      case 'testing':
        return <Badge variant="outline" className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">Testing</Badge>;
      case 'disabled':
        return <Badge variant="outline" className="bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300">Disabled</Badge>;
      default:
        return null;
    }
  };

  // Get status icon
  const getStatusIcon = (status: Strategy['status']) => {
    switch (status) {
      case 'active':
        return <Play size={16} className="text-green-500" />;
      case 'idle':
        return <Pause size={16} className="text-gray-500" />;
      case 'testing':
        return <Activity size={16} className="text-blue-500" />;
      case 'disabled':
        return <AlertTriangle size={16} className="text-red-500" />;
      default:
        return null;
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center text-lg">
            <Activity size={18} className="mr-2" />
            Strategy Prioritization
          </CardTitle>
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="showActiveOnly"
                checked={showOnlyActive}
                onChange={() => setShowOnlyActive(!showOnlyActive)}
                className="mr-2"
              />
              <label htmlFor="showActiveOnly" className="text-xs">Show Active Only</label>
            </div>
            {lastUpdateTime && (
              <span className="text-xs text-muted-foreground">
                Updated: {lastUpdateTime.toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="h-24 bg-muted/20 animate-pulse rounded-md"></div>
            ))}
          </div>
        ) : strategies.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8">
            <AlertTriangle size={24} className="text-muted-foreground mb-2" />
            <p className="text-muted-foreground">No strategies available</p>
          </div>
        ) : (
          <div className="space-y-4 max-h-[500px] overflow-y-auto pr-1">
            {getSortedStrategies().map((strategy) => (
              <div 
                key={strategy.id}
                className={`p-3 border rounded-md transition-colors ${
                  strategy.status === 'active' ? 'border-green-300 dark:border-green-800' : ''
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center">
                    {getStatusIcon(strategy.status)}
                    <h3 className="font-medium ml-2">{strategy.name}</h3>
                  </div>
                  {getStatusBadge(strategy.status)}
                </div>
                
                <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                  {strategy.description || 'No description available'}
                </p>
                
                <div className="mb-2">
                  <div className="flex justify-between text-xs mb-1">
                    <span>Priority Score</span>
                    <span className="font-medium">{(strategy.priority_score * 100).toFixed(0)}</span>
                  </div>
                  <Progress value={strategy.priority_score * 100} className="h-1.5" />
                </div>
                
                {strategy.reasoning && (
                  <div className="mt-2 text-xs bg-muted/20 p-2 rounded">
                    <span className="font-medium">Reasoning: </span>
                    {strategy.reasoning}
                  </div>
                )}
                
                {strategy.last_signal && (
                  <div className="mt-2 text-xs flex items-center">
                    <span className="font-medium mr-1">Last Signal:</span>
                    <span className={`font-medium ${
                      strategy.last_signal.action === 'buy' ? 'text-green-500' :
                      strategy.last_signal.action === 'sell' ? 'text-red-500' :
                      'text-amber-500'
                    }`}>
                      {strategy.last_signal.action.toUpperCase()} {strategy.last_signal.symbol}
                    </span>
                    <span className="ml-2 text-muted-foreground">
                      ({new Date(strategy.last_signal.timestamp).toLocaleTimeString()})
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default StrategyPrioritization;
