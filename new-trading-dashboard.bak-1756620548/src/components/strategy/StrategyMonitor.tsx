import React, { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  RefreshCw, 
  AlertTriangle, 
  BarChart3,
  Settings,
  TrendingUp,
  TrendingDown,
  Minus,
  ToggleLeft,
  ToggleRight
} from 'lucide-react';
import { Strategy, strategyApi } from '@/services/strategyApi';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { formatDistanceToNow } from 'date-fns';
import { toast } from 'react-hot-toast';

// Import custom styles
import '@/styles/strategy-monitor.css';

interface StrategyMonitorProps {
  className?: string;
}

const StrategyMonitor: React.FC<StrategyMonitorProps> = ({ className }) => {
  const queryClient = useQueryClient();
  const [expandedStrategy, setExpandedStrategy] = useState<string | null>(null);
  
  // Fetch active strategies
  const { 
    data: strategies, 
    isLoading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['active-strategies'],
    queryFn: async () => {
      try {
        const response = await strategyApi.getActiveStrategies();
        if (!response.success) {
          throw new Error(response.error || 'Failed to fetch active strategies');
        }
        return response.data;
      } catch (err: any) {
        console.error('Error fetching active strategies:', err);
        toast.error(`Failed to load strategies: ${err.message || 'Unknown error'}`);
        throw err;
      }
    },
    staleTime: 60000, // 1 minute
    retry: 2,
  });
  
  // Handle strategy toggle
  const handleToggleStrategy = async (strategyId: string) => {
    try {
      toast.loading(`Updating strategy status...`, { id: `toggle-${strategyId}` });
      
      const response = await strategyApi.toggleStrategy(strategyId);
      
      if (response.success) {
        toast.success(`Strategy ${response.data.strategy.enabled ? 'enabled' : 'disabled'} successfully`, { id: `toggle-${strategyId}` });
        
        // Update the cache with the updated strategy
        queryClient.setQueryData(['active-strategies'], (oldData: Strategy[] | undefined) => {
          if (!oldData) return oldData;
          
          return oldData.map(strategy => 
            strategy.id === strategyId 
              ? response.data.strategy 
              : strategy
          );
        });
      } else {
        toast.error(`Failed to update strategy: ${response.error}`, { id: `toggle-${strategyId}` });
      }
    } catch (err: any) {
      toast.error(`Error updating strategy: ${err.message || 'Unknown error'}`, { id: `toggle-${strategyId}` });
    }
  };
  
  // Render signal direction icon
  const renderSignalDirection = (direction?: 'long' | 'short' | 'neutral') => {
    switch (direction) {
      case 'long':
        return <TrendingUp className="text-bull" size={18} />;
      case 'short':
        return <TrendingDown className="text-bear" size={18} />;
      case 'neutral':
      default:
        return <Minus className="text-muted-foreground" size={18} />;
    }
  };
  
  // Get status badge styling
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge className="bg-green-500">Active</Badge>;
      case 'paused':
        return <Badge className="bg-amber-500">Paused</Badge>;
      case 'error':
        return <Badge className="bg-red-500">Error</Badge>;
      case 'idle':
      default:
        return <Badge className="bg-slate-500">Idle</Badge>;
    }
  };
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-xl">Active Strategies</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="p-4 border rounded-lg space-y-2">
                <div className="flex justify-between items-center">
                  <div className="h-6 w-40 bg-muted/20 animate-pulse rounded-md"></div>
                  <div className="h-6 w-16 bg-muted/20 animate-pulse rounded-md"></div>
                </div>
                <div className="h-4 w-full bg-muted/20 animate-pulse rounded-md"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }
  
  if (error || !strategies) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-xl">Active Strategies</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="py-4 flex flex-col items-center justify-center">
            <AlertTriangle size={40} className="text-amber-500 mb-2" />
            <p className="text-muted-foreground text-center mb-2">Could not load strategy data</p>
            <Button onClick={() => refetch()} size="sm">Retry</Button>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader className="relative">
        <CardTitle className="text-xl">Active Strategies</CardTitle>
        <Button
          variant="ghost" 
          size="icon"
          className="absolute top-2 right-2 p-1 h-8 w-8"
          onClick={() => {
            refetch();
            toast.success('Refreshing strategy data...');
          }}
        >
          <RefreshCw size={16} />
        </Button>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {strategies.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <BarChart3 size={32} className="mb-2" />
              <p className="text-sm">No active strategies</p>
            </div>
          ) : (
            <div className="space-y-4">
              {strategies.map((strategy: Strategy) => (
                <div 
                  key={strategy.id}
                  className={`border rounded-lg transition-all ${
                    strategy.enabled 
                      ? 'border-primary/30 bg-primary/5' 
                      : 'border-border bg-card'
                  }`}
                >
                  <div className="p-4">
                    <div className="flex justify-between items-center">
                      <div>
                        <div className="flex items-center space-x-2">
                          <h3 className="font-medium">{strategy.name}</h3>
                          {getStatusBadge(strategy.status)}
                          <Badge variant="outline" className="bg-slate-500/10">
                            {strategy.category}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{strategy.description}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button 
                          variant="ghost" 
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => setExpandedStrategy(expandedStrategy === strategy.id ? null : strategy.id)}
                          aria-label={expandedStrategy === strategy.id ? "Collapse strategy details" : "Expand strategy details"}
                          aria-expanded={expandedStrategy === strategy.id}
                          title={expandedStrategy === strategy.id ? "Collapse strategy details" : "Expand strategy details"}
                        >
                          {expandedStrategy === strategy.id ? 
                            <Minus size={16} /> : 
                            <BarChart3 size={16} />
                          }
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => handleToggleStrategy(strategy.id)}
                          aria-label="Configure strategy settings"
                          title="Configure strategy settings"
                        >
                          <Settings size={16} />
                        </Button>
                        <div className="flex items-center space-x-1">
                          <Button 
                            variant="ghost" 
                            size="sm"
                            className="flex items-center gap-1 px-2 py-1 h-auto"
                            onClick={() => handleToggleStrategy(strategy.id)}
                            aria-label={strategy.enabled ? "Disable strategy" : "Enable strategy"}
                            title={strategy.enabled ? "Disable strategy" : "Enable strategy"}
                          >
                            {strategy.enabled ? (
                              <>
                                <ToggleRight size={16} className="text-green-500" />
                                <span className="text-xs">Enabled</span>
                              </>
                            ) : (
                              <>
                                <ToggleLeft size={16} className="text-muted-foreground" />
                                <span className="text-xs">Disabled</span>
                              </>
                            )}
                          </Button>
                        </div>
                      </div>
                    </div>
                    
                    {strategy.lastSignal && (
                      <div className="flex items-center mt-2 text-sm">
                        <div className="flex items-center mr-4">
                          <span className="text-muted-foreground mr-1">Signal:</span>
                          <div className="flex items-center">
                            {renderSignalDirection(strategy.lastSignal.direction)}
                            <span className="ml-1">
                              {strategy.lastSignal.direction.charAt(0).toUpperCase() + 
                               strategy.lastSignal.direction.slice(1)}
                            </span>
                            <span className="ml-1 text-xs text-muted-foreground">
                              ({Math.round(strategy.lastSignal.strength * 100)}%)
                            </span>
                          </div>
                        </div>
                        {strategy.lastSignal.timestamp && (
                          <div className="text-xs text-muted-foreground">
                            {formatDistanceToNow(new Date(strategy.lastSignal.timestamp), { addSuffix: true })}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {expandedStrategy === strategy.id && (
                      <div className="mt-4 pt-4 border-t border-border">
                        <div className="grid grid-cols-4 gap-2">
                          <div className="text-center p-2 bg-card rounded-md">
                            <p className="text-xs text-muted-foreground mb-1">Daily</p>
                            <p className={`text-sm font-bold ${
                              strategy.performance.daily >= 0 ? 'text-bull' : 'text-bear'
                            }`}>
                              {strategy.performance.daily >= 0 ? '+' : ''}
                              {strategy.performance.daily.toFixed(2)}%
                            </p>
                          </div>
                          <div className="text-center p-2 bg-card rounded-md">
                            <p className="text-xs text-muted-foreground mb-1">Weekly</p>
                            <p className={`text-sm font-bold ${
                              strategy.performance.weekly >= 0 ? 'text-bull' : 'text-bear'
                            }`}>
                              {strategy.performance.weekly >= 0 ? '+' : ''}
                              {strategy.performance.weekly.toFixed(2)}%
                            </p>
                          </div>
                          <div className="text-center p-2 bg-card rounded-md">
                            <p className="text-xs text-muted-foreground mb-1">Monthly</p>
                            <p className={`text-sm font-bold ${
                              strategy.performance.monthly >= 0 ? 'text-bull' : 'text-bear'
                            }`}>
                              {strategy.performance.monthly >= 0 ? '+' : ''}
                              {strategy.performance.monthly.toFixed(2)}%
                            </p>
                          </div>
                          <div className="text-center p-2 bg-card rounded-md">
                            <p className="text-xs text-muted-foreground mb-1">Overall</p>
                            <p className={`text-sm font-bold ${
                              strategy.performance.overall >= 0 ? 'text-bull' : 'text-bear'
                            }`}>
                              {strategy.performance.overall >= 0 ? '+' : ''}
                              {strategy.performance.overall.toFixed(2)}%
                            </p>
                          </div>
                        </div>
                        
                        <div className="mt-4">
                          <p className="text-xs font-medium text-muted-foreground mb-2">
                            Market Suitability Score
                          </p>
                          <div className="h-2 w-full bg-muted/20 rounded-full overflow-hidden">
                            <div 
                              className={`h-full bg-primary market-suitability-${Math.round(Math.min(100, Math.max(0, strategy.marketSuitability * 100)) / 10) * 10}`}
                            ></div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default StrategyMonitor;
