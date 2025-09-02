import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { PieChart, BarChart3, Award, Sliders } from 'lucide-react';
import { contextApi } from '@/services/contextApi';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Slider } from '@/components/ui/Slider';
import { Button } from '@/components/ui/Button';
import { Skeleton } from '@/components/ui/Skeleton';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';

type StrategyAllocationProps = {
  onAllocationChange?: (allocation: Record<string, number>) => void;
};

export const StrategyAllocationDisplay: React.FC<StrategyAllocationProps> = ({ 
  onAllocationChange 
}) => {
  const [performanceWeight, setPerformanceWeight] = useState<number>(0.3);
  const [activeTab, setActiveTab] = useState<string>('chart');
  
  const { 
    data: allocationData, 
    isLoading, 
    isError, 
    refetch 
  } = useQuery({
    queryKey: ['strategyAllocation', performanceWeight],
    queryFn: () => contextApi.getStrategyAllocation({ 
      performanceWeight 
    }),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 15 * 60 * 1000, // 15 minutes
    onSuccess: (data) => {
      if (data?.success && data.data?.allocation) {
        onAllocationChange?.(data.data.allocation);
      }
    }
  });
  
  const handlePerformanceWeightChange = (value: number[]) => {
    setPerformanceWeight(value[0]);
  };
  
  const applyWeightChange = () => {
    refetch();
  };
  
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <PieChart size={18} className="mr-2" />
            Strategy Allocation
          </CardTitle>
          <CardDescription>Loading allocation data...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-48 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }
  
  if (isError || !allocationData?.success) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <PieChart size={18} className="mr-2" />
            Strategy Allocation
          </CardTitle>
          <CardDescription>Error loading allocation data</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">
            <p>Failed to load strategy allocation information.</p>
            <Button 
              variant="outline" 
              size="sm" 
              className="mt-2" 
              onClick={() => refetch()}
            >
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  const { 
    allocation = {}, 
    market_context = {}, 
    performance_metrics = {},
    performance_weight: actualWeight
  } = allocationData.data || {};
  
  // Sort strategies by allocation percentage (descending)
  const sortedStrategies = Object.entries(allocation)
    .sort(([, allocA], [, allocB]) => (allocB as number) - (allocA as number));
  
  // Create data for charts
  const chartData = sortedStrategies.map(([name, value]) => ({
    name,
    value: Number(value).toFixed(1),
    color: getColorForStrategy(name, Number(value))
  }));
  
  // Helper function to get color for strategy based on name or allocation
  function getColorForStrategy(name: string, allocation: number): string {
    // Map strategy names to colors - this is just an example
    const colorMap: Record<string, string> = {
      'momentum': '#4ade80',
      'mean_reversion': '#3b82f6',
      'trend_following': '#f97316',
      'breakout': '#a855f7',
      'value': '#06b6d4',
      'sentiment': '#ec4899',
      'volatility': '#8b5cf6',
      'pairs_trading': '#10b981',
      'arbitrage': '#f43f5e'
    };
    
    // Return color by name or default based on allocation value
    return colorMap[name.toLowerCase()] || 
      (allocation > 15 ? '#4ade80' : 
       allocation > 10 ? '#3b82f6' : 
       allocation > 5 ? '#f97316' : '#a3a3a3');
  }
  
  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center">
            <PieChart size={18} className="mr-2" />
            Strategy Allocation
          </CardTitle>
          <Tabs defaultValue={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-[180px] grid-cols-2">
              <TabsTrigger value="chart">Chart</TabsTrigger>
              <TabsTrigger value="table">Table</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        <CardDescription>
          Dynamic capital allocation between strategies
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Performance weight slider */}
          <div className="bg-muted/30 rounded-lg p-4">
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center">
                <Sliders size={16} className="mr-2" />
                <h3 className="text-sm font-medium">Performance Weight</h3>
              </div>
              <span className="text-xs font-medium">{(performanceWeight * 100).toFixed(0)}%</span>
            </div>
            <Slider
              defaultValue={[performanceWeight]}
              min={0}
              max={0.5}
              step={0.05}
              value={[performanceWeight]}
              onValueChange={handlePerformanceWeightChange}
              onValueCommit={applyWeightChange}
              className="mb-2"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Market Driven</span>
              <span>Balanced</span>
              <span>Performance Driven</span>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Adjust how much recent performance affects strategy allocation
            </p>
          </div>
          
          {/* Allocation visualization */}
          <TabsContent value="chart" className="mt-0 pt-0">
            {/* Chart visualization would go here */}
            <div className="h-48 border rounded-md flex items-center justify-center">
              <p className="text-muted-foreground text-sm">Strategy allocation chart visualization</p>
            </div>
          </TabsContent>
          
          <TabsContent value="table" className="mt-0 pt-0 space-y-2">
            {chartData.map(strategy => (
              <div 
                key={strategy.name} 
                className="flex items-center justify-between py-2 border-b border-muted last:border-0"
              >
                <div className="flex items-center">
                  <div 
                    className="w-3 h-3 rounded-full mr-2" 
                    style={{ backgroundColor: strategy.color }}
                  ></div>
                  <span className="font-medium">{strategy.name}</span>
                  
                  {/* Performance metrics */}
                  {performance_metrics[strategy.name] && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-6 w-6 ml-1">
                            <Award size={14} />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <div className="text-xs space-y-1">
                            <p>Sharpe: {performance_metrics[strategy.name].sharpe_ratio.toFixed(2)}</p>
                            <p>Win Rate: {(performance_metrics[strategy.name].win_rate * 100).toFixed(0)}%</p>
                            <p>Avg Return: {(performance_metrics[strategy.name].avg_return * 100).toFixed(2)}%</p>
                          </div>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                </div>
                <div>
                  <div className="flex items-center">
                    <div className="w-36 bg-muted rounded-full h-2 mr-2">
                      <div 
                        className="h-full rounded-full" 
                        style={{ 
                          width: `${strategy.value}%`,
                          backgroundColor: strategy.color
                        }}
                      ></div>
                    </div>
                    <span className="text-sm">{strategy.value}%</span>
                  </div>
                </div>
              </div>
            ))}
          </TabsContent>
          
          {/* Context information */}
          <div className="text-xs text-muted-foreground mt-4">
            <div className="flex justify-between mb-1">
              <span>Market Regime:</span>
              <span className="font-medium capitalize">
                {market_context.regime?.regime?.replace(/_/g, ' ') || 'Unknown'}
              </span>
            </div>
            <div className="flex justify-between mb-1">
              <span>Market Bias:</span>
              <span className="font-medium capitalize">{market_context.bias || 'Neutral'}</span>
            </div>
            <div className="flex justify-between">
              <span>Volatility:</span>
              <span className="font-medium capitalize">{market_context.volatility || 'Normal'}</span>
            </div>
          </div>
          
          <div className="flex justify-end mt-2">
            <Button variant="ghost" size="sm" onClick={() => refetch()}>
              Refresh
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default StrategyAllocationDisplay;
