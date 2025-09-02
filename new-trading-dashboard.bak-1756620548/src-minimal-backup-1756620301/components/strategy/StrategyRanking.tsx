import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, AlertCircle, ArrowUpDown, Info, Filter } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Progress } from '@/components/ui/Progress';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuLabel, 
  DropdownMenuCheckboxItem, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from '@/components/ui/DropdownMenu';
import { ScrollArea } from '@/components/ui/ScrollArea';
import { } from '@/services/api';
import { strategyApi } from '@/services/api';

// Will eventually move these interfaces to a shared types file
export interface StrategyMetric {
  name: string;
  value: number;
  weight: number;
  description: string;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  priority_score: number;
  confidence: number;
  direction: 'long' | 'short' | 'neutral';
  timeframe: string;
  category: string;
  tags: string[];
  metrics: StrategyMetric[];
  symbols: string[];
  risk_level: 'low' | 'medium' | 'high';
  is_active: boolean;
  last_updated: string;
}

interface StrategyRankingProps {
  className?: string;
}

// This would ideally be in our strategyApi module
const getActiveStrategies = async () => strategyApi.getActiveStrategies();

const StrategyRanking: React.FC<StrategyRankingProps> = ({ className }) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [filters, setFilters] = useState({
    direction: [] as ('long' | 'short' | 'neutral')[],
    timeframe: [] as string[],
    category: [] as string[],
    riskLevel: [] as ('low' | 'medium' | 'high')[]
  });
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['active-strategies'],
    queryFn: async () => {
      const response = await getActiveStrategies();
      return response.success ? response.data : null;
    },
    staleTime: 30000, // 30 seconds
  });

  const getDirectionIcon = (direction: 'long' | 'short' | 'neutral') => {
    switch (direction) {
      case 'long': return <TrendingUp className="text-bull" size={16} />;
      case 'short': return <TrendingDown className="text-bear" size={16} />;
      default: return <ArrowUpDown className="text-muted-foreground" size={16} />;
    }
  };

  const getDirectionColor = (direction: 'long' | 'short' | 'neutral') => {
    switch (direction) {
      case 'long': return 'text-bull';
      case 'short': return 'text-bear';
      default: return 'text-muted-foreground';
    }
  };

  const getRiskBadge = (risk: 'low' | 'medium' | 'high') => {
    switch (risk) {
      case 'low': return <Badge variant="outline" className="text-success bg-success/10">Low Risk</Badge>;
      case 'medium': return <Badge variant="default">Medium Risk</Badge>;
      case 'high': return <Badge variant="destructive">High Risk</Badge>;
    }
  };

  // Apply filters to strategies
  const filteredStrategies = React.useMemo(() => {
    if (!data) return [];
    
    // Sort by priority score (highest first)
    const sorted = [...data].sort((a, b) => b.priority_score - a.priority_score);
    
    return sorted.filter(strategy => {
      const directionMatch = filters.direction.length === 0 || filters.direction.includes(strategy.direction);
      const timeframeMatch = filters.timeframe.length === 0 || filters.timeframe.includes(strategy.timeframe);
      const categoryMatch = filters.category.length === 0 || filters.category.includes(strategy.category);
      const riskMatch = filters.riskLevel.length === 0 || filters.riskLevel.includes(strategy.risk_level);
      
      return directionMatch && timeframeMatch && categoryMatch && riskMatch;
    });
  }, [data, filters]);

  // Get unique values for filter dropdowns
  const getUniqueValues = (field: 'timeframe' | 'category') => {
    if (!data) return [];
    const values = new Set<string>();
    data.forEach(strategy => values.add(strategy[field]));
    return Array.from(values);
  };
  
  const renderStrategyCard = (strategy: Strategy) => {
    const isExpanded = selectedStrategy === strategy.id;
    
    return (
      <div 
        key={strategy.id} 
        className={`p-3 border rounded-lg mb-3 cursor-pointer transition-all ${
          isExpanded ? 'border-primary/50 bg-primary/5' : 'border-border hover:border-primary/30'
        }`}
        onClick={() => setSelectedStrategy(isExpanded ? null : strategy.id)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            {getDirectionIcon(strategy.direction)}
            <span className={`ml-2 font-medium ${getDirectionColor(strategy.direction)}`}>
              {strategy.name}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {getRiskBadge(strategy.risk_level)}
            <Badge variant="outline" className="bg-muted/30">{strategy.timeframe}</Badge>
          </div>
        </div>
        
        <div className="mt-2">
          <div className="flex justify-between text-xs mb-1">
            <span>Priority Score: {(strategy.priority_score * 100).toFixed(0)}%</span>
            <span>Confidence: {(strategy.confidence * 100).toFixed(0)}%</span>
          </div>
          <Progress 
            value={strategy.priority_score * 100} 
            className={`h-2 ${
              strategy.direction === 'long' ? 'bg-bull/20' : 
              strategy.direction === 'short' ? 'bg-bear/20' : 'bg-muted/20'
            }`}
            indicatorClassName={
              strategy.direction === 'long' ? 'bg-bull' : 
              strategy.direction === 'short' ? 'bg-bear' : 'bg-muted'
            }
          />
        </div>
        
        <div className="flex flex-wrap gap-1 mt-2">
          {strategy.symbols.map(symbol => (
            <div key={symbol} className="text-xs font-medium px-2 py-0.5 bg-primary/10 text-primary rounded-full">
              ${symbol}
            </div>
          ))}
        </div>
        
        {isExpanded && (
          <div className="mt-3 space-y-3 text-sm">
            <p>{strategy.description}</p>
            
            <div>
              <h4 className="text-xs font-medium mb-2">Strategy Metrics:</h4>
              <div className="space-y-2">
                {strategy.metrics.map(metric => (
                  <div key={metric.name} className="text-xs">
                    <div className="flex justify-between mb-1">
                      <span className="font-medium">{metric.name}</span>
                      <span className="flex items-center">
                        Value: {metric.value.toFixed(2)}
                        <span className="ml-2 text-muted-foreground">
                          (Weight: {(metric.weight * 100).toFixed(0)}%)
                        </span>
                      </span>
                    </div>
                    <Progress 
                      value={metric.value * 100} 
                      className="h-1.5 bg-muted/20"
                    />
                    <p className="text-muted-foreground mt-1 text-xs">{metric.description}</p>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="flex flex-wrap gap-1.5">
              {strategy.tags.map(tag => (
                <Badge key={tag} variant="outline" className="bg-muted/20">{tag}</Badge>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };
  
  const renderSkeleton = () => {
    return Array(5).fill(0).map((_, index) => (
      <div key={index} className="p-3 border rounded-lg mb-3">
        <div className="flex items-center justify-between">
          <div className="h-5 bg-muted/20 animate-pulse rounded-md w-1/3"></div>
          <div className="flex items-center gap-2">
            <div className="h-5 bg-muted/20 animate-pulse rounded-md w-16"></div>
            <div className="h-5 bg-muted/20 animate-pulse rounded-md w-16"></div>
          </div>
        </div>
        <div className="mt-2">
          <div className="flex justify-between mb-1">
            <div className="h-4 bg-muted/20 animate-pulse rounded-md w-24"></div>
            <div className="h-4 bg-muted/20 animate-pulse rounded-md w-24"></div>
          </div>
          <div className="h-2 bg-muted/20 animate-pulse rounded-md w-full"></div>
        </div>
        <div className="flex gap-1 mt-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-5 bg-muted/20 animate-pulse rounded-full w-16"></div>
          ))}
        </div>
      </div>
    ));
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-base">Strategy Ranking</CardTitle>
          <div className="flex gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="h-8 gap-1">
                  <Filter size={14} /> Filters
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>Direction</DropdownMenuLabel>
                {['long', 'short', 'neutral'].map(direction => (
                  <DropdownMenuCheckboxItem
                    key={direction}
                    checked={filters.direction.includes(direction as any)}
                    onCheckedChange={(checked) => {
                      setFilters(prev => ({
                        ...prev,
                        direction: checked 
                          ? [...prev.direction, direction as 'long' | 'short' | 'neutral']
                          : prev.direction.filter(d => d !== direction)
                      }));
                    }}
                  >
                    {direction.charAt(0).toUpperCase() + direction.slice(1)}
                  </DropdownMenuCheckboxItem>
                ))}
                
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Timeframe</DropdownMenuLabel>
                {getUniqueValues('timeframe').map(timeframe => (
                  <DropdownMenuCheckboxItem
                    key={timeframe}
                    checked={filters.timeframe.includes(timeframe)}
                    onCheckedChange={(checked) => {
                      setFilters(prev => ({
                        ...prev,
                        timeframe: checked 
                          ? [...prev.timeframe, timeframe]
                          : prev.timeframe.filter(t => t !== timeframe)
                      }));
                    }}
                  >
                    {timeframe}
                  </DropdownMenuCheckboxItem>
                ))}
                
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Risk Level</DropdownMenuLabel>
                {['low', 'medium', 'high'].map(risk => (
                  <DropdownMenuCheckboxItem
                    key={risk}
                    checked={filters.riskLevel.includes(risk as any)}
                    onCheckedChange={(checked) => {
                      setFilters(prev => ({
                        ...prev,
                        riskLevel: checked 
                          ? [...prev.riskLevel, risk as 'low' | 'medium' | 'high']
                          : prev.riskLevel.filter(r => r !== risk)
                      }));
                    }}
                  >
                    {risk.charAt(0).toUpperCase() + risk.slice(1)} Risk
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            
            <Button variant="ghost" size="sm" className="h-8">
              <Info size={16} className="mr-1" /> Info
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="flex items-center text-destructive text-sm">
            <AlertCircle size={16} className="mr-2" />
            Failed to load strategy data
          </div>
        ) : (
          <ScrollArea className="h-[500px] pr-4">
            {isLoading ? renderSkeleton() : (
              filteredStrategies.length === 0 ? (
                <p className="text-sm text-muted-foreground">No strategies yet — connectivity OK.</p>
              ) : (
                filteredStrategies.map(renderStrategyCard)
              )
            )}
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
};

export default StrategyRanking;
