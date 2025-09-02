import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Gauge, TrendingUp, AlertTriangle } from 'lucide-react';
import { contextApi } from '@/services/contextApi';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Progress } from '@/components/ui/Progress';
import { Skeleton } from '@/components/ui/Skeleton';

type MarketRegimeProps = {
  symbols?: string[];
  onRegimeChange?: (regime: string, confidence: number) => void;
};

const regimeColors: Record<string, string> = {
  bullish: 'bg-emerald-500',
  'strongly_bullish': 'bg-emerald-600',
  'cautiously_bullish': 'bg-emerald-400',
  bearish: 'bg-red-500',
  'strongly_bearish': 'bg-red-600',
  'cautiously_bearish': 'bg-red-400',
  neutral: 'bg-blue-500',
  choppy: 'bg-orange-500',
  volatile: 'bg-purple-500',
  'low_volatility': 'bg-sky-500',
  'high_volatility': 'bg-fuchsia-500',
  recovery: 'bg-teal-500',
  reversal: 'bg-amber-500',
  breakdown: 'bg-rose-500',
  unknown: 'bg-gray-500'
};

const regimeIcons: Record<string, React.ReactNode> = {
  bullish: <TrendingUp className="text-emerald-500" />,
  'strongly_bullish': <TrendingUp className="text-emerald-600" />,
  'cautiously_bullish': <TrendingUp className="text-emerald-400" />,
  bearish: <TrendingUp className="text-red-500 transform rotate-180" />,
  'strongly_bearish': <TrendingUp className="text-red-600 transform rotate-180" />,
  'cautiously_bearish': <TrendingUp className="text-red-400 transform rotate-180" />,
  neutral: <Gauge className="text-blue-500" />,
  choppy: <AlertTriangle className="text-orange-500" />,
  volatile: <AlertTriangle className="text-purple-500" />,
  'low_volatility': <Gauge className="text-sky-500" />,
  'high_volatility': <AlertTriangle className="text-fuchsia-500" />,
  recovery: <TrendingUp className="text-teal-500" />,
  reversal: <AlertTriangle className="text-amber-500" />,
  breakdown: <AlertTriangle className="text-rose-500" />,
  unknown: <Gauge className="text-gray-500" />
};

export const MarketRegimeDisplay: React.FC<MarketRegimeProps> = ({ 
  symbols = [], 
  onRegimeChange 
}) => {
  const [timeframe, setTimeframe] = useState<string>('daily');
  
  const { 
    data: regimeData, 
    isLoading, 
    isError, 
    refetch 
  } = useQuery({
    queryKey: ['marketRegime', symbols, timeframe],
    queryFn: () => contextApi.getMarketRegime({ 
      symbols: symbols.length > 0 ? symbols : undefined, 
      timeframe 
    }),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 15 * 60 * 1000, // 15 minutes
  });
  
  // Notify parent component when regime changes
  useEffect(() => {
    if (regimeData?.success && regimeData.data?.aggregate_regime) {
      const { primary_regime, confidence } = regimeData.data.aggregate_regime;
      onRegimeChange?.(primary_regime, confidence);
    }
  }, [regimeData, onRegimeChange]);
  
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Gauge size={18} className="mr-2" />
            Market Regime Analysis
          </CardTitle>
          <CardDescription>Loading market regime data...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }
  
  if (isError || !regimeData?.success) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <AlertTriangle size={18} className="mr-2 text-destructive" />
            Market Regime Analysis
          </CardTitle>
          <CardDescription>Error loading market regime data</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">
            <p>Failed to load market regime information.</p>
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
    aggregate_regime, 
    symbol_regimes = {}, 
    recommended_strategies = []
  } = regimeData.data || {};
  
  const { primary_regime, confidence, secondary_regimes = [] } = aggregate_regime || {};
  
  return (
    <Card className="border-l-4" style={{ borderLeftColor: `var(--${regimeColors[primary_regime || 'unknown'] || 'bg-gray-500'})` }}>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center">
            <span className="mr-2">
              {regimeIcons[primary_regime || 'unknown']}
            </span>
            Market Regime Analysis
          </CardTitle>
          <Tabs defaultValue={timeframe} onValueChange={setTimeframe}>
            <TabsList className="grid w-[250px] grid-cols-3">
              <TabsTrigger value="daily">Daily</TabsTrigger>
              <TabsTrigger value="weekly">Weekly</TabsTrigger>
              <TabsTrigger value="hourly">Hourly</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        <CardDescription>
          Current market conditions and regime classification
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Aggregate market regime */}
          <div className="bg-muted/30 rounded-lg p-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-sm font-medium">Current Market Regime</h3>
              <Badge variant={confidence > 70 ? "default" : confidence > 50 ? "outline" : "secondary"}>
                {confidence}% Confidence
              </Badge>
            </div>
            <h2 className="text-2xl font-bold mb-1 capitalize">
              {primary_regime?.replace(/_/g, ' ') || 'Unknown'}
            </h2>
            <Progress value={confidence} className="h-1.5 mb-2" />
            
            {secondary_regimes && secondary_regimes.length > 0 && (
              <div className="mt-2">
                <p className="text-xs text-muted-foreground mb-1">Secondary Regimes:</p>
                <div className="flex flex-wrap gap-1">
                  {secondary_regimes.map((regime: any) => (
                    <Badge key={regime.regime} variant="outline" className="capitalize">
                      {regime.regime.replace(/_/g, ' ')} ({regime.probability}%)
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Recommended strategies */}
          {recommended_strategies && recommended_strategies.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium mb-2">Recommended Strategies</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {recommended_strategies.map((strategy: string) => (
                  <div key={strategy} className="bg-muted/20 rounded p-2 text-xs border border-muted">
                    {strategy}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Individual symbols if any */}
          {Object.keys(symbol_regimes).length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium mb-2">Symbol Regimes</h3>
              <div className="grid grid-cols-1 gap-2">
                {Object.entries(symbol_regimes).map(([symbol, data]: [string, any]) => (
                  <div key={symbol} className="flex items-center justify-between p-2 bg-muted/20 rounded">
                    <div className="font-medium">{symbol}</div>
                    <div className="flex items-center">
                      <Badge 
                        className="capitalize mr-2" 
                        variant="outline"
                        style={{
                          backgroundColor: `var(--${regimeColors[data.primary_regime || 'unknown']})`,
                          color: 'white',
                          opacity: 0.9
                        }}
                      >
                        {data.primary_regime?.replace(/_/g, ' ') || 'Unknown'}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {data.confidence}% conf.
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
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

export default MarketRegimeDisplay;
