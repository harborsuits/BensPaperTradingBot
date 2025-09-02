import React, { useMemo } from 'react';
import { useBars, Timeframe } from '@/hooks/useBars';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';
import { ZeroState } from '@/components/ui/ZeroState';
import { RefreshCw } from 'lucide-react';
import { Bar } from '@/schemas/market';

interface MarketChartProps {
  symbol: string;
  timeframe?: Timeframe;
  limit?: number;
  height?: number;
  title?: string;
  className?: string;
}

export const MarketChart: React.FC<MarketChartProps> = ({
  symbol,
  timeframe = '1Day',
  limit = 30,
  height = 200,
  title,
  className
}) => {
  const { data, isLoading, refetch } = useBars(symbol, timeframe, { limit });
  
  const chartData = useMemo(() => {
    if (!data?.bars || data.bars.length === 0) return null;
    
    // Find min/max for scaling
    let minPrice = Infinity;
    let maxPrice = -Infinity;
    
    data.bars.forEach(bar => {
      minPrice = Math.min(minPrice, bar.l);
      maxPrice = Math.max(maxPrice, bar.h);
    });
    
    // Add some padding
    const range = maxPrice - minPrice;
    minPrice = minPrice - range * 0.05;
    maxPrice = maxPrice + range * 0.05;
    
    return {
      bars: data.bars,
      minPrice,
      maxPrice,
      range: maxPrice - minPrice
    };
  }, [data]);
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">{title || `${symbol} Chart`}</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <Skeleton className="h-[200px] w-full" />
        </CardContent>
      </Card>
    );
  }
  
  if (!chartData || !data?.bars || data.bars.length === 0) {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">{title || `${symbol} Chart`}</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <ZeroState
            title={`No data for ${symbol}`}
            message="Unable to fetch price history"
            action={{
              label: "Retry",
              onClick: () => refetch()
            }}
          />
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center justify-between">
          <span>{title || `${symbol} Chart (${timeframe})`}</span>
          {data.stale && <span className="text-xs text-muted-foreground">(Stale data)</span>}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <div style={{ height: `${height}px`, position: 'relative' }}>
          <svg width="100%" height="100%" viewBox={`0 0 ${chartData.bars.length} ${height}`}>
            {/* Price bars */}
            {chartData.bars.map((bar, i) => (
              <g key={i} transform={`translate(${i}, 0)`}>
                {/* Candle body */}
                <rect
                  x={0.2}
                  y={scalePrice(Math.max(bar.o, bar.c), chartData.minPrice, chartData.maxPrice, height)}
                  width={0.6}
                  height={Math.abs(
                    scalePrice(bar.o, chartData.minPrice, chartData.maxPrice, height) - 
                    scalePrice(bar.c, chartData.minPrice, chartData.maxPrice, height)
                  ) || 1}
                  fill={bar.c >= bar.o ? '#22c55e' : '#ef4444'}
                />
                
                {/* High/low wicks */}
                <line
                  x1={0.5}
                  y1={scalePrice(bar.h, chartData.minPrice, chartData.maxPrice, height)}
                  x2={0.5}
                  y2={scalePrice(Math.max(bar.o, bar.c), chartData.minPrice, chartData.maxPrice, height)}
                  stroke={bar.c >= bar.o ? '#22c55e' : '#ef4444'}
                  strokeWidth={0.1}
                />
                <line
                  x1={0.5}
                  y1={scalePrice(Math.min(bar.o, bar.c), chartData.minPrice, chartData.maxPrice, height)}
                  x2={0.5}
                  y2={scalePrice(bar.l, chartData.minPrice, chartData.maxPrice, height)}
                  stroke={bar.c >= bar.o ? '#22c55e' : '#ef4444'}
                  strokeWidth={0.1}
                />
              </g>
            ))}
          </svg>
          
          {/* Price labels */}
          <div className="absolute top-0 right-0 text-xs text-muted-foreground">
            {chartData.maxPrice.toFixed(2)}
          </div>
          <div className="absolute bottom-0 right-0 text-xs text-muted-foreground">
            {chartData.minPrice.toFixed(2)}
          </div>
          
          {/* Refresh button */}
          <button 
            onClick={() => refetch()}
            className="absolute top-0 right-12 p-1 rounded-full bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
            title="Refresh chart data"
          >
            <RefreshCw size={12} />
          </button>
        </div>
      </CardContent>
    </Card>
  );
};

// Helper function to scale price to chart coordinates
function scalePrice(price: number, min: number, max: number, height: number): number {
  return height - ((price - min) / (max - min) * height);
}
