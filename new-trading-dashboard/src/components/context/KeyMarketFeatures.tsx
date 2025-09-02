import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowUp, ArrowDown, RefreshCw } from 'lucide-react';
import { contextApi } from '@/services/contextApi';
import { useContextUpdates } from '@/hooks/useWebSocketSubscriptions';
import { Progress } from '@/components/ui/Progress';
import { toast } from 'react-hot-toast';

interface KeyMarketFeaturesProps {
  className?: string;
}

interface FeatureItemProps {
  name: string;
  value: number;
  change?: number;
  unit?: string;
  minValue?: number;
  maxValue?: number;
}

interface FeatureData {
  value: number;
  change: number;
}

interface MarketFeatures {
  [key: string]: FeatureData;
}

const FeatureItem: React.FC<FeatureItemProps> = ({ 
  name, 
  value, 
  change, 
  unit = '',
  minValue = -1,
  maxValue = 1
}) => {
  // Calculate normalized value for the progress bar (0-100)
  const normalizedValue = Math.min(100, Math.max(0, 
    ((value - minValue) / (maxValue - minValue)) * 100
  ));
  
  // Determine if the value is positive, negative or neutral
  const valueClass = value > 0 
    ? 'text-bull' 
    : value < 0 
      ? 'text-bear' 
      : 'text-muted-foreground';
  
  // Determine if the change is positive, negative or neutral
  const changeClass = change 
    ? change > 0 
      ? 'text-bull' 
      : 'text-bear' 
    : 'text-muted-foreground';
  
  return (
    <div className="py-3 border-b border-border last:border-0">
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium">{name}</span>
        <div className="flex items-center gap-1">
          <span className={`text-sm font-medium ${valueClass}`}>
            {value.toFixed(2)}{unit}
          </span>
          {change !== undefined && (
            <span className={`text-xs flex items-center ${changeClass}`}>
              {change > 0 ? <ArrowUp size={12} /> : <ArrowDown size={12} />}
              {Math.abs(change).toFixed(2)}%
            </span>
          )}
        </div>
      </div>
      <Progress value={normalizedValue} className="h-1" />
    </div>
  );
};

const KeyMarketFeatures: React.FC<KeyMarketFeaturesProps> = ({ className = '' }) => {
  const queryClient = useQueryClient();
  
  // Query for market features
  const { 
    data: features, 
    isLoading, 
    isError,
    refetch 
  } = useQuery({
    queryKey: ['market-features'],
    queryFn: async () => {
      try {
        const response = await contextApi.getMarketFeatures();
        if (!response.success) {
          throw new Error(response.error || 'Failed to fetch market features');
        }
        return response.data;
      } catch (err: any) {
        console.error('Error fetching market features:', err);
        toast.error(`Failed to load market features: ${err.message || 'Unknown error'}`);
        throw err;
      }
    },
    staleTime: 300000, // 5 minutes
    retry: 2,
  });
  
  // Use the context updates hook for feature updates
  useContextUpdates(
    undefined, // No regime change handler
    (featureData) => {
      // Update the query cache with the new feature data
      queryClient.setQueryData(['market-features'], (oldData: any) => {
        // If we didn't have data before, just use the new data
        if (!oldData) return featureData;
        
        // Otherwise merge the new data with the old data
        return { ...oldData, ...featureData };
      });
      
      // Notify user of significant changes
      if (featureData.volatility && Math.abs(featureData.volatility.change) > 5) {
        toast(`Volatility ${featureData.volatility.change > 0 ? 'increased' : 'decreased'} significantly`, {
          icon: featureData.volatility.change > 0 ? '⚠️' : '✅'
        });
      }
    }
  );
  
  if (isLoading) {
    return (
      <div className={`space-y-4 ${className}`}>
        <p className="text-muted-foreground text-sm">Loading key market features...</p>
        <div className="space-y-6">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-between">
                <div className="h-4 w-24 bg-muted rounded animate-pulse"></div>
                <div className="h-4 w-12 bg-muted rounded animate-pulse"></div>
              </div>
              <div className="h-1 bg-muted rounded animate-pulse"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  if (isError || !features) {
    return (
      <div className={`space-y-2 ${className}`}>
        <p className="text-muted-foreground text-sm">Failed to load market features</p>
        <button 
          className="text-xs text-primary hover:underline"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }
  
  // Default feature data if API returns no data
  const defaultFeatures: MarketFeatures = {
    momentum: { value: 0.65, change: 2.1 },
    volatility: { value: 0.48, change: -1.2 },
    breadth: { value: 0.72, change: 3.5 },
    liquidity: { value: 0.52, change: 0.3 },
    sentiment: { value: 0.38, change: -0.8 },
    fundamentals: { value: 0.61, change: 0.0 },
    technicals: { value: 0.58, change: 1.7 },
    correlation: { value: 0.42, change: -0.5 }
  };
  
  // Use features from API response or fall back to default features
  const marketFeatures = (features as MarketFeatures) || defaultFeatures;
  
  return (
    <div className={className}>
      <div className="relative">  
        {/* Refresh button */}
        <button 
          onClick={() => { 
            refetch();
            toast.success('Refreshing market features...');
          }}
          className="absolute -top-8 right-0 p-1 rounded-full bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
          title="Refresh market features"
        >
          <RefreshCw size={14} />
        </button>
        
        <div className="space-y-0 divide-y divide-border">
          {Object.entries(marketFeatures).map(([key, data]) => (
            <FeatureItem 
              key={key}
              name={key.charAt(0).toUpperCase() + key.slice(1)} // Capitalize first letter
              value={Number.isFinite(Number((data as any)?.value)) ? Number((data as any).value) : 0}
              change={Number.isFinite(Number((data as any)?.change)) ? Number((data as any).change) : undefined}
              minValue={0}
              maxValue={1}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default KeyMarketFeatures;
