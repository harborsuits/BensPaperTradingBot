import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, BarChart2, Zap, HelpCircle, RefreshCw } from 'lucide-react';
import { MarketRegimeData, contextApi } from '@/services/contextApi';
import { Card, CardContent } from '@/components/ui/Card';
import { useContextUpdates } from '@/hooks/useWebSocketSubscriptions';
import { toast } from 'react-hot-toast';
import styles from './MarketRegimeIndicator.module.css';
import { safeSplit, toString } from '@/utils/safe';
import ZeroState from '@/components/ui/ZeroState';

interface MarketRegimeIndicatorProps {
  className?: string;
}

const RegimeIcons = {
  bullish: <TrendingUp className="text-bull" size={24} />,
  bearish: <TrendingDown className="text-bear" size={24} />,
  neutral: <BarChart2 className="text-muted-foreground" size={24} />,
  volatile: <Zap className="text-warning" size={24} />,
  recovery: <TrendingUp className="text-info" size={24} />,
  unknown: <HelpCircle className="text-muted" size={24} />
};

const RegimeColors = {
  bullish: 'bg-bull/10 text-bull border-bull/20',
  bearish: 'bg-bear/10 text-bear border-bear/20',
  neutral: 'bg-muted/30 text-muted-foreground border-muted',
  volatile: 'bg-warning/10 text-warning border-warning/20',
  recovery: 'bg-info/10 text-info border-info/20',
  unknown: 'bg-muted/20 text-muted-foreground border-muted'
};

const MarketRegimeIndicator: React.FC<MarketRegimeIndicatorProps> = ({ className }) => {
  const queryClient = useQueryClient();
  
  const { 
    data, 
    isLoading, 
    error,
    refetch
  } = useQuery({
    queryKey: ['market-regime'],
    queryFn: async () => {
      try {
        const response = await contextApi.getMarketRegime();
        if (!response.success) {
          throw new Error(response.error || 'Failed to fetch market regime');
        }
        return response.data;
      } catch (err: any) {
        console.error('Error fetching market regime:', err);
        toast.error(`Failed to load market regime: ${err.message || 'Unknown error'}`);
        throw err;
      }
    },
    staleTime: 60000, // 1 minute
    retry: 2,
  });
  
  // Subscribe to market context WebSocket channel for regime changes
  useContextUpdates(
    (regimeData) => {
      // Update the query cache with new regime data
      queryClient.setQueryData(['market-regime'], regimeData);
      
      // Show a toast notification about the regime change
      const regimeChangedToast = `Market regime changed to ${regimeData.regime}`;
      toast(regimeChangedToast, { 
        icon: regimeData.regime === 'bullish' ? '📈' : 
              regimeData.regime === 'bearish' ? '📉' : 
              regimeData.regime === 'volatile' ? '⚡' : 'ℹ️' 
      });
    }
  );

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Market Regime</h3>
              <div className="h-8 w-32 bg-muted/20 animate-pulse rounded-md mt-2"></div>
            </div>
            <div className="h-10 w-10 bg-muted/20 animate-pulse rounded-full"></div>
          </div>
          <div className="h-4 w-24 bg-muted/20 animate-pulse rounded-md mt-3"></div>
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Market Regime</h3>
              <p className="text-lg font-bold">Unknown</p>
            </div>
            <HelpCircle className="text-muted" size={24} />
          </div>
          <p className="text-xs text-muted-foreground mt-2">Unable to fetch market regime data</p>
        </CardContent>
      </Card>
    );
  }

  const regime = data as MarketRegimeData;
  const confidencePercent = Math.round((regime.confidence || 0) * 100);
  const regimeType = toString(regime.regime, 'unknown');
  
  // Safely handle the RegimeColors split operation
  const colorClass = RegimeColors[regimeType] || RegimeColors.unknown;
  const primaryColor = safeSplit(colorClass, ' ')[0] || '';
  
  // Check if we have enough data to render properly
  if (!regimeType || !primaryColor) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <ZeroState
            title="No regime data yet"
            message="Waiting on market context analyzer."
            action={{ label: "Refresh", onClick: () => refetch() }}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Market Regime</h3>
            <p className="text-lg font-bold capitalize">{regimeType}</p>
          </div>
          {RegimeIcons[regimeType] || RegimeIcons.unknown}
        </div>
        <div className="mt-2">
          <div className="flex justify-between text-xs mb-1">
            <span>{confidencePercent}% confidence</span>
            <span className="text-muted-foreground">
              Since {regime.since ? new Date(regime.since).toLocaleDateString() : 'unknown'}
            </span>
          </div>
          <div className={styles.confidenceBar}>
            <div 
              className={`${primaryColor} ${styles.confidenceFill} ${styles[`confidence${Math.floor(confidencePercent / 10) * 10}`] || styles.confidence50}`} 
              data-testid="confidence-bar"
            ></div>
          </div>
        </div>
        <p className="text-xs mt-3">{regime.description || 'No description available'}</p>
        
        {/* Refresh button */}
        <button 
          onClick={() => { 
            refetch();
            toast.success('Refreshing market data...');
          }}
          className="absolute top-3 right-3 p-1 rounded-full bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
          title="Refresh market regime data"
        >
          <RefreshCw size={14} />
        </button>
      </CardContent>
    </Card>
  );
};

export default MarketRegimeIndicator;
