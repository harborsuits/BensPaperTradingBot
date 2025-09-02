import React from 'react';
import { TrendingUp, TrendingDown, BarChart3, Globe, Signal } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { useWebSocketChannel, useWebSocketMessage } from '@/services/websocket';
import { Position } from '@/types/api.types';
import { Badge } from '@/components/ui/Badge';

interface MarketSignal {
  symbol: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  volume_trend: 'increasing' | 'decreasing' | 'stable';
  momentum: number; // -1 to 1
  volatility: number; // 0 to 1
  timestamp: string;
  market_regime: string;
}

interface MarketContextCardProps {
  className?: string;
  positions: Position[];
  account: 'paper' | 'live';
}

const MarketContextCard: React.FC<MarketContextCardProps> = ({ className, positions, account }) => {
  const [marketSignals, setMarketSignals] = React.useState<Record<string, MarketSignal>>({});
  const [lastUpdated, setLastUpdated] = React.useState<Date | null>(null);
  
  // Subscribe to market context channel
  useWebSocketChannel('market_context' as any, true);
  
  // Handle real-time market context updates
  useWebSocketMessage<any>(
    'market_signals_update',
    (message) => {
      if (message.data && Array.isArray(message.data.signals)) {
        // Filter signals for the current account if account-specific data is provided
        const signals = message.data.account === account 
          ? message.data.signals 
          : message.data.signals;
          
        const newSignals: Record<string, MarketSignal> = {};
        signals.forEach((signal: MarketSignal) => {
          newSignals[signal.symbol] = signal;
        });
        
        setMarketSignals(prev => ({ ...prev, ...newSignals }));
        setLastUpdated(new Date());
      }
    },
    [account]
  );
  
  // Filter to only show signals for current positions
  const positionSymbols = positions.map(p => p.symbol);
  const relevantSignals = Object.values(marketSignals)
    .filter(signal => positionSymbols.includes(signal.symbol));
  
  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return <TrendingUp className="text-bull" size={16} />;
      case 'bearish': return <TrendingDown className="text-bear" size={16} />;
      default: return <BarChart3 className="text-muted-foreground" size={16} />;
    }
  };
  
  const getMomentumColor = (momentum: number) => {
    if (momentum > 0.3) return 'text-bull';
    if (momentum < -0.3) return 'text-bear';
    return 'text-muted-foreground';
  };
  
  const getVolatilityClass = (volatility: number) => {
    if (volatility > 0.7) return 'bg-red-500/10 text-red-500';
    if (volatility > 0.4) return 'bg-amber-500/10 text-amber-500';
    return 'bg-green-500/10 text-green-500';
  };
  
  // If we have no positions or no signals, show a placeholder message
  if (positions.length === 0 || relevantSignals.length === 0) {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center">
            <Signal className="mr-2 h-4 w-4" />
            Market Context
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-6 text-center">
            <Globe className="h-8 w-8 text-muted-foreground mb-2" />
            <p className="text-muted-foreground text-sm">
              {positions.length === 0
                ? "Add positions to see market context"
                : "Waiting for market signals..."}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center">
          <Signal className="mr-2 h-4 w-4" />
          Market Context
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {relevantSignals.map((signal) => (
            <div key={signal.symbol} className="border-b pb-3 last:border-0 last:pb-0">
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center">
                  {getSentimentIcon(signal.sentiment)}
                  <span className="ml-2 font-medium">{signal.symbol}</span>
                </div>
                <Badge variant="outline" className={getVolatilityClass(signal.volatility)}>
                  Vol: {(signal.volatility * 100).toFixed(0)}%
                </Badge>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Sentiment:</span>
                  <span className={
                    signal.sentiment === 'bullish' ? 'text-bull' : 
                    signal.sentiment === 'bearish' ? 'text-bear' : 
                    'text-muted-foreground'
                  }>
                    {signal.sentiment.charAt(0).toUpperCase() + signal.sentiment.slice(1)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Volume:</span>
                  <span>{signal.volume_trend.charAt(0).toUpperCase() + signal.volume_trend.slice(1)}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Momentum:</span>
                  <span className={getMomentumColor(signal.momentum)}>
                    {signal.momentum > 0 ? '+' : ''}{signal.momentum.toFixed(2)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Regime:</span>
                  <span>{signal.market_regime}</span>
                </div>
              </div>
            </div>
          ))}
          
          {lastUpdated && (
            <div className="text-xs text-muted-foreground pt-2 border-t border-border mt-2">
              Updated: {lastUpdated.toLocaleTimeString()}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default MarketContextCard;
