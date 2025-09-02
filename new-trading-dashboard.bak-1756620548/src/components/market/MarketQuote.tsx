import React from 'react';
import { useQuote } from '@/hooks/useQuotes';
import { fmtNum } from '@/utils/number';
import { Card, CardContent } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';
import { ZeroState } from '@/components/ui/ZeroState';
import { ArrowUp, ArrowDown, RefreshCw } from 'lucide-react';

interface MarketQuoteProps {
  symbol: string;
  className?: string;
}

export const MarketQuote: React.FC<MarketQuoteProps> = ({ symbol, className }) => {
  const { quote, isLoading, refetch } = useQuote(symbol);
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-16" />
            <Skeleton className="h-6 w-24" />
          </div>
          <div className="mt-2">
            <Skeleton className="h-4 w-32" />
          </div>
        </CardContent>
      </Card>
    );
  }
  
  if (!quote || !quote.quote) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <ZeroState
            title={`No data for ${symbol}`}
            message="Unable to fetch quote data"
            action={{
              label: "Retry",
              onClick: () => refetch()
            }}
          />
        </CardContent>
      </Card>
    );
  }
  
  const { quote: data } = quote;
  
  // Use mid price (average of bid and ask)
  const price = data.ap && data.bp 
    ? (data.ap + data.bp) / 2 
    : data.ap || data.bp || 0;
  
  // Determine if price is up or down (mock implementation)
  // In a real app, you'd compare to previous close
  const direction = symbol.charCodeAt(0) % 2 === 0 ? 'up' : 'down';
  const change = direction === 'up' ? 0.5 : -0.5;
  const changePercent = direction === 'up' ? 0.8 : -0.8;
  
  return (
    <Card className={className}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold">{symbol}</h3>
            <p className="text-sm text-muted-foreground">
              {quote.stale && "(Stale) "}
              Bid: {fmtNum(data.bp, 2)} Ask: {fmtNum(data.ap, 2)}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xl font-bold">{fmtNum(price, 2)}</p>
            <p className={`text-sm flex items-center ${direction === 'up' ? 'text-bull' : 'text-bear'}`}>
              {direction === 'up' ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
              <span className="ml-1">{fmtNum(change, 2)} ({fmtNum(changePercent, 2)}%)</span>
            </p>
          </div>
        </div>
        
        {/* Refresh button */}
        <button 
          onClick={() => refetch()}
          className="absolute top-3 right-3 p-1 rounded-full bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
          title="Refresh quote data"
        >
          <RefreshCw size={14} />
        </button>
      </CardContent>
    </Card>
  );
};
