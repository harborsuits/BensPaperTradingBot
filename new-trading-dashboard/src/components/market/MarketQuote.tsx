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
  const { data, isLoading, refetch } = useQuote(symbol);
  
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
  
  if (!data) {
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
  
  const last = Number((data as any)?.last ?? (data as any)?.price ?? 0);
  const bid = Number((data as any)?.bid ?? 0);
  const ask = Number((data as any)?.ask ?? 0);
  const prevClose = Number((data as any)?.prevClose ?? (data as any)?.previousClose ?? 0);
  const price = last || (bid && ask ? (bid + ask) / 2 : bid || ask || 0);
  const change = prevClose ? price - prevClose : 0;
  const changePercent = prevClose ? (change / prevClose) * 100 : 0;
  const direction = change >= 0 ? 'up' : 'down';
  
  return (
    <Card className={className}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold">{symbol}</h3>
            <p className="text-sm text-muted-foreground">
              Bid: {fmtNum(bid, 2)} Ask: {fmtNum(ask, 2)}
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
