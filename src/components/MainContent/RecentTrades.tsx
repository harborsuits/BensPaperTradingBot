import React from 'react';
import { useRecentTrades } from '../../hooks/useRecentTrades';
import { Spinner } from '../ui/Spinner';
import { ErrorMessage } from '../ui/ErrorMessage';

export function RecentTrades() {
  const { data: trades, isLoading, error, refetch } = useRecentTrades(5);
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage retry={() => refetch()} />;
  
  function formatDate(dateString: string) {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'short', 
      day: 'numeric'
    }).format(date);
  }
  
  return (
    <div className="recent-trades-container">
      {trades && trades.length > 0 ? (
        <div className="trades-items space-y-3">
          {trades.map(trade => (
            <div 
              key={trade.id}
              className={`trade-item flex justify-between items-center p-2 rounded
                ${trade.side === 'buy' ? 'bg-primary/10' : 'bg-destructive/10'}`}
            >
              <div>
                <span className="text-sm font-medium block">
                  {trade.side === 'buy' ? 'Buy' : 'Sell'} {trade.symbol}
                </span>
                <span className="text-xs text-muted-foreground">
                  {trade.quantity} shares @ ${trade.price.toLocaleString(undefined, { 
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2 
                  })}
                </span>
              </div>
              <span 
                className={`text-xs px-2 py-1 rounded
                  ${trade.side === 'buy' ? 'bg-primary/20' : 'bg-destructive/20'}`}
              >
                {formatDate(trade.timestamp)}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center justify-center h-32 bg-muted/30 rounded text-muted-foreground">
          No recent trades
        </div>
      )}
    </div>
  );
} 