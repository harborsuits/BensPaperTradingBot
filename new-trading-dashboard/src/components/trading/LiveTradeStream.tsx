import React, { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowUpRight, ArrowDownRight, RefreshCw } from 'lucide-react';
import { useWebSocketMessage } from '@/services/websocket';

interface Trade {
  id: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  status: string;
  ts: string;
  strategy_id?: string;
}

const LiveTradeStream: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [lastTradeId, setLastTradeId] = useState<string | null>(null);
  const queryClient = useQueryClient();
  
  // Fetch initial trades
  const { data, isLoading, error, refetch } = useQuery<{items: Trade[]}>({
    queryKey: ['trades'],
    queryFn: async () => {
      const res = await fetch('/api/trades');
      if (!res.ok) throw new Error('Failed to fetch trades');
      return res.json();
    },
    refetchInterval: 10000,
    staleTime: 5000,
  });

  // Update trades when data changes
  useEffect(() => {
    if (data?.items && data.items.length > 0) {
      setTrades(prevTrades => {
        // If we have no trades yet, use all the data
        if (prevTrades.length === 0) {
          return data.items.slice(0, 10);
        }
        
        // Otherwise, find new trades
        const newTrades = data.items.filter(
          trade => !prevTrades.some(pt => pt.id === trade.id)
        );
        
        if (newTrades.length > 0) {
          setLastTradeId(newTrades[0].id);
          return [...newTrades, ...prevTrades].slice(0, 10);
        }
        
        return prevTrades;
      });
    }
  }, [data]);

  // Listen for WebSocket trade updates
  useWebSocketMessage<any>(
    'trade_executed',
    (message) => {
      const trade = message.data;
      if (trade && trade.id) {
        setTrades(prevTrades => {
          // Check if trade already exists
          if (prevTrades.some(pt => pt.id === trade.id)) {
            return prevTrades;
          }
          
          setLastTradeId(trade.id);
          return [trade, ...prevTrades].slice(0, 10);
        });
        
        // Invalidate trades query to refresh data
        queryClient.invalidateQueries(['trades']);
      }
    },
    []
  );

  // Format timestamp
  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString();
    } catch {
      return '—';
    }
  };

  if (isLoading) {
    return (
      <div className="p-4 border rounded-lg bg-card border-border animate-pulse">
        <div className="h-6 w-1/3 bg-muted rounded mb-4"></div>
        <div className="space-y-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-16 bg-muted rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 border rounded-lg bg-card border-border">
        <div className="text-rose-400 flex items-center justify-between">
          <span>Error loading trades</span>
          <button 
            onClick={() => refetch()} 
            className="p-1 rounded hover:bg-muted"
            aria-label="Retry"
          >
            <RefreshCw size={16} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 border rounded-lg bg-card border-border">
      <h3 className="text-lg font-semibold mb-4 flex items-center justify-between">
        <span>Live Trade Stream</span>
        <button 
          onClick={() => refetch()} 
          className="p-1 rounded hover:bg-muted"
          aria-label="Refresh"
        >
          <RefreshCw size={16} />
        </button>
      </h3>
      
      <div className="space-y-2">
        {trades.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            No trades yet
          </div>
        ) : (
          trades.map((trade, index) => (
            <div 
              key={trade.id} 
              className={`p-3 border rounded-md ${trade.id === lastTradeId ? 'border-primary bg-primary/5 animate-pulse' : 'border-border'}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{trade.symbol}</span>
                  <span className={`flex items-center text-sm ${trade.side.toLowerCase() === 'buy' ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {trade.side.toLowerCase() === 'buy' ? (
                      <ArrowUpRight size={16} className="mr-1" />
                    ) : (
                      <ArrowDownRight size={16} className="mr-1" />
                    )}
                    {trade.side}
                  </span>
                </div>
                <span className="text-sm text-muted-foreground">
                  {formatTime(trade.ts)}
                </span>
              </div>
              
              <div className="mt-1 flex items-center justify-between">
                <div className="text-sm">
                  <span className="text-muted-foreground mr-2">Qty:</span>
                  <span className="font-medium">{trade.qty}</span>
                </div>
                <div className="text-sm">
                  <span className="text-muted-foreground mr-2">Price:</span>
                  <span className="font-medium">${trade.price.toFixed(2)}</span>
                </div>
                <div className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                  {trade.strategy_id || 'auto'}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default LiveTradeStream;

