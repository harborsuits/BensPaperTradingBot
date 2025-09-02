import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ArrowUpRight, ArrowDownRight, RefreshCw, Activity, TrendingUp } from 'lucide-react';
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

const AutoLoopActivityCard: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [lastTradeId, setLastTradeId] = useState<string | null>(null);
  const [stats, setStats] = useState({
    buyCount: 0,
    sellCount: 0,
    avgPrice: 0,
    avgQty: 0,
    totalTrades: 0,
    tradesPerMinute: 0
  });
  
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
      setTrades(data.items.slice(0, 10));
      
      // Calculate stats
      const buyTrades = data.items.filter(t => t.side.toLowerCase() === 'buy');
      const sellTrades = data.items.filter(t => t.side.toLowerCase() === 'sell');
      
      const totalPrice = data.items.reduce((sum, t) => sum + t.price, 0);
      const totalQty = data.items.reduce((sum, t) => sum + t.qty, 0);
      
      // Calculate trades per minute (if we have timestamps)
      let tradesPerMinute = 0;
      if (data.items.length >= 2) {
        try {
          const newest = new Date(data.items[0].ts).getTime();
          const oldest = new Date(data.items[data.items.length - 1].ts).getTime();
          const minutesDiff = (newest - oldest) / (1000 * 60);
          if (minutesDiff > 0) {
            tradesPerMinute = data.items.length / minutesDiff;
          }
        } catch (e) {
          console.error('Error calculating trades per minute:', e);
        }
      }
      
      setStats({
        buyCount: buyTrades.length,
        sellCount: sellTrades.length,
        avgPrice: data.items.length ? totalPrice / data.items.length : 0,
        avgQty: data.items.length ? totalQty / data.items.length : 0,
        totalTrades: data.items.length,
        tradesPerMinute
      });
    }
  }, [data]);

  // Listen for WebSocket trade updates
  useWebSocketMessage<any>(
    'trade_executed',
    (message) => {
      const trade = message.data;
      if (trade && trade.id) {
        setLastTradeId(trade.id);
        setTrades(prevTrades => {
          // Check if trade already exists
          if (prevTrades.some(pt => pt.id === trade.id)) {
            return prevTrades;
          }
          
          return [trade, ...prevTrades].slice(0, 10);
        });
        
        // Invalidate trades query to refresh data
        refetch();
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

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <button 
          onClick={() => refetch()} 
          className="p-1 rounded hover:bg-muted"
          aria-label="Refresh"
        >
          <RefreshCw size={16} />
        </button>
        <span className="text-xs text-neutral-400">
          {trades.length > 0 ? `Last updated: ${formatTime(trades[0].ts)}` : ''}
        </span>
      </div>
      
      <div className="flex flex-wrap gap-4">
        {/* Stats Section */}
        <div className="flex-1 min-w-[300px]">
          <div className="grid grid-cols-2 gap-4 h-full">
            <div className="border border-neutral-800 rounded-md p-3">
              <div className="text-xs text-neutral-400 mb-1">Buy Orders</div>
              <div className="text-xl font-bold flex items-center gap-1 text-emerald-400">
                <ArrowUpRight size={16} />
                {stats.buyCount}
              </div>
            </div>
            
            <div className="border border-neutral-800 rounded-md p-3">
              <div className="text-xs text-neutral-400 mb-1">Sell Orders</div>
              <div className="text-xl font-bold flex items-center gap-1 text-rose-400">
                <ArrowDownRight size={16} />
                {stats.sellCount}
              </div>
            </div>
            
            <div className="border border-neutral-800 rounded-md p-3">
              <div className="text-xs text-neutral-400 mb-1">Avg Order Size</div>
              <div className="text-xl font-bold">
                {stats.avgQty.toFixed(2)}
              </div>
            </div>
            
            <div className="border border-neutral-800 rounded-md p-3">
              <div className="text-xs text-neutral-400 mb-1">Trades/Min</div>
              <div className="text-xl font-bold">
                {stats.tradesPerMinute.toFixed(1)}
              </div>
            </div>
          </div>
        </div>
        
        {/* Recent Trades Section */}
        <div className="flex-1 min-w-[300px]">
          <div className="mb-2">
            <h4 className="text-sm font-medium text-neutral-400">Recent Trades</h4>
          </div>
          
          {isLoading ? (
            <div className="space-y-2">
              {[1, 2].map(i => (
                <div key={i} className="h-12 bg-neutral-800/50 animate-pulse rounded-md"></div>
              ))}
            </div>
          ) : error ? (
            <div className="p-4 text-center text-rose-400 border border-rose-800/50 rounded-md">
              Error loading trades
            </div>
          ) : trades.length === 0 ? (
            <div className="p-4 text-center text-neutral-400 border border-neutral-800 rounded-md">
              <TrendingUp className="mx-auto h-6 w-6 mb-2 opacity-50" />
              <p className="text-sm">No trades recorded yet</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
              {trades.slice(0, 4).map((trade) => (
                <div 
                  key={trade.id} 
                  className={`p-2 border rounded-md flex items-center justify-between
                    ${trade.id === lastTradeId ? 'border-primary bg-primary/5 animate-pulse' : 'border-neutral-800'}`}
                >
                  <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-8 rounded-sm ${trade.side.toLowerCase() === 'buy' ? 'bg-emerald-500' : 'bg-rose-500'}`}></div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{trade.symbol}</span>
                        <span className={`text-xs px-1.5 py-0.5 rounded-full ${trade.side.toLowerCase() === 'buy' ? 'bg-emerald-900/30 text-emerald-300' : 'bg-rose-900/30 text-rose-300'}`}>
                          {trade.side}
                        </span>
                      </div>
                      <div className="text-xs text-neutral-400">
                        {trade.qty} @ ${trade.price.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-neutral-400">{formatTime(trade.ts)}</div>
                    <div className="text-xs">{trade.id.split('_')[1]?.slice(0, 8) || '—'}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AutoLoopActivityCard;

