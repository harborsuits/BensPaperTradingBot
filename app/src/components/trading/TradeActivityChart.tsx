import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useWebSocketMessage } from '@/services/websocket';

interface TradeActivity {
  timestamp: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
}

const TradeActivityChart: React.FC = () => {
  const [activities, setActivities] = useState<TradeActivity[]>([]);
  const [maxActivities] = useState(20); // Maximum number of activities to display
  
  // Fetch initial trade data
  const { data } = useQuery<{items: any[]}>({
    queryKey: ['trades'],
    queryFn: async () => {
      const res = await fetch('/api/trades');
      if (!res.ok) throw new Error('Failed to fetch trades');
      return res.json();
    },
    refetchInterval: 10000,
  });

  // Initialize with existing trades
  useEffect(() => {
    if (data?.items && data.items.length > 0) {
      const tradeActivities = data.items.slice(0, maxActivities).map(trade => ({
        timestamp: trade.ts,
        symbol: trade.symbol,
        side: trade.side,
        qty: trade.qty,
        price: trade.price
      }));
      
      setActivities(tradeActivities);
    }
  }, [data, maxActivities]);

  // Listen for WebSocket trade updates
  useWebSocketMessage<any>(
    'trade_executed',
    (message) => {
      const trade = message.data;
      if (trade) {
        const newActivity = {
          timestamp: trade.ts || new Date().toISOString(),
          symbol: trade.symbol,
          side: trade.side,
          qty: trade.qty,
          price: trade.price
        };
        
        setActivities(prev => [newActivity, ...prev].slice(0, maxActivities));
      }
    },
    []
  );

  // Calculate time scale
  const timeScale = () => {
    if (activities.length < 2) return [];
    
    const timestamps = activities.map(a => new Date(a.timestamp).getTime());
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);
    const range = maxTime - minTime;
    
    // Generate time scale points
    return Array.from({ length: 5 }, (_, i) => {
      const time = new Date(minTime + (range * i / 4));
      return time.toLocaleTimeString();
    });
  };

  // Calculate position on the timeline
  const getTimePosition = (timestamp: string) => {
    if (activities.length < 2) return '0%';
    
    const time = new Date(timestamp).getTime();
    const timestamps = activities.map(a => new Date(a.timestamp).getTime());
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);
    const range = maxTime - minTime;
    
    if (range === 0) return '0%';
    return `${((time - minTime) / range) * 100}%`;
  };

  return (
    <div className="p-4 border rounded-lg bg-card border-border">
      <h3 className="text-lg font-semibold mb-4">Trade Activity</h3>
      
      {activities.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          No trade activity yet
        </div>
      ) : (
        <div>
          {/* Timeline */}
          <div className="relative h-40 mb-4">
            {/* Time scale */}
            <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-muted-foreground">
              {timeScale().map((time, i) => (
                <div key={i}>{time}</div>
              ))}
            </div>
            
            {/* Activity points */}
            <div className="absolute top-0 left-0 right-0 bottom-6 border-b border-border">
              {activities.map((activity, i) => (
                <div
                  key={i}
                  className="absolute bottom-0 transform -translate-x-1/2"
                  style={{ left: getTimePosition(activity.timestamp) }}
                >
                  <div 
                    className={`w-2 h-2 rounded-full mb-1 ${activity.side.toLowerCase() === 'buy' ? 'bg-emerald-500' : 'bg-rose-500'}`}
                  ></div>
                  <div className="text-xs font-medium whitespace-nowrap">
                    {activity.symbol}
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Activity stats */}
          <div className="grid grid-cols-2 gap-4">
            <div className="border border-border rounded-md p-3">
              <div className="text-sm text-muted-foreground mb-1">Buy Orders</div>
              <div className="text-2xl font-bold">
                {activities.filter(a => a.side.toLowerCase() === 'buy').length}
              </div>
            </div>
            
            <div className="border border-border rounded-md p-3">
              <div className="text-sm text-muted-foreground mb-1">Sell Orders</div>
              <div className="text-2xl font-bold">
                {activities.filter(a => a.side.toLowerCase() === 'sell').length}
              </div>
            </div>
            
            <div className="border border-border rounded-md p-3">
              <div className="text-sm text-muted-foreground mb-1">Avg Order Size</div>
              <div className="text-2xl font-bold">
                {activities.length > 0 
                  ? (activities.reduce((sum, a) => sum + a.qty, 0) / activities.length).toFixed(2)
                  : '0.00'}
              </div>
            </div>
            
            <div className="border border-border rounded-md p-3">
              <div className="text-sm text-muted-foreground mb-1">Avg Price</div>
              <div className="text-2xl font-bold">
                ${activities.length > 0 
                  ? (activities.reduce((sum, a) => sum + a.price, 0) / activities.length).toFixed(2)
                  : '0.00'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradeActivityChart;

