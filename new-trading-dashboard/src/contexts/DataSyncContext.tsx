import React, { createContext, useContext, useEffect, useState, useCallback, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { createSSEConnection } from '@/services/improvedSSE';
import { sseManager } from '@/services/sseManager';
import { DATA_REFRESH_CONFIG } from '@/config/dataRefreshConfig';
import { useWebSocket } from './WebSocketContext';

interface Portfolio {
  cash: number;
  equity: number;
  day_pnl: number;
  open_pnl: number;
  positions: Position[];
  asOf: string;
}

interface Position {
  symbol: string;
  qty: number;
  avg_cost: number;
  last: number;
  pnl: number;
  market_value?: number;
}

interface Account {
  total_equity: number;
  total_cash: number;
  market_value: number;
}

interface Quote {
  symbol: string;
  last: number;
  bid: number;
  ask: number;
  volume: number;
  change: number;
  change_percentage: number;
}

interface DataSyncState {
  // Core data
  portfolio: Portfolio | null;
  account: Account | null;
  positions: Position[];
  quotes: Map<string, Quote>;
  
  // Trading data
  openOrders: any[];
  recentTrades: any[];
  decisions: any[];
  
  // Market context
  marketContext: any;
  strategies: any[];
  
  // System status
  isConnected: boolean;
  lastSync: Date | null;
  
  // Methods
  refreshPortfolio: () => Promise<void>;
  refreshAll: () => Promise<void>;
  subscribeToUpdates: (symbols: string[]) => void;
}

const DataSyncContext = createContext<DataSyncState | undefined>(undefined);

export const DataSyncProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = useQueryClient();
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [quotes, setQuotes] = useState<Map<string, Quote>>(new Map());
  const [openOrders, setOpenOrders] = useState<any[]>([]);
  const [recentTrades, setRecentTrades] = useState<any[]>([]);
  const [decisions, setDecisions] = useState<any[]>([]);
  const [marketContext, setMarketContext] = useState<any>(null);
  const [strategies, setStrategies] = useState<any[]>([]);
  const [lastSync, setLastSync] = useState<Date | null>(null);

  // Use existing WebSocket context
  const wsContext = useWebSocket();
  const orderStreamSSE = useMemo(() => 
    sseManager.getConnection(`${window.location.origin}/api/paper/orders/stream`), 
  []);
  
  // Refresh portfolio data
  const refreshPortfolio = useCallback(async () => {
    try {
      // Fetch portfolio summary
      const portfolioResp = await fetch('/api/portfolio/summary');
      const portfolioData = await portfolioResp.json();
      setPortfolio(portfolioData);
      
      // Fetch account details
      const accountResp = await fetch('/api/paper/account');
      const accountData = await accountResp.json();
      setAccount(accountData.balances);
      
      // Fetch positions
      const positionsResp = await fetch('/api/paper/positions');
      const positionsData = await positionsResp.json();
      setPositions(positionsData);
      
      // Update React Query cache
      queryClient.setQueryData(['portfolio', 'summary'], portfolioData);
      queryClient.setQueryData(['paper', 'account'], accountData);
      queryClient.setQueryData(['paper', 'positions'], positionsData);
      
      setLastSync(new Date());
    } catch (error) {
      console.error('[DataSync] Portfolio refresh error:', error);
    }
  }, [queryClient]);
  
  // Refresh all data
  const refreshAll = useCallback(async () => {
    console.log('[DataSync] Refreshing all data...');
    
    // Parallel fetch all data
    const promises = [
      // Portfolio data
      refreshPortfolio(),
      
      // Orders and trades
      fetch('/api/paper/orders?limit=100').then(r => r.json()),
      fetch('/api/trades').then(r => r.json()),
      
      // Decisions and strategies
      fetch('/api/decisions').then(r => r.json()),
      fetch('/api/strategies').then(r => r.json()),
      
      // Market context
      fetch('/api/context').then(r => r.json()),
    ];
    
    try {
      const [, ordersData, tradesData, decisionsData, strategiesData, contextData] = await Promise.all(promises);
      
      // Update state
      setOpenOrders(ordersData.items || ordersData || []);
      setRecentTrades(tradesData.items || tradesData.trades || []);
      setDecisions(decisionsData.items || decisionsData || []);
      setStrategies(strategiesData.items || strategiesData || []);
      setMarketContext(contextData);
      
      // Update React Query cache
      queryClient.setQueryData(['paper', 'orders'], ordersData);
      queryClient.setQueryData(['trades'], tradesData);
      queryClient.setQueryData(['decisions'], decisionsData);
      queryClient.setQueryData(['strategies'], strategiesData);
      queryClient.setQueryData(['context'], contextData);
      
      setLastSync(new Date());
    } catch (error) {
      console.error('[DataSync] Full refresh error:', error);
    }
  }, [queryClient, refreshPortfolio]);
  
  // Subscribe to quote updates for specific symbols
  const subscribeToUpdates = useCallback((symbols: string[]) => {
    if (symbols.length === 0) return;
    
    // Request quotes via REST first
    fetch(`/api/quotes?symbols=${symbols.join(',')}`)
      .then(r => r.json())
      .then(quotesData => {
        const newQuotes = new Map(quotes);
        quotesData.forEach((q: any) => {
          newQuotes.set(q.symbol, {
            symbol: q.symbol,
            last: q.last || q.price || 0,
            bid: q.bid || 0,
            ask: q.ask || 0,
            volume: q.volume || 0,
            change: q.change || 0,
            change_percentage: q.change_percentage || 0,
          });
        });
        setQuotes(newQuotes);
      })
      .catch(err => console.error('[DataSync] Quote fetch error:', err));
  }, [quotes]);
  
  // Initialize WebSocket handlers
  useEffect(() => {
    // WebSocket connection status is managed by WebSocketContext

    // Subscribe to channels when connected
    if (wsContext.isConnected) {
      console.log('[DataSync] WebSocket already connected, subscribing to channels...');
      // Note: WebSocket subscriptions should be handled by WebSocketContext
    }
  }, [wsContext.isConnected]);
  
  // Initialize SSE for order updates
  useEffect(() => {
    orderStreamSSE.on('order_update', (message) => {
      console.log('[DataSync] Order SSE update:', message);
      
      // Update open orders
      setOpenOrders(prev => {
        const existing = prev.find(o => o.order_id === message.data.order_id);
        if (existing) {
          return prev.map(o => o.order_id === message.data.order_id ? message.data : o);
        } else {
          return [message.data, ...prev].slice(0, 100);
        }
      });
      
      // Refresh portfolio if order is filled
      if (message.data.status === 'filled') {
        refreshPortfolio();
      }
      
      // Invalidate queries
      queryClient.invalidateQueries({ queryKey: ['paper', 'orders'] });
    });
    
    orderStreamSSE.connect();
    
    return () => {
      orderStreamSSE.destroy();
    };
  }, [queryClient, refreshPortfolio, orderStreamSSE]);
  
  // Initial data load
  useEffect(() => {
    refreshAll();
  }, [refreshAll]);
  
  // Periodic sync
  useEffect(() => {
    const interval = setInterval(() => {
      refreshAll();
    }, DATA_REFRESH_CONFIG.default.refetchInterval);
    
    return () => clearInterval(interval);
  }, [refreshAll]);
  
  const value: DataSyncState = {
    portfolio,
    account,
    positions,
    quotes,
    openOrders,
    recentTrades,
    decisions,
    marketContext,
    strategies,
    isConnected: wsContext.isConnected,
    lastSync,
    refreshPortfolio,
    refreshAll,
    subscribeToUpdates,
  };
  
  return (
    <DataSyncContext.Provider value={value}>
      {children}
    </DataSyncContext.Provider>
  );
};

export const useDataSync = () => {
  const context = useContext(DataSyncContext);
  if (!context) {
    throw new Error('useDataSync must be used within DataSyncProvider');
  }
  return context;
};
