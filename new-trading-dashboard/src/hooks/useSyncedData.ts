import { useEffect } from 'react';
import { useDataSync } from '@/contexts/DataSyncContext';
import { useQuery } from '@tanstack/react-query';
import { DATA_REFRESH_CONFIG } from '@/config/dataRefreshConfig';

// Hook for portfolio data
export function useSyncedPortfolio() {
  const { portfolio } = useDataSync();
  
  const query = useQuery({
    queryKey: ['portfolio', 'summary'],
    queryFn: async () => {
      const response = await fetch('/api/paper/account');
      const data = await response.json();
      
      // Transform the API response to match expected format
      if (data?.balances) {
        return {
          cash: data.balances.total_cash || 0,
          cash_balance: data.balances.total_cash || 0,
          equity: data.balances.total_equity || 0,
          total_equity: data.balances.total_equity || 0,
          day_pnl: data.balances.close_pl || 0,
          daily_pl: data.balances.close_pl || 0,
          open_pnl: data.balances.open_pl || 0,
          unrealized_pl: data.balances.open_pl || 0,
          positions: [],
          broker: 'tradier',
          mode: 'paper',
          asOf: new Date().toISOString(),
          ...data.meta
        };
      }
      return data;
    },
    initialData: portfolio,
    refetchInterval: DATA_REFRESH_CONFIG.paperAccount?.refetchInterval || 15000,
    staleTime: DATA_REFRESH_CONFIG.paperAccount?.staleTime || 10000,
  });
  
  return {
    ...query,
    data: query.data || portfolio,
  };
}

// Hook for account data
export function useSyncedAccount() {
  const { account } = useDataSync();
  
  return useQuery({
    queryKey: ['paper', 'account'],
    queryFn: async () => {
      const response = await fetch('/api/paper/account');
      const data = await response.json();
      return data;
    },
    initialData: account ? { balances: account } : undefined,
    refetchInterval: DATA_REFRESH_CONFIG.paperAccount?.refetchInterval || 15000,
    staleTime: DATA_REFRESH_CONFIG.paperAccount?.staleTime || 10000,
  });
}

// Hook for positions
export function useSyncedPositions() {
  const { positions } = useDataSync();
  
  return useQuery({
    queryKey: ['paper', 'positions'],
    queryFn: async () => {
      const response = await fetch('/api/paper/positions');
      const data = await response.json();
      // Handle both array and object with items property
      return Array.isArray(data) ? data : (data?.items || data?.positions || []);
    },
    initialData: positions,
    refetchInterval: DATA_REFRESH_CONFIG.paperPositions?.refetchInterval || 15000,
    staleTime: DATA_REFRESH_CONFIG.paperPositions?.staleTime || 10000,
  });
}

// Hook for quotes
export function useSyncedQuotes(symbols: string[]) {
  const { quotes, subscribeToUpdates } = useDataSync();
  
  // Subscribe to updates for these symbols
  useEffect(() => {
    if (symbols.length > 0) {
      subscribeToUpdates(symbols);
    }
  }, [symbols, subscribeToUpdates]);
  
  const symbolQuotes = symbols.map(symbol => quotes.get(symbol)).filter(Boolean);
  
  return useQuery({
    queryKey: ['quotes', ...symbols],
    queryFn: async () => {
      const response = await fetch(`/api/quotes?symbols=${symbols.join(',')}`);
      return response.json();
    },
    initialData: symbolQuotes.length > 0 ? symbolQuotes : undefined,
    refetchInterval: DATA_REFRESH_CONFIG.quotes.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.quotes.staleTime,
  });
}

// Hook for open orders
export function useSyncedOrders() {
  const { openOrders } = useDataSync();
  
  return useQuery({
    queryKey: ['paper', 'orders'],
    queryFn: async () => {
      const response = await fetch('/api/paper/orders?limit=100');
      return response.json();
    },
    initialData: openOrders,
    refetchInterval: DATA_REFRESH_CONFIG.orders.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.orders.staleTime,
  });
}

// Hook for recent trades
export function useSyncedTrades() {
  const { recentTrades } = useDataSync();
  
  return useQuery({
    queryKey: ['trades'],
    queryFn: async () => {
      const response = await fetch('/api/trades');
      const data = await response.json();
      // Ensure we return data in expected format: { items: [...] }
      if (Array.isArray(data)) {
        return { items: data };
      }
      return data;
    },
    initialData: recentTrades ? { items: recentTrades } : undefined,
    refetchInterval: DATA_REFRESH_CONFIG.trades.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.trades.staleTime,
  });
}

// Hook for decisions
export function useSyncedDecisions() {
  const { decisions } = useDataSync();
  
  return useQuery({
    queryKey: ['decisions', 'recent'],
    queryFn: async () => {
      const response = await fetch('/api/decisions/recent');
      const data = await response.json();
      // Handle both array and object with items property
      if (Array.isArray(data)) {
        return data;
      }
      return data?.items || [];
    },
    initialData: decisions,
    refetchInterval: DATA_REFRESH_CONFIG.decisions.refetchInterval || 7000, // 7-10s as per spec
    staleTime: DATA_REFRESH_CONFIG.decisions.staleTime || 5000,
  });
}

// Hook for strategies
export function useSyncedStrategies() {
  const { strategies } = useDataSync();
  
  return useQuery({
    queryKey: ['strategies', 'active'],
    queryFn: async () => {
      const response = await fetch('/api/strategies/active');
      const data = await response.json();
      // Handle both array and object with items property + asOf ISO
      if (Array.isArray(data)) {
        return data;
      }
      return data?.items || [];
    },
    initialData: strategies,
    refetchInterval: DATA_REFRESH_CONFIG.strategies.refetchInterval || 60000,
    staleTime: DATA_REFRESH_CONFIG.strategies.staleTime || 45000,
  });
}

// Hook for market context
export function useSyncedContext() {
  const { marketContext } = useDataSync();
  
  return useQuery({
    queryKey: ['context'],
    queryFn: async () => {
      const response = await fetch('/api/context');
      return response.json();
    },
    initialData: marketContext,
    refetchInterval: DATA_REFRESH_CONFIG.context.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.context.staleTime,
  });
}

// Hook to get sync status
export function useSyncStatus() {
  const { isConnected, lastSync } = useDataSync();
  
  return {
    isConnected,
    lastSync,
    timeSinceSync: lastSync ? Date.now() - lastSync.getTime() : null,
  };
}

// Hook for health data
export function useSyncedHealth() {
  const { isConnected } = useDataSync();
  
  return useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      try {
        // Try JSON metrics endpoint first
        const metricsResponse = await fetch('/api/metrics');
        if (metricsResponse.ok) {
          const metrics = await metricsResponse.json();
          // Convert metrics format to health format
          return {
            breaker: 'GREEN', // Default to GREEN if metrics are available
            marketData: {
              ageSec: metrics.quotesAge || 0
            },
            broker: {
              rttMs: metrics.averageLatency || 0
            },
            wsConnected: isConnected,
            asOf: metrics.asOf || new Date().toISOString(),
            quote_age_s: metrics.quotesAge || 0,
            broker_age_s: (metrics.averageLatency || 0) / 1000
          };
        }
      } catch (e) {
        // Metrics endpoint failed, continue to fallback
      }
      
      // Fallback to /api/health
      const healthResponse = await fetch('/api/health');
      const health = await healthResponse.json();
      return {
        ...health,
        wsConnected: isConnected
      };
    },
    refetchInterval: DATA_REFRESH_CONFIG.health?.refetchInterval || 30000,
    staleTime: DATA_REFRESH_CONFIG.health?.staleTime || 20000,
  });
}

// Hook for autoloop status
export function useSyncedAutoloop() {
  return useQuery({
    queryKey: ['autoloop', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/autoloop/status');
      return response.json();
    },
    refetchInterval: 5000, // 5 seconds for critical status
    staleTime: 3000,
  });
}

// Hook for pipeline health
export function useSyncedPipelineHealth(window = '15m') {
  return useQuery({
    queryKey: ['brain', 'flow', 'summary', window],
    queryFn: async () => {
      try {
        const response = await fetch(`/api/brain/flow/summary?window=${window}`);
        if (!response.ok) {
          console.error('Pipeline health fetch failed:', response.status, response.statusText);
          return null;
        }
        const data = await response.json();
        // Extract the pipeline data, handling the wrapper fields
        return {
          total_scores: data.total_scores || 0,
          avg_score: data.avg_score || 0,
          high_confidence: data.high_confidence || 0,
          by_symbol: data.by_symbol || {},
          unique_symbols: Object.keys(data.by_symbol || {}).length,
          asOf: data.asof_ts || data.timestamp || new Date().toISOString()
        };
      } catch (error) {
        console.error('Error fetching pipeline health:', error);
        return null;
      }
    },
    refetchInterval: 10000, // 10 seconds
    staleTime: 5000,
    retry: 2,
  });
}

// Hook for decisions summary
export function useSyncedDecisionsSummary(window = '15m') {
  return useQuery({
    queryKey: ['decisions', 'summary', window],
    queryFn: async () => {
      const response = await fetch(`/api/decisions/summary?window=${window}`);
      return response.json();
    },
    refetchInterval: DATA_REFRESH_CONFIG.decisions.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.decisions.staleTime,
  });
}

// Hook for brain status
export function useSyncedBrainStatus() {
  return useQuery({
    queryKey: ['brain', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/brain/status');
      return response.json();
    },
    refetchInterval: 15000, // 15 seconds
    staleTime: 10000,
  });
}

// Hook for evo status
export function useSyncedEvoStatus() {
  return useQuery({
    queryKey: ['evo', 'status'],
    queryFn: async () => {
      const response = await fetch('/api/evo/status');
      return response.json();
    },
    refetchInterval: 15000, // 15 seconds
    staleTime: 10000,
  });
}

// Hook to manually trigger refresh
export function useRefreshData() {
  const { refreshAll, refreshPortfolio } = useDataSync();
  
  return {
    refreshAll,
    refreshPortfolio,
  };
}
