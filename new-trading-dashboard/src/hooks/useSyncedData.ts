import { useEffect } from 'react';
import { useDataSync } from '@/contexts/DataSyncContext';
import { useQuery } from '@tanstack/react-query';
import { DATA_REFRESH_CONFIG } from '@/config/dataRefreshConfig';

// Hook for portfolio data
export function useSyncedPortfolio() {
  const { portfolio, refreshPortfolio } = useDataSync();
  
  const query = useQuery({
    queryKey: ['portfolio', 'summary'],
    queryFn: async () => {
      if (!portfolio) {
        await refreshPortfolio();
      }
      return portfolio;
    },
    initialData: portfolio,
    refetchInterval: DATA_REFRESH_CONFIG.paperAccount.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.paperAccount.staleTime,
  });
  
  return {
    ...query,
    data: portfolio || query.data,
  };
}

// Hook for account data
export function useSyncedAccount() {
  const { account } = useDataSync();
  
  return useQuery({
    queryKey: ['paper', 'account'],
    queryFn: async () => {
      const response = await fetch('/api/paper/account');
      return response.json();
    },
    initialData: account ? { balances: account } : undefined,
    refetchInterval: DATA_REFRESH_CONFIG.paperAccount.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.paperAccount.staleTime,
  });
}

// Hook for positions
export function useSyncedPositions() {
  const { positions } = useDataSync();
  
  return useQuery({
    queryKey: ['paper', 'positions'],
    queryFn: async () => {
      const response = await fetch('/api/paper/positions');
      return response.json();
    },
    initialData: positions,
    refetchInterval: DATA_REFRESH_CONFIG.paperPositions.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.paperPositions.staleTime,
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
      return response.json();
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
    queryKey: ['decisions'],
    queryFn: async () => {
      const response = await fetch('/api/decisions');
      return response.json();
    },
    initialData: decisions,
    refetchInterval: DATA_REFRESH_CONFIG.decisions.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.decisions.staleTime,
  });
}

// Hook for strategies
export function useSyncedStrategies() {
  const { strategies } = useDataSync();
  
  return useQuery({
    queryKey: ['strategies'],
    queryFn: async () => {
      const response = await fetch('/api/strategies');
      return response.json();
    },
    initialData: strategies,
    refetchInterval: DATA_REFRESH_CONFIG.strategies.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.strategies.staleTime,
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
      const response = await fetch('/api/health');
      return response.json();
    },
    refetchInterval: DATA_REFRESH_CONFIG.health.refetchInterval,
    staleTime: DATA_REFRESH_CONFIG.health.staleTime,
    // Add connection status to health data
    select: (data) => ({
      ...data,
      wsConnected: isConnected,
    }),
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
