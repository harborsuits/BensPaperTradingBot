import { useQuery } from '@tanstack/react-query';
import { api } from '../utils/api';

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  total: number;
  timestamp: string;
  status: 'completed' | 'pending' | 'failed';
  pnl?: number;
  pnlPercent?: number;
}

export function useRecentTrades(limit = 10) {
  return useQuery<Trade[]>({
    queryKey: ['trades', 'recent', limit],
    queryFn: async () => {
      const response = await api.get(`/trades/recent?limit=${limit}`);
      return response.data;
    },
    refetchInterval: 30000, // Refresh more frequently than portfolio
    staleTime: 60000,
  });
} 