import { useQuery } from '@tanstack/react-query';
import { api } from '@/utils/api';

export interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  value: number;
  pnl: number;
  pnlPercent: number;
}

export interface Metrics {
  totalPnl: {
    value: number;
    percent: number;
  };
  todayPnl: {
    value: number;
    percent: number;
  };
  openPositions: Position[];
  winRate: {
    percent: number;
    wins: number;
    losses: number;
  };
  exposure: {
    percent: number;
    invested: number;
    cash: number;
  };
  balance: {
    total: number;
    updated: string; // ISO date string
  };
}

export function useMetrics() {
  return useQuery<Metrics>({
    queryKey: ['metrics'],
    queryFn: async () => {
      const response = await api.get('/metrics');
      return response.data;
    },
    refetchInterval: 30000,
    staleTime: 20000,
  });
}