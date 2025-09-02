import { useQuery } from '@tanstack/react-query';
import { api } from '../utils/api';

export interface PortfolioData {
  value: number;
  history: {
    date: string;
    value: number;
  }[];
  startDate: string;
  endDate: string;
  initialValue: number;
  currentValue: number;
  percentChange: number;
}

export function usePortfolio() {
  return useQuery<PortfolioData>({
    queryKey: ['portfolio'],
    queryFn: async () => {
      const response = await api.get('/portfolio');
      return response.data;
    },
    refetchInterval: 60000,    // Refresh data every minute
    staleTime: 300000,        // Keep data for 5 minutes if window loses focus
  });
} 