import { useQuery } from '@tanstack/react-query';
import { api } from '../utils/api';

export interface NewsItem {
  id: number;
  title: string;
  source: string;
  date: string;
  sentiment: 'positive' | 'negative' | 'neutral';
}

export interface NewsData {
  items: NewsItem[];
}

export function useNews() {
  return useQuery({
    queryKey: ['news'],
    queryFn: async () => {
      const response = await api.get('/news');
      return response.data;
    },
    refetchInterval: 300000,  // Refresh data every 5 minutes
    staleTime: 600000,        // Keep data for 10 minutes if window loses focus
  });
} 