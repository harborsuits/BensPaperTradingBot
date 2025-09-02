import { useQuery } from '@tanstack/react-query';
import { api } from '../utils/api';

export interface Allocation {
  name: string;
  value: number;
  percent: number;
  color?: string;
}

export interface AllocationData {
  allocations: Allocation[];
  totalValue: number;
}

export function useAllocation() {
  return useQuery<AllocationData>({
    queryKey: ['allocation'],
    queryFn: async () => {
      const response = await api.get('/allocation');
      // Transform data to match expected format
      const categories = response.data.categories || [];
      
      return {
        allocations: categories.map(cat => ({
          name: cat.name,
          percent: cat.value,
          value: 0, // We'd calculate this in a real app from totalValue
        })),
        totalValue: 100000 // Placeholder
      };
    },
    refetchInterval: 300000,  // Refresh data every 5 minutes
    staleTime: 600000,        // Keep data for 10 minutes if window loses focus
  });
} 