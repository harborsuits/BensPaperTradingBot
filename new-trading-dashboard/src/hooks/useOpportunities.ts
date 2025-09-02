import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Candidate } from '@/types/candidate';

export function useOpportunities(filters?: Record<string, any>) {
  return useQuery<Candidate[]>({
    queryKey: ['opportunities', filters],
    queryFn: async () => {
      const { data } = await axios.get('/api/opportunities', { params: filters });
      return data;
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

export function useProbeOpportunity() {
  return async (id: string) => {
    const { data } = await axios.post(`/api/opportunities/${id}/probe`);
    return data;
  };
}

export function usePaperOrderOpportunity() {
  return async (id: string) => {
    const { data } = await axios.post(`/api/opportunities/${id}/paper-order`);
    return data;
  };
}
