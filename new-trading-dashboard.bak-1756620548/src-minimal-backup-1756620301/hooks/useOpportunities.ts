import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Candidate } from '@/types/candidate';

export function useOpportunities(filters?: Record<string, any>) {
  return useQuery<Candidate[]>({
    queryKey: ['opportunities', filters],
    queryFn: async () => {
      try {
        // Use our dedicated opportunities server
        const { data } = await axios.get('http://localhost:4001/api/opportunities', { params: filters });
        return data;
      } catch (error) {
        // If it fails, try the main API
        console.log('Falling back to main API opportunities');
        try {
          const { data } = await axios.get('/api/opportunities', { params: filters });
          return data;
        } catch (secondError) {
          console.error('Both opportunities endpoints failed', secondError);
          // Return an empty array as a last resort
          return [];
        }
      }
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

export function useProbeOpportunity() {
  return async (id: string) => {
    try {
      // Try the dedicated server first
      const { data } = await axios.post(`http://localhost:4001/api/opportunities/${id}/probe`);
      return data;
    } catch (error) {
      // Fall back to the main API
      console.log('Falling back to main API for probe');
      const { data } = await axios.post(`/api/opportunities/${id}/probe`);
      return data;
    }
  };
}

export function usePaperOrderOpportunity() {
  return async (id: string) => {
    try {
      // Try the dedicated server first
      const { data } = await axios.post(`http://localhost:4001/api/opportunities/${id}/paper-order`);
      return data;
    } catch (error) {
      // Fall back to the main API
      console.log('Falling back to main API for paper order');
      const { data } = await axios.post(`/api/opportunities/${id}/paper-order`);
      return data;
    }
  };
}
