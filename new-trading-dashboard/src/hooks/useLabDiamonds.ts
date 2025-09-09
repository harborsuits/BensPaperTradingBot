import { useQuery } from '@tanstack/react-query';

export interface DiamondSymbol {
  symbol: string;
  score: number;
  features: {
    impact1h: number;
    impact24h: number;
    count24h: number;
    gapPct: number;
    spreadPct: number;
    rvol: number;
  };
}

interface DiamondsResponse {
  items: DiamondSymbol[];
  asOf: string;
}

export const useLabDiamonds = (limit: number = 25, universe: string = 'all') => {
  return useQuery<DiamondsResponse>({
    queryKey: ['lab', 'diamonds', limit, universe],
    queryFn: async () => {
      const response = await fetch(`/api/lab/diamonds?limit=${limit}&universe=${universe}`);
      if (!response.ok) {
        throw new Error('Failed to fetch diamonds data');
      }
      return response.json();
    },
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    staleTime: 2 * 60 * 1000, // Consider data fresh for 2 minutes
  });
};

export const useDiamondDetails = (symbol: string) => {
  return useQuery({
    queryKey: ['lab', 'diamond', symbol],
    queryFn: async () => {
      const response = await fetch(`/api/quotes?symbols=${symbol}`);
      if (!response.ok) {
        throw new Error('Failed to fetch diamond details');
      }
      const data = await response.json();
      return data[0] || null;
    },
    enabled: !!symbol,
    staleTime: 30 * 1000, // 30 seconds
  });
};
