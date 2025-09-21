import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export interface MarketOverview {
  marketStatus: string;
  asOf: string;
  spx?: { symbol: string; last: number; prevClose: number; change: number; pct: number } | null;
  vix?: { symbol: string; last: number; prevClose: number; change: number; pct: number } | null;
}

export function useMarketOverview() {
  return useQuery<MarketOverview>({
    queryKey: ["market-overview"],
    queryFn: async () => {
      const { data } = await axios.get("/api/overview");
      return data as MarketOverview;
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}
