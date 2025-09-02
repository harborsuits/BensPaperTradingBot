import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export type Candidate = {
  symbol: string; last: number;
  score: number; confidence: number; side: "buy"|"sell";
  plan: { entry: number; stop: number; take: number; type: string };
  risk: { suggestedQty: number; spreadOK: boolean; liquidityOK: boolean };
  explain: { impact1h:number; impact24h:number; count24h:number; rvol:number; gapPct:number; spreadPct:number; atr:number; outlets:string[] };
  asOf: string;
};

export function useScannerCandidates(list = 'small_caps_liquid', limit = 30) {
  return useQuery<Candidate[]>({
    queryKey: ['scanner', list, limit],
    queryFn: async () => {
      const { data } = await axios.get(`/api/scanner/candidates`, { params: { list, limit }});
      return Array.isArray(data) ? data : [];
    },
    refetchInterval: 30_000, staleTime: 20_000, retry: 1
  });
}
