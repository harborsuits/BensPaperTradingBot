import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useTrades(limit = 20) {
  return useQuery({
    queryKey: ["trades", limit],
    queryFn: async () => (await axios.get(`/api/trades?limit=${limit}`)).data,
    refetchInterval: 7_000,
    staleTime: 5_000,
    select: (d: any) => Array.isArray(d?.items) ? d.items : (Array.isArray(d) ? d : []),
  });
}


