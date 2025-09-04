import { useQuery } from "@tanstack/react-query";
import type { DecisionRow } from "@/contracts/types";

export function useDecisions(){
  return useQuery<DecisionRow[]>({
    queryKey:["decisions","recent"],
    queryFn: async()=> (await fetch("/api/decisions/recent")).json(),
    refetchInterval: 10000,
    staleTime: 7000,
  });
}


