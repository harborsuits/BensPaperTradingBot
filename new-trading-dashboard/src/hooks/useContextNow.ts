import { useQuery } from "@tanstack/react-query";
import type { ContextRow } from "@/contracts/types";

export function useContextNow(){
  return useQuery<ContextRow>({
    queryKey:["context","now"],
    queryFn: async()=> (await fetch("/api/context")).json(),
    refetchInterval: 45000,
    staleTime: 30000,
  });
}


