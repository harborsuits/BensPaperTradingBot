import { useQuery } from "@tanstack/react-query";
import type { PortfolioAllocationsResponse } from "@/contracts/types";

export function usePortfolioAllocations(){
  return useQuery<PortfolioAllocationsResponse>({
    queryKey:["portfolio","allocations"],
    queryFn: async()=> (await fetch("/api/portfolio/allocations")).json(),
    refetchInterval: 15000,
    staleTime: 10000,
  });
}


