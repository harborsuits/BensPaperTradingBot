import { useQuery } from "@tanstack/react-query";
import type { IngestEvent } from "@/contracts/types";

export function useIngestionEvents(limit = 30){
  return useQuery<IngestEvent[]>({
    queryKey:["ingestion","events", limit],
    queryFn: async()=> (await fetch(`/api/ingestion/events?limit=${limit}`)).json(),
    refetchInterval: 8000,
    staleTime: 5000,
  });
}


