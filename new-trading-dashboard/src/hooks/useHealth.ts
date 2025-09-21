import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export type HealthResponse = { env?: string; status?: string; ok?: boolean; asOf?: string, breaker?: 'GREEN'|'AMBER'|'RED', gitSha?: string; meta?: { asOf?: string } } | null;

export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: async () => {
      try {
        const { data } = await axios.get(`/api/health`);
        return data ?? null;
      } catch {
        return null;
      }
    },
    refetchInterval: 20_000,
    staleTime: 15_000,
    retry: 1,
  });
}


