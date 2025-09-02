import { useQuery } from "@tanstack/react-query";
import axios from "axios";

type Freshness = "live" | "warn" | "stale";

export function useActiveStrategies() {
  return useQuery({
    queryKey: ['strategies', 'active'],
    queryFn: async () => {
      try {
        const { data } = await axios.get("/api/strategies/active");
        return data;
      } catch {
        const { data } = await axios.get("/api/strategies");
        return data?.items ?? data ?? [];
      }
    },
    refetchInterval: 60_000,
    staleTime: 45_000,
    select: (arr: any[]) => (Array.isArray(arr) ? arr : []).map((s: any) => {
      const asOf: string | undefined = s.asOf ?? s.updatedAt ?? s.ts;
      const ageMs = asOf ? Math.max(0, Date.now() - new Date(asOf).getTime()) : Infinity;
      let freshness: Freshness = "live";
      if (ageMs >= 180_000) freshness = "stale";
      else if (ageMs >= 90_000) freshness = "warn";
      return { ...s, asOf, ageMs, freshness };
    }),
  });
}

