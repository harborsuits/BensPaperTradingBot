import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useDecisionsRecent(limit = 20) {
  return useQuery({
    queryKey: ["decisionsRecent", limit],
    queryFn: async () => (await axios.get(`/api/decisions/recent?limit=${limit}`)).data,
    refetchInterval: 7_000,
    staleTime: 5_000,
    select: (data: any) => {
      // Normalize the response - handle both array and {items: [...]} formats
      const items = Array.isArray(data) ? data : data?.items || [];
      // Filter out invalid decisions and provide defaults
      return items.filter((d: any) => d?.symbol).map((d: any) => ({
        ...d,
        strategy: d.strategy || d.strategyName || "Unknown",
        decidedAt: d.decidedAt || d.timestamp || new Date().toISOString(),
        reasons: Array.isArray(d.reasons) ? d.reasons : d.reason ? [d.reason] : ["No reason provided"],
        plan: d.plan || { sizePct: 0, slPct: 0, tpPct: 0 }
      }));
    },
  });
}


