import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useMetrics() {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: async () => {
      try {
        // Try the API metrics endpoint first
        const response = await axios.get("/api/metrics");
        return response.data;
      } catch (error) {
        // Fallback to root metrics if API endpoint fails
        try {
          const fallbackResponse = await axios.get("/metrics");
          return fallbackResponse.data;
        } catch {
          // Return default metrics if both fail
          return {
            totalSymbolsTracked: 0,
            errorRate: 0,
            requestsLastHour: 0,
            averageLatency: 0,
            timestamp: new Date().toISOString()
          };
        }
      }
    },
    refetchInterval: 30_000,
    staleTime: 20_000,
  });
}


