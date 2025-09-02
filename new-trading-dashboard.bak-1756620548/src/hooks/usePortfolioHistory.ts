import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function usePortfolioHistory(mode: "paper"|"live" = "paper", days = 90) {
  const path = mode === "paper" ? "/api/portfolio/paper/history" : "/api/portfolio/live/history";
  return useQuery({
    queryKey: ["portfolioHistory", mode, days],
    queryFn: async () => (await axios.get(`${path}?days=${days}`)).data,
    refetchInterval: mode === "paper" ? 30_000 : 60_000,
    staleTime: mode === "paper" ? 20_000 : 45_000,
  });
}


