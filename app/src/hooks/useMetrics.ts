import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useMetrics() {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: async () => (await axios.get("/metrics")).data,
    refetchInterval: 30_000,
    staleTime: 20_000,
  });
}


