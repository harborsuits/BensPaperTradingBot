import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useSafetyStatus() {
  return useQuery({
    queryKey: ["safetyStatus"],
    queryFn: async () => (await axios.get("/api/safety/status")).data,
    refetchInterval: 30_000,
    staleTime: 20_000,
  });
}


