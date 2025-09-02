import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function usePaperPositions() {
  return useQuery({
    queryKey: ["paperPositions"],
    queryFn: async () => (await axios.get("/api/paper/positions")).data,
    refetchInterval: 15_000,
    staleTime: 10_000,
    select: (d: any) => Array.isArray(d?.items) ? d.items : (Array.isArray(d) ? d : []),
  });
}


