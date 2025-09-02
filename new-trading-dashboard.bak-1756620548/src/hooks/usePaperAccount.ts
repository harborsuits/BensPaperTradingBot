import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function usePaperAccount() {
  return useQuery({
    queryKey: ["paperAccount"],
    queryFn: async () => (await axios.get("/api/paper/account")).data,
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}
