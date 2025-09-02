import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import axios from "axios";

export function useUniverse() {
  const qc = useQueryClient();
  const list = useQuery({
    queryKey: ["universe"],
    queryFn: async () => {
      const response = await axios.get("/api/universe");
      // Return symbols array
      return response.data.symbols || response.data || [];
    },
    staleTime: 30_000,
  });
  const setUniverse = useMutation({
    mutationFn: async (id: string) => (await axios.post("/api/universe", { id })).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["universe"] });
      qc.invalidateQueries({ queryKey: ["quotes"] });
      qc.invalidateQueries({ queryKey: ["bars"] });
      qc.invalidateQueries({ queryKey: ["decisions", "recent"] });
      qc.invalidateQueries({ queryKey: ["trades"] });
    },
  });
  const watchlists = useQuery({
    queryKey: ["watchlists"],
    queryFn: async () => {
      try {
        const response = await axios.get("/api/watchlists");
        const data = response.data;
        // Accept shapes:
        // 1) array: [{id,name,symbols}]
        // 2) object map: { default:[...], tech:[...] }
        // 3) {items:[...]}
        if (Array.isArray(data)) return data;
        if (Array.isArray(data?.items)) return data.items;
        if (data && typeof data === 'object') {
          return Object.entries(data).map(([id, symbols]) => ({ id, name: id, symbols }));
        }
        return [];
      } catch (error) {
        console.warn("Watchlists not available, using fallback");
        return [{ id: 'default', name: 'Default', symbols: ['SPY', 'QQQ'] }];
      }
    },
    staleTime: 60_000,
  });
  return { list, watchlists, setUniverse };
}


