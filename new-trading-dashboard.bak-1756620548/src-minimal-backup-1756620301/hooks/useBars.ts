import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useBars(symbol: string, timeframe = "1Day", limit = 30) {
  return useQuery({
    queryKey: ["bars", symbol, timeframe, limit],
    queryFn: async () => {
      const { data } = await axios.get(`/api/bars`, { 
        params: { symbol, timeframe, limit }
      });
      // Accept either {bars:[...]} or [...]
      const bars = Array.isArray(data?.bars) ? data.bars : Array.isArray(data) ? data : [];
      return { 
        bars: bars.map((b:any) => ({
          t: b.t ?? b.time ?? b.timestamp,
          o: Number(b.o ?? b.open ?? 0),
          h: Number(b.h ?? b.high ?? 0),
          l: Number(b.l ?? b.low ?? 0),
          c: Number(b.c ?? b.close ?? 0),
          v: Number(b.v ?? b.volume ?? 0),
        }))
      };
    },
    enabled: !!symbol,
    refetchInterval: 60_000, 
    staleTime: 30_000,
  });
}