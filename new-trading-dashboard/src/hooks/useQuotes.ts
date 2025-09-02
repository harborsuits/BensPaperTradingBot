import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useQuotes(symbols: string[]) {
  const validSymbols = symbols.filter(Boolean);
  const joined = validSymbols.slice().sort().join(",");
  
  return useQuery({
    queryKey: ["quotes", joined],
    queryFn: async () => {
      if (!joined) return [];
      const { data } = await axios.get(`/api/quotes?symbols=${joined}`);
      // Accept either {quotes:[...]} or [...]
      const quotes = Array.isArray(data) ? data : Array.isArray(data?.quotes) ? data.quotes : [];
      return quotes.map((r:any) => ({
        symbol: r.symbol,
        last: Number(r.last ?? r.price ?? r.close ?? 0),
        bid: Number(r.bid ?? 0),
        ask: Number(r.ask ?? 0),
        prevClose: Number(r.prevClose ?? r.previousClose ?? 0),
        change: Number(r.change ?? 0),
        pct: Number(r.pct ?? r.changePercent ?? 0),
        ts: r.ts ?? r.timestamp ?? null,
      }));
    },
    enabled: validSymbols.length > 0,
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}

// Alias for compatibility
export const usePrices = useQuotes;

// Single quote hook
export function useQuote(symbol: string) {
  return useQuery({
    queryKey: ["quotes", symbol],
    queryFn: async () => {
      const response = await axios.get(`/api/quotes?symbols=${symbol}`);
      return response.data?.[0] || null;
    },
    refetchInterval: 15_000, 
    staleTime: 10_000,
  });
}