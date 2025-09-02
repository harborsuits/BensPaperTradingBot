import { useQuery } from "@tanstack/react-query";
import { fetchNewsSentiment } from "@/services/news";
import { TCategorySentiment } from "@/schemas/news";
import { usePageVisible } from "./usePageVisible";

/**
 * Hook for fetching news sentiment data by category
 * 
 * @param category News category (markets, politics, tech, crypto, macro)
 * @param query Optional filter for headlines containing this text
 * @param perSource Maximum number of articles to process per source
 * @returns React Query result with news sentiment data
 */
export function useNewsSentiment(category = "markets", query = "", perSource = 5) {
  const visible = usePageVisible();

  return useQuery<TCategorySentiment>({
    queryKey: ["news-sentiment", category, query, perSource],
    queryFn: () => fetchNewsSentiment(category, query, perSource),
    staleTime: 60_000, // 1 minute
    gcTime: 300_000, // 5 minutes
    refetchInterval: visible ? 300_000 : false, // 5 minutes when visible
    refetchOnWindowFocus: false,
    retry: (failureCount, error: any) => {
      // Don't retry on 404s
      if (error?.response?.status === 404) return false;
      // Retry up to 2 times on other errors
      return failureCount < 2;
    }
  });
}
