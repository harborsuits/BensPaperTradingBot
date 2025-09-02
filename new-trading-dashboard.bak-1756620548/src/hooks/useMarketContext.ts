import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useRegime() {
  return useQuery({
    queryKey: ["context","regime"],
    queryFn: async () => (await axios.get("/api/context/regime")).data,
    refetchInterval: 45_000,
    staleTime: 30_000,
  });
}

export function useVolatility() {
  return useQuery({
    queryKey: ["context","volatility"],
    queryFn: async () => (await axios.get("/api/context/volatility")).data,
    refetchInterval: 45_000,
    staleTime: 30_000,
  });
}

export function useSentiment() {
  return useQuery({
    queryKey: ["context","sentiment"],
    queryFn: async () => (await axios.get("/api/context/sentiment")).data,
    refetchInterval: 45_000,
    staleTime: 30_000,
    select: (d: any) => {
      if (!d) return d;
      const confidence = typeof d.confidence === "number" ? d.confidence : 0;
      return { ...d, confidencePct: Math.round(confidence * 100) };
    },
  });
}

export function useSentimentHistory(days = 30) {
  return useQuery({
    queryKey: ["context","sentiment","history", days],
    queryFn: async () => (await axios.get(`/api/context/sentiment/history?days=${days}`)).data,
    refetchInterval: 60_000,
    staleTime: 45_000,
  });
}

export function useSentimentAnomalies(limit = 10) {
  return useQuery({
    queryKey: ["context","sentiment","anomalies", limit],
    queryFn: async () => (await axios.get(`/api/context/sentiment/anomalies?limit=${limit}`)).data,
    refetchInterval: 60_000,
    staleTime: 45_000,
  });
}

export function useContextNews(limit = 10) {
  return useQuery({
    queryKey: ["context","news", limit],
    queryFn: async () => (await axios.get(`/api/context/news?limit=${limit}`)).data,
    refetchInterval: 60_000,
    staleTime: 45_000,
  });
}

export function useNewsSentiment(category = "markets", perSource = 5) {
  return useQuery({
    queryKey: ["news","sentiment", category, perSource],
    queryFn: async () => (await axios.get(`/api/news/sentiment?category=${category}&per_source=${perSource}`)).data,
    refetchInterval: 60_000,
    staleTime: 45_000,
  });
}

export function useMarketContext() {
  const regime = useRegime();
  const volatility = useVolatility();
  const sentiment = useSentiment();
  const news = useContextNews();
  const newsSentiment = useNewsSentiment();
  
  // Transform news sentiment from outlets object to array for UI compatibility
  const newsSentimentArray = newsSentiment.data?.outlets ? 
    Object.entries(newsSentiment.data.outlets).map(([source, data]: [string, any]) => ({
      source,
      score: data.score,
      count: data.count
    })) : [];
  
  return {
    regime: regime.data,
    volatility: volatility.data,
    sentiment: sentiment.data,
    news: news.data,
    newsSentiment: newsSentimentArray,
    isLoading: regime.isLoading || volatility.isLoading || sentiment.isLoading,
    error: regime.error || volatility.error || sentiment.error,
  };
}


