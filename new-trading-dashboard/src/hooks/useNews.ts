import { useMemo } from 'react';
import { useNewsSentiment } from './useNewsSentiment';

type SentimentBucket = 'positive' | 'neutral' | 'negative';

export interface SimpleNewsItem {
  id: string;
  title: string;
  summary?: string;
  source?: string;
  timestamp?: string;
  sentiment?: SentimentBucket;
  impact?: number;
  url?: string;
}

/**
 * Thin adapter that maps CategorySentiment to simple items for Sidebar.tsx
 */
export function useNews(bucket: SentimentBucket = 'neutral', perBucket: number = 3) {
  // Use markets category as default; adjust if Sidebar needs other categories later
  const { data, isLoading, error } = useNewsSentiment('markets', '', 5);

  const items: SimpleNewsItem[] | undefined = useMemo(() => {
    if (!data?.clusters) return [];
    // Map clusters to simple items and sort/group by sentiment bucket
    const mapped = data.clusters.map((c, idx) => {
      // Normalize sentiment: positive (>0.2), negative (<-0.2), else neutral
      const s: SentimentBucket = c.sentiment > 0.2 ? 'positive' : c.sentiment < -0.2 ? 'negative' : 'neutral';
      return {
        id: `${idx}-${c.url}`,
        title: c.headline,
        summary: c.articles?.[0]?.title,
        source: c.sources?.[0],
        timestamp: undefined,
        sentiment: s,
        impact: Math.max(0, Math.min(100, Math.round(Math.abs(c.informational) * 100))) || undefined,
        url: c.url,
      } as SimpleNewsItem;
    });

    const filtered = mapped.filter(m => m.sentiment === bucket);
    return filtered.slice(0, perBucket);
  }, [data, bucket, perBucket]);

  return { data: items, isLoading, error } as const;
}

export default useNews;


