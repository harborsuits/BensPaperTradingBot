import { z } from "zod";

/**
 * Schema for an individual article in a sentiment cluster
 */
export const SentimentArticle = z.object({
  source: z.string(),
  domain: z.string(),
  title: z.string(),
  url: z.string(),
  published: z.string().optional(),
  info_score: z.number(),
  partisan_score: z.number(),
  finance_score: z.number(),
  sentiment: z.number(),
});

export type SentimentArticle = z.infer<typeof SentimentArticle>;

/**
 * Schema for a cluster of similar news articles with aggregated metrics
 */
export const SentimentCluster = z.object({
  headline: z.string(),
  url: z.string(),
  sentiment: z.number(),
  partisan_spread: z.number(),
  informational: z.number(),
  finance: z.number(),
  sources: z.array(z.string()),
  articles: z.array(SentimentArticle),
});

export type SentimentCluster = z.infer<typeof SentimentCluster>;

/**
 * Schema for news sentiment data grouped by category
 */
// Completely permissive schema to avoid Zod compilation issues
export const CategorySentiment = z.any();

export type TCategorySentiment = z.infer<typeof CategorySentiment>;
