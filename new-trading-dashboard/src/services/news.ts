import { api } from "@/services/apiClient";
import { CategorySentiment } from "@/schemas/news";

/**
 * Fetches news sentiment data from the API for a specific category
 * 
 * @param category News category (markets, politics, tech, crypto, macro)
 * @param query Optional filter for headlines containing this text
 * @param perSource Maximum number of articles to process per source
 * @returns Parsed news sentiment data with clusters and outlet metrics
 */
export async function fetchNewsSentiment(category = "markets", query = "", perSource = 5) {
  const r = await api.get("/news/sentiment", { 
    params: { 
      category, 
      query, 
      per_source: perSource 
    }
  });
  return CategorySentiment.parse(r.data);
}
