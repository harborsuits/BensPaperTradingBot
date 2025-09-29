import { useQuery } from "@tanstack/react-query";
import axios from "axios";

// Fetch quotes in smaller batches to avoid timeouts
async function fetchQuotesInBatches(symbols: string[]) {
  const BATCH_SIZE = 20; // Fetch 20 symbols at a time
  const batches: string[][] = [];
  
  // Split symbols into batches
  for (let i = 0; i < symbols.length; i += BATCH_SIZE) {
    batches.push(symbols.slice(i, i + BATCH_SIZE));
  }
  
  // Fetch each batch
  const allQuotes: any[] = [];
  for (const batch of batches) {
    try {
      const response = await axios.get('/api/quotes', {
        params: { symbols: batch.join(',') },
        timeout: 5000 // 5 second timeout per batch
      });
      
      // Handle both array and object responses
      const data = response.data;
      if (Array.isArray(data)) {
        allQuotes.push(...data);
      } else if (data && typeof data === 'object') {
        // If it's an object, convert to array
        allQuotes.push(...Object.values(data));
      }
    } catch (error) {
      console.error(`[Quotes] Error fetching batch:`, error);
      // Continue with other batches even if one fails
    }
  }
  
  return allQuotes;
}

export function useQuotesBatched(symbols: string[]) {
  return useQuery({
    queryKey: ['quotes', symbols],
    queryFn: () => fetchQuotesInBatches(symbols),
    refetchInterval: 5000,
    staleTime: 3000,
    retry: 1,
    retryDelay: 1000
  });
}
