import { z } from "zod";

/**
 * Schema for a stock quote from Alpaca
 */
export const QuoteSchema = z.object({
  symbol: z.string(),
  quote: z.object({
    ap: z.number().optional(), // ask price
    bp: z.number().optional(), // bid price
    as: z.number().optional(), // ask size
    bs: z.number().optional(), // bid size
    t: z.string().optional(),  // timestamp
  }).nullable().optional(),
  stale: z.boolean().optional(),
  error: z.string().optional(),
  timestamp: z.string().optional(),
});

export type Quote = z.infer<typeof QuoteSchema>;

/**
 * Schema for a batch of quotes
 */
export const QuoteBatchSchema = z.object({
  quotes: z.record(z.string(), QuoteSchema)
});

export type QuoteBatch = z.infer<typeof QuoteBatchSchema>;

/**
 * Schema for a single bar (candlestick)
 */
export const BarSchema = z.object({
  t: z.string(),      // timestamp
  o: z.number(),      // open
  h: z.number(),      // high
  l: z.number(),      // low
  c: z.number(),      // close
  v: z.number(),      // volume
  n: z.number().optional(), // number of trades
  vw: z.number().optional(), // volume weighted price
});

export type Bar = z.infer<typeof BarSchema>;

/**
 * Schema for a batch of bars
 */
export const BarBatchSchema = z.object({
  bars: z.array(BarSchema),
  symbol: z.string().optional(),
  next_page_token: z.string().optional(),
  stale: z.boolean().optional(),
});

export type BarBatch = z.infer<typeof BarBatchSchema>;

/**
 * Helper to extract price from a quote
 * Returns the mid price if both ask and bid are available,
 * otherwise falls back to ask, bid, or undefined
 */
export function getQuotePrice(quote: Quote | null | undefined): number | undefined {
  if (!quote || !quote.quote) return undefined;
  
  const { ap, bp } = quote.quote;
  
  // Calculate mid price if both ask and bid are available
  if (ap !== undefined && bp !== undefined) {
    return (ap + bp) / 2;
  }
  
  // Fall back to ask or bid
  return ap !== undefined ? ap : bp;
}

/**
 * Helper to get quote staleness status
 */
export function isQuoteStale(quote: Quote | null | undefined): boolean {
  if (!quote) return true;
  return !!quote.stale || !!quote.error;
}
