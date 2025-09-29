import type { QuoteProvider } from './QuoteProvider';
import { TradierQuoteProvider } from './tradier';
import { SyntheticQuoteProvider } from './synthetic';

export function resolveQuoteProvider(): QuoteProvider {
  const provider = process.env.QUOTES_PROVIDER?.toLowerCase() || 'auto';

  // Check for Tradier token in multiple env vars
  const token = process.env.TRADIER_TOKEN || 
                process.env.TRADIER_API_KEY || 
                'KU2iUnOZIUFre0wypgyOn8TgmGxI'; // Fallback to sandbox token
                
  if (token) {
    return new TradierQuoteProvider(token);
  }
  
  // This should never happen now with fallback token
  throw new Error('No quote provider available');
}

export * from './QuoteProvider';
