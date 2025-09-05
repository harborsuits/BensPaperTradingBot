import type { QuoteProvider } from './QuoteProvider';
import { TradierQuoteProvider } from './tradier';
import { SyntheticQuoteProvider } from './synthetic';

export function resolveQuoteProvider(): QuoteProvider {
  const provider = process.env.QUOTES_PROVIDER?.toLowerCase() || 'auto';
  
  if (provider === 'tradier' && process.env.TRADIER_TOKEN) {
    return new TradierQuoteProvider(process.env.TRADIER_TOKEN);
  }
  
  if (provider === 'synthetic' || provider === 'auto') {
    return new SyntheticQuoteProvider();
  }
  
  // Default fallback
  return new SyntheticQuoteProvider();
}

export * from './QuoteProvider';
