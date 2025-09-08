import type { QuoteProvider } from './QuoteProvider';
import { TradierQuoteProvider } from './tradier';
import { SyntheticQuoteProvider } from './synthetic';

export function resolveQuoteProvider(): QuoteProvider {
  const provider = process.env.QUOTES_PROVIDER?.toLowerCase() || 'auto';

  const token = process.env.TRADIER_TOKEN || process.env.TRADIER_API_KEY;
  if ((provider === 'tradier' || provider === 'auto') && token) {
    return new TradierQuoteProvider(token);
  }

  if (provider === 'synthetic' || provider === 'auto') {
    return new SyntheticQuoteProvider();
  }

  // 'none' or anything else → throw on use (forces fail-close)
  return {
    async getQuotes() {
      throw new Error('QUOTES_PROVIDER_DISABLED');
    }
  } as unknown as QuoteProvider;
}

export * from './QuoteProvider';
