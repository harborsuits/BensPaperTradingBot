import { Quote, QuoteProvider } from './QuoteProvider';

export class SyntheticQuoteProvider implements QuoteProvider {
  private baseValues: Record<string, number> = {};
  
  async getQuotes(symbols: string[]): Promise<Record<string, Quote>> {
    const result: Record<string, Quote> = {};
    const now = new Date();
    
    for (const symbol of symbols) {
      // Initialize base value if not exists
      if (!this.baseValues[symbol]) {
        this.baseValues[symbol] = 100 + Math.random() * 900;
      }
      
      const base = this.baseValues[symbol];
      // Generate random movement (-2% to +2%)
      const change = base * (Math.random() * 0.04 - 0.02);
      const price = base + change;
      
      // Update base value with some mean reversion
      this.baseValues[symbol] = base * 0.995 + price * 0.005;
      
      result[symbol] = {
        symbol,
        price,
        prevClose: base,
        change,
        changePct: (change / base) * 100,
        time: now.toISOString()
      };
    }
    
    return result;
  }
}
