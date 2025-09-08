import axios from 'axios';
import { Quote, QuoteProvider } from './QuoteProvider';

export class TradierQuoteProvider implements QuoteProvider {
  private token: string;
  private baseUrl: string;
  
  constructor(token: string) {
    this.token = token;
    this.baseUrl = process.env.TRADIER_BASE_URL || 'https://sandbox.tradier.com/v1';
  }
  
  async getQuotes(symbols: string[]): Promise<Record<string, Quote>> {
    if (!symbols.length) return {};
    
    try {
      const response = await axios.get(`${this.baseUrl}/markets/quotes`, {
        params: {
          symbols: symbols.join(','),
          includeQuotes: true
        },
        headers: {
          Authorization: `Bearer ${this.token}`,
          Accept: 'application/json'
        }
      });
      
      const quotes = response.data?.quotes?.quote;
      if (!quotes) return {};
      
      // Handle single quote (not in array)
      const quotesArray = Array.isArray(quotes) ? quotes : [quotes];
      
      const result: Record<string, Quote> = {};
      for (const q of quotesArray) {
        const last = Number(q.last || q.close || q.bid || q.ask || 0);
        const prevClose = Number(q.prevclose || q.previous_close || q.previousClose || 0);
        result[q.symbol] = {
          symbol: q.symbol,
          price: last,
          open: q.open || undefined,
          high: q.high || undefined,
          low: q.low || undefined,
          prevClose,
          change: typeof q.change === 'number' ? q.change : (last && prevClose ? last - prevClose : undefined),
          changePct: typeof q.change_percentage === 'number' ? q.change_percentage : (last && prevClose ? ((last/prevClose)-1)*100 : undefined),
          time: new Date().toISOString()
        };
      }
      
      return result;
    } catch (error) {
      console.error('Tradier quote provider error:', error);
      return {};
    }
  }
}
