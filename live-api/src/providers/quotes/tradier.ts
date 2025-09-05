import axios from 'axios';
import { Quote, QuoteProvider } from './QuoteProvider';

export class TradierQuoteProvider implements QuoteProvider {
  private token: string;
  private baseUrl = 'https://api.tradier.com/v1';
  
  constructor(token: string) {
    this.token = token;
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
        result[q.symbol] = {
          symbol: q.symbol,
          price: q.last || q.close || q.bid || q.ask || 0,
          open: q.open || undefined,
          high: q.high || undefined,
          low: q.low || undefined,
          prevClose: q.prevclose || undefined,
          change: q.change || undefined,
          changePct: q.change_percentage || undefined,
          time: new Date().toISOString() // Tradier doesn't always provide a timestamp
        };
      }
      
      return result;
    } catch (error) {
      console.error('Tradier quote provider error:', error);
      return {};
    }
  }
}
