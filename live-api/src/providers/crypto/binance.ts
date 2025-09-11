/**
 * Binance Crypto Provider
 * Connects to Binance exchange for crypto trading data
 */

import axios from 'axios';
import {
  CryptoProvider,
  CryptoQuote,
  CryptoOrderBook,
  CryptoBalance
} from './CryptoProvider';

export class BinanceCryptoProvider implements CryptoProvider {
  private baseUrl: string;
  private apiKey?: string;
  private secretKey?: string;

  constructor(apiKey?: string, secretKey?: string) {
    this.baseUrl = 'https://api.binance.com/api/v3';
    this.apiKey = apiKey;
    this.secretKey = secretKey;
  }

  async getQuotes(symbols: string[]): Promise<Record<string, CryptoQuote>> {
    if (!symbols.length) return {};

    try {
      const result: Record<string, CryptoQuote> = {};

      // Binance uses different symbol format (BTCUSDT vs BTC/USD)
      const binanceSymbols = symbols.map(s => s.replace('/', '').replace('USD', 'USDT'));

      for (const symbol of binanceSymbols) {
        try {
          const response = await axios.get(`${this.baseUrl}/ticker/24hr`, {
            params: { symbol },
            timeout: 5000
          });

          const data = response.data;
          const baseSymbol = symbol.replace('USDT', '');
          const quoteSymbol = `${baseSymbol}/USD`;

          result[quoteSymbol] = {
            symbol: quoteSymbol,
            price: parseFloat(data.lastPrice),
            bid: parseFloat(data.bidPrice),
            ask: parseFloat(data.askPrice),
            volume24h: parseFloat(data.volume),
            timestamp: new Date(data.closeTime).toISOString(),
            exchange: 'binance',
            spreadBps: ((parseFloat(data.askPrice) - parseFloat(data.bidPrice)) / parseFloat(data.bidPrice)) * 10000
          };
        } catch (symbolError) {
          console.warn(`Failed to get quote for ${symbol}:`, symbolError.message);
        }
      }

      return result;
    } catch (error) {
      console.error('Binance quote provider error:', error);
      return {};
    }
  }

  async getOrderBook(symbol: string, depth: number = 20): Promise<CryptoOrderBook> {
    try {
      const binanceSymbol = symbol.replace('/', '').replace('USD', 'USDT');

      const response = await axios.get(`${this.baseUrl}/depth`, {
        params: { symbol: binanceSymbol, limit: depth },
        timeout: 5000
      });

      const data = response.data;

      return {
        symbol,
        bids: data.bids.map((b: [string, string]) => [parseFloat(b[0]), parseFloat(b[1])]),
        asks: data.asks.map((a: [string, string]) => [parseFloat(a[0]), parseFloat(a[1])]),
        timestamp: new Date().toISOString(),
        exchange: 'binance'
      };
    } catch (error) {
      console.error('Binance order book error:', error);
      throw error;
    }
  }

  async getBalances(): Promise<Record<string, CryptoBalance>> {
    // Placeholder - would require API key and signature
    console.warn('Binance balances not implemented - requires API key authentication');
    return {};
  }

  async isHealthy(): Promise<boolean> {
    try {
      await axios.get(`${this.baseUrl}/ping`, { timeout: 3000 });
      return true;
    } catch (error) {
      return false;
    }
  }

  async getFees(): Promise<Record<string, number>> {
    // Binance standard fees (maker/taker)
    return {
      maker: 0.001,  // 0.1%
      taker: 0.001   // 0.1%
    };
  }
}
