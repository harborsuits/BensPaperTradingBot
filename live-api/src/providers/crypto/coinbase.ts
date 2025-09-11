/**
 * Coinbase Crypto Provider
 * Connects to Coinbase Sandbox for crypto trading and account management
 */

import axios from 'axios';
import {
  CryptoProvider,
  CryptoQuote,
  CryptoOrderBook,
  CryptoBalance
} from './CryptoProvider';

export interface CoinbaseConfig {
  apiKey: string;
  apiSecret: string;
  isSandbox: boolean;
}

export class CoinbaseCryptoProvider implements CryptoProvider {
  private baseUrl: string;
  private apiKey: string;
  private apiSecret: string;
  private isSandbox: boolean;

  constructor(config: CoinbaseConfig) {
    this.apiKey = config.apiKey;
    this.apiSecret = config.apiSecret;
    this.isSandbox = config.isSandbox;

    // Use sandbox or production URL
    this.baseUrl = this.isSandbox
      ? 'https://api-public.sandbox.exchange.coinbase.com'
      : 'https://api.exchange.coinbase.com';
  }

  /**
   * Create authentication headers for Coinbase API
   */
  private createAuthHeaders(method: string, path: string, body: string = ''): Record<string, string> {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const message = timestamp + method + path + body;
    const signature = require('crypto')
      .createHmac('sha256', this.apiSecret)
      .update(message)
      .digest('hex');

    return {
      'CB-ACCESS-KEY': this.apiKey,
      'CB-ACCESS-SIGN': signature,
      'CB-ACCESS-TIMESTAMP': timestamp,
      'CB-ACCESS-PASSPHRASE': '', // Not required for sandbox
      'Content-Type': 'application/json'
    };
  }

  async getQuotes(symbols: string[]): Promise<Record<string, CryptoQuote>> {
    if (!symbols.length) return {};

    try {
      const result: Record<string, CryptoQuote> = {};

      for (const symbol of symbols) {
        try {
          const path = `/products/${symbol}/ticker`;
          const headers = this.createAuthHeaders('GET', path);

          const response = await axios.get(`${this.baseUrl}${path}`, {
            headers,
            timeout: 5000
          });

          const data = response.data;

          result[symbol] = {
            symbol,
            price: parseFloat(data.price),
            bid: parseFloat(data.bid || data.price),
            ask: parseFloat(data.ask || data.price),
            volume24h: parseFloat(data.volume_24h || '0'),
            timestamp: new Date().toISOString(),
            exchange: 'coinbase',
            spreadBps: data.bid && data.ask ?
              ((parseFloat(data.ask) - parseFloat(data.bid)) / parseFloat(data.bid)) * 10000 : 0
          };
        } catch (symbolError) {
          console.warn(`Failed to get Coinbase quote for ${symbol}:`, symbolError.message);
        }
      }

      return result;
    } catch (error) {
      console.error('Coinbase quote provider error:', error);
      return {};
    }
  }

  async getOrderBook(symbol: string, depth: number = 20): Promise<CryptoOrderBook> {
    try {
      const path = `/products/${symbol}/book?level=2`;
      const headers = this.createAuthHeaders('GET', path);

      const response = await axios.get(`${this.baseUrl}${path}`, {
        headers,
        timeout: 5000
      });

      const data = response.data;

      return {
        symbol,
        bids: (data.bids || []).slice(0, depth).map((b: [string, string]) => ({
          price: parseFloat(b[0]),
          size: parseFloat(b[1])
        })),
        asks: (data.asks || []).slice(0, depth).map((a: [string, string]) => ({
          price: parseFloat(a[0]),
          size: parseFloat(a[1])
        })),
        timestamp: new Date().toISOString(),
        exchange: 'coinbase'
      };
    } catch (error) {
      console.error('Coinbase order book error:', error);
      throw error;
    }
  }

  async getBalances(): Promise<Record<string, CryptoBalance>> {
    try {
      const path = '/accounts';
      const headers = this.createAuthHeaders('GET', path);

      const response = await axios.get(`${this.baseUrl}${path}`, {
        headers,
        timeout: 5000
      });

      const result: Record<string, CryptoBalance> = {};

      for (const account of response.data) {
        if (account.currency) {
          result[account.currency] = {
            asset: account.currency,
            free: parseFloat(account.available || '0'),
            locked: parseFloat(account.hold || '0'),
            total: parseFloat(account.balance || '0')
          };
        }
      }

      return result;
    } catch (error) {
      console.error('Coinbase balances error:', error);
      return {};
    }
  }

  async isHealthy(): Promise<boolean> {
    try {
      const path = '/time';
      const headers = this.createAuthHeaders('GET', path);

      await axios.get(`${this.baseUrl}${path}`, {
        headers,
        timeout: 3000
      });
      return true;
    } catch (error) {
      return false;
    }
  }

  async getFees(): Promise<Record<string, number>> {
    // Coinbase fees (maker/taker)
    return {
      maker: 0.004,  // 0.4%
      taker: 0.006   // 0.6%
    };
  }

  /**
   * Fund sandbox account with specified amount
   */
  async fundSandboxAccount(currency: string = 'USD', amount: number = 10000): Promise<any> {
    if (!this.isSandbox) {
      throw new Error('Funding is only available in sandbox mode');
    }

    try {
      const path = '/accounts/sandbox-funding';
      const body = JSON.stringify({
        currency,
        amount: amount.toString()
      });

      const headers = this.createAuthHeaders('POST', path, body);

      const response = await axios.post(`${this.baseUrl}${path}`, body, {
        headers,
        timeout: 10000
      });

      return response.data;
    } catch (error) {
      console.error('Coinbase sandbox funding error:', error);
      throw error;
    }
  }

  /**
   * Get account information
   */
  async getAccounts(): Promise<any[]> {
    try {
      const path = '/accounts';
      const headers = this.createAuthHeaders('GET', path);

      const response = await axios.get(`${this.baseUrl}${path}`, {
        headers,
        timeout: 5000
      });

      return response.data;
    } catch (error) {
      console.error('Coinbase accounts error:', error);
      return [];
    }
  }

  /**
   * Create a market order
   */
  async createMarketOrder(productId: string, side: 'buy' | 'sell', size: string): Promise<any> {
    try {
      const path = '/orders';
      const body = JSON.stringify({
        type: 'market',
        side,
        product_id: productId,
        size
      });

      const headers = this.createAuthHeaders('POST', path, body);

      const response = await axios.post(`${this.baseUrl}${path}`, body, {
        headers,
        timeout: 10000
      });

      return response.data;
    } catch (error) {
      console.error('Coinbase order error:', error);
      throw error;
    }
  }
}
