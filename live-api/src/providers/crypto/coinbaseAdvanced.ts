/**
 * Coinbase Advanced Trade API Provider
 * Uses the Advanced Trade Sandbox API (no authentication required)
 */

import axios from 'axios';

const ADVANCED_SANDBOX_URL = 'https://api-sandbox.coinbase.com/api/v3/brokerage';

export interface CryptoQuote {
  product_id: string;
  price: string;
  size: string;
  time: string;
}

export interface CryptoAccount {
  uuid: string;
  name: string;
  currency: string;
  available_balance: {
    value: string;
    currency: string;
  };
  hold: {
    value: string;
    currency: string;
  };
}

export interface CryptoOrder {
  order_id: string;
  product_id: string;
  side: 'BUY' | 'SELL';
  order_configuration: any;
  status: string;
  created_time: string;
}

export class CoinbaseAdvancedProvider {
  /**
   * Get crypto quotes (simulated for sandbox)
   */
  async getQuotes(symbols: string[]): Promise<CryptoQuote[]> {
    try {
      // For sandbox, we'll simulate quotes since real-time quotes aren't available
      const simulatedQuotes: CryptoQuote[] = symbols.map(symbol => ({
        product_id: symbol,
        price: (Math.random() * 100 + 50).toFixed(2), // Random price between 50-150
        size: (Math.random() * 10 + 1).toFixed(8),
        time: new Date().toISOString()
      }));

      return simulatedQuotes;
    } catch (error) {
      console.error('Error getting crypto quotes:', error);
      throw error;
    }
  }

  /**
   * Get account balances
   */
  async getBalances(): Promise<CryptoAccount[]> {
    try {
      const response = await axios.get(`${ADVANCED_SANDBOX_URL}/accounts`);
      return response.data.accounts;
    } catch (error) {
      console.error('Error getting crypto balances:', error);
      throw error;
    }
  }

  /**
   * Get account by currency
   */
  async getAccount(currency: string): Promise<CryptoAccount | null> {
    try {
      const accounts = await this.getBalances();
      return accounts.find(account => account.currency === currency) || null;
    } catch (error) {
      console.error('Error getting crypto account:', error);
      throw error;
    }
  }

  /**
   * Create a market order
   */
  async createMarketOrder(
    productId: string,
    side: 'BUY' | 'SELL',
    size: string
  ): Promise<CryptoOrder> {
    try {
      const orderData = {
        side,
        order_configuration: {
          market_market_ioc: {
            base_size: size
          }
        },
        product_id: productId
      };

      const response = await axios.post(
        `${ADVANCED_SANDBOX_URL}/orders`,
        orderData,
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      return response.data.order;
    } catch (error) {
      console.error('Error creating crypto order:', error);
      throw error;
    }
  }

  /**
   * Get order book for a product
   */
  async getOrderBook(productId: string): Promise<any> {
    try {
      // For sandbox, simulate order book
      return {
        product_id: productId,
        bids: [
          [(Math.random() * 100 + 50).toFixed(2), (Math.random() * 10).toFixed(8)],
          [(Math.random() * 100 + 49).toFixed(2), (Math.random() * 10).toFixed(8)]
        ],
        asks: [
          [(Math.random() * 100 + 51).toFixed(2), (Math.random() * 10).toFixed(8)],
          [(Math.random() * 100 + 52).toFixed(2), (Math.random() * 10).toFixed(8)]
        ]
      };
    } catch (error) {
      console.error('Error getting crypto order book:', error);
      throw error;
    }
  }

  /**
   * Get historical orders
   */
  async getOrders(): Promise<CryptoOrder[]> {
    try {
      const response = await axios.get(`${ADVANCED_SANDBOX_URL}/orders/historical/batch`);
      return response.data.orders || [];
    } catch (error) {
      console.error('Error getting crypto orders:', error);
      throw error;
    }
  }

  /**
   * Check if the API is healthy
   */
  async isHealthy(): Promise<boolean> {
    try {
      await axios.get(`${ADVANCED_SANDBOX_URL}/accounts`);
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get trading fees (simulated for sandbox)
   */
  async getFees(): Promise<any> {
    return {
      maker_fee_rate: '0.004',
      taker_fee_rate: '0.006',
      usd_volume: '0'
    };
  }

  /**
   * Fund sandbox account (simulation)
   */
  async fundSandboxAccount(amount: number, currency: string = 'USD'): Promise<void> {
    console.log(`🎉 Simulated funding: +${amount} ${currency}`);
    console.log('Note: This is a sandbox environment with simulated balances');
  }
}
