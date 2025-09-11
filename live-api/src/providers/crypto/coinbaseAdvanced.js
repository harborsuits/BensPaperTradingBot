/**
 * Coinbase Advanced Trade API Provider (JavaScript version)
 * Uses the Advanced Trade Sandbox API (no authentication required)
 */

const axios = require('axios');

const ADVANCED_SANDBOX_URL = 'https://api-sandbox.coinbase.com/api/v3/brokerage';

/**
 * Coinbase Advanced Trade Provider
 */
class CoinbaseAdvancedProvider {
  /**
   * Get crypto quotes (simulated for sandbox)
   */
  async getQuotes(symbols) {
    try {
      // For sandbox, we'll simulate quotes since real-time quotes aren't available
      const simulatedQuotes = symbols.map(symbol => ({
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
  async getBalances() {
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
  async getAccount(currency) {
    try {
      const accounts = await this.getBalances();
      return accounts.find(account => account.currency === currency.toUpperCase()) || null;
    } catch (error) {
      console.error('Error getting crypto account:', error);
      throw error;
    }
  }

  /**
   * Create a market order
   */
  async createMarketOrder(productId, side, size) {
    try {
      const orderData = {
        side: side.toUpperCase(),
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
  async getOrderBook(productId) {
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
  async getOrders() {
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
  async isHealthy() {
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
  async getFees() {
    return {
      maker_fee_rate: '0.004',
      taker_fee_rate: '0.006',
      usd_volume: '0'
    };
  }

  /**
   * Fund sandbox account (simulation)
   */
  async fundSandboxAccount(amount, currency = 'USD') {
    console.log(`🎉 Simulated funding: +${amount} ${currency}`);
    console.log('Note: This is a sandbox environment with simulated balances');
  }
}

module.exports = { CoinbaseAdvancedProvider };
