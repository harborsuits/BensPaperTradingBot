/**
 * Gemini Crypto Exchange Provider
 * Supports both live and sandbox environments
 */

const crypto = require('crypto');
const axios = require('axios');

class GeminiProvider {
  constructor(apiKey, apiSecret, isSandbox = true) {
    this.apiKey = apiKey;
    this.apiSecret = apiSecret;
    this.isSandbox = isSandbox;

    // Gemini API endpoints
    this.baseUrl = isSandbox
      ? 'https://api.sandbox.gemini.com'
      : 'https://api.gemini.com';

    this.apiVersion = 'v1';
  }

  /**
   * Generate HMAC SHA384 signature for Gemini API authentication
   */
  generateSignature(payload) {
    const secret = Buffer.from(this.apiSecret, 'base64');
    const message = JSON.stringify(payload);

    return crypto
      .createHmac('sha384', secret)
      .update(message)
      .digest('hex');
  }

  /**
   * Make authenticated request to Gemini API
   */
  async makeRequest(endpoint, method = 'GET', data = null) {
    const url = `${this.baseUrl}/${this.apiVersion}${endpoint}`;

    const headers = {
      'Content-Type': 'text/plain',
      'Content-Length': '0',
      'X-GEMINI-APIKEY': this.apiKey,
      'X-GEMINI-PAYLOAD': '',
      'X-GEMINI-SIGNATURE': '',
      'Cache-Control': 'no-cache'
    };

    // For POST requests, add payload and signature
    if (method === 'POST' && data) {
      const payload = {
        request: endpoint,
        nonce: Date.now(),
        ...data
      };

      headers['X-GEMINI-PAYLOAD'] = Buffer.from(JSON.stringify(payload)).toString('base64');
      headers['X-GEMINI-SIGNATURE'] = this.generateSignature(payload);
      headers['Content-Type'] = 'text/plain';
      headers['Content-Length'] = '0';
    } else if (method === 'GET') {
      // For GET requests, still need basic auth headers
      const payload = {
        request: endpoint,
        nonce: Date.now()
      };

      headers['X-GEMINI-PAYLOAD'] = Buffer.from(JSON.stringify(payload)).toString('base64');
      headers['X-GEMINI-SIGNATURE'] = this.generateSignature(payload);
    }

    const config = {
      method,
      url,
      headers,
      timeout: 10000
    };

    if (method === 'POST' && data) {
      // For POST requests, the payload is in the headers, not body
      config.data = '';
    }

    try {
      const response = await axios(config);
      return response.data;
    } catch (error) {
      console.error('Gemini API error:', error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Get account balances
   */
  async getBalances() {
    try {
      // Gemini uses different endpoints. Let's try /v1/account
      const data = await this.makeRequest('/account');

      // Convert Gemini format to our standardized format
      const balances = data.map(balance => ({
        currency: balance.currency,
        amount: balance.amount,
        available: balance.available,
        availableForWithdrawal: balance.availableForWithdrawal,
        type: balance.type,
        since: new Date(balance.timestampms || Date.now()).toISOString()
      }));

      return balances;
    } catch (error) {
      console.error('Error getting Gemini balances:', error);
      throw error;
    }
  }

  /**
   * Get account balance for specific currency
   */
  async getBalance(currency) {
    try {
      const balances = await this.getBalances();
      return balances.find(balance =>
        balance.currency.toLowerCase() === currency.toLowerCase()
      );
    } catch (error) {
      console.error('Error getting Gemini balance:', error);
      throw error;
    }
  }

  /**
   * Get market data (ticker)
   */
  async getTicker(symbol) {
    try {
      const response = await axios.get(`${this.baseUrl}/${this.apiVersion}/pubticker/${symbol}`, {
        timeout: 5000
      });

      return {
        symbol,
        last: parseFloat(response.data.last),
        bid: parseFloat(response.data.bid),
        ask: parseFloat(response.data.ask),
        volume: parseFloat(response.data.volume),
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error getting Gemini ticker:', error);
      throw error;
    }
  }

  /**
   * Get multiple tickers
   */
  async getTickers(symbols) {
    try {
      const tickers = await Promise.all(
        symbols.map(symbol => this.getTicker(symbol))
      );
      return tickers;
    } catch (error) {
      console.error('Error getting Gemini tickers:', error);
      throw error;
    }
  }

  /**
   * Place a new order
   */
  async placeOrder(orderData) {
    try {
      const payload = {
        symbol: orderData.symbol,
        amount: orderData.amount.toString(),
        price: orderData.price?.toString(),
        side: orderData.side.toLowerCase(),
        type: orderData.type || 'exchange limit'
      };

      // Add options for different order types
      if (orderData.type === 'market') {
        payload.type = 'exchange stop';
        payload.options = ['immediate-or-cancel'];
      }

      const response = await this.makeRequest('/order/new', 'POST', payload);

      return {
        order_id: response.order_id,
        id: response.id,
        symbol: response.symbol,
        side: response.side,
        type: response.type,
        price: response.price,
        amount: response.original_amount,
        executed_amount: response.executed_amount,
        remaining_amount: response.remaining_amount,
        status: response.is_live ? 'open' : 'filled',
        timestamp: response.timestampms
      };
    } catch (error) {
      console.error('Error placing Gemini order:', error);
      throw error;
    }
  }

  /**
   * Get order status
   */
  async getOrder(orderId) {
    try {
      const response = await this.makeRequest(`/order/status`, 'POST', {
        order_id: parseInt(orderId)
      });

      return {
        order_id: response.order_id,
        symbol: response.symbol,
        side: response.side,
        price: response.price,
        amount: response.original_amount,
        executed_amount: response.executed_amount,
        remaining_amount: response.remaining_amount,
        status: response.is_cancelled ? 'cancelled' : (response.is_live ? 'open' : 'filled'),
        timestamp: response.timestampms
      };
    } catch (error) {
      console.error('Error getting Gemini order:', error);
      throw error;
    }
  }

  /**
   * Cancel order
   */
  async cancelOrder(orderId) {
    try {
      const response = await this.makeRequest('/order/cancel', 'POST', {
        order_id: parseInt(orderId)
      });

      return {
        order_id: response.order_id,
        status: 'cancelled',
        timestamp: response.timestampms
      };
    } catch (error) {
      console.error('Error cancelling Gemini order:', error);
      throw error;
    }
  }

  /**
   * Get order history
   */
  async getOrderHistory() {
    try {
      const response = await this.makeRequest('/orders/history');

      return response.map(order => ({
        order_id: order.order_id,
        symbol: order.symbol,
        side: order.side,
        price: order.price,
        amount: order.original_amount,
        executed_amount: order.executed_amount,
        remaining_amount: order.remaining_amount,
        status: order.is_cancelled ? 'cancelled' : (order.is_live ? 'open' : 'filled'),
        timestamp: order.timestampms
      }));
    } catch (error) {
      console.error('Error getting Gemini order history:', error);
      throw error;
    }
  }

  /**
   * Get available trading symbols
   */
  async getSymbols() {
    try {
      const response = await axios.get(`${this.baseUrl}/${this.apiVersion}/symbols`, {
        timeout: 5000
      });

      return response.data;
    } catch (error) {
      console.error('Error getting Gemini symbols:', error);
      throw error;
    }
  }

  /**
   * Check API connectivity
   */
  async isHealthy() {
    try {
      await this.getBalances();
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get trading fees
   */
  async getFees() {
    try {
      // Gemini fees are typically 0.35% for maker, 0.25% for taker in sandbox
      return {
        maker_fee: '0.0035',
        taker_fee: '0.0025',
        usd_volume: '0'
      };
    } catch (error) {
      console.error('Error getting Gemini fees:', error);
      throw error;
    }
  }
}

module.exports = { GeminiProvider };
