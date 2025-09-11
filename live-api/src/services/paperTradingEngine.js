/**
 * Paper Trading Engine
 * Simulates real crypto trading with Gemini market data
 */

const { GeminiProvider } = require('../providers/crypto/gemini');
const path = require('path');
const fs = require('fs').promises;

class PaperTradingEngine {
  constructor(geminiApiKey, geminiApiSecret) {
    this.gemini = new GeminiProvider(geminiApiKey, geminiApiSecret, true);

    // Paper account data
    this.account = {
      usd_balance: 10000.00, // Starting with $10K
      positions: new Map(), // symbol -> {quantity, avg_price, total_cost}
      orders: new Map(), // order_id -> order_data
      orderHistory: [],
      trades: [],
      lastOrderId: 1000
    };

    this.dataPath = path.join(__dirname, '../../data/paper-account.json');
  }

  /**
   * Initialize or load paper account
   */
  async initialize() {
    try {
      // Try to load existing account data
      const data = await fs.readFile(this.dataPath, 'utf8');
      const savedAccount = JSON.parse(data);

      // Restore account state
      this.account = {
        ...this.account,
        ...savedAccount,
        positions: new Map(savedAccount.positions || []),
        orders: new Map(savedAccount.orders || [])
      };

      console.log('📁 Loaded existing paper account');
      console.log(`💰 USD Balance: $${this.account.usd_balance.toFixed(2)}`);
      console.log(`📊 Positions: ${this.account.positions.size}`);

    } catch (error) {
      console.log('🆕 Creating new paper account');
      await this.saveAccount();
    }
  }

  /**
   * Save account state to disk
   */
  async saveAccount() {
    try {
      const dataToSave = {
        ...this.account,
        positions: Array.from(this.account.positions.entries()),
        orders: Array.from(this.account.orders.entries())
      };

      await fs.writeFile(this.dataPath, JSON.stringify(dataToSave, null, 2));
    } catch (error) {
      console.error('Failed to save paper account:', error);
    }
  }

  /**
   * Fund the paper account (virtual money)
   */
  async fundAccount(amount) {
    if (amount <= 0) {
      throw new Error('Funding amount must be positive');
    }

    this.account.usd_balance += amount;

    // Record funding transaction
    const fundingTransaction = {
      id: `funding_${Date.now()}`,
      type: 'funding',
      amount: amount,
      balance_after: this.account.usd_balance,
      timestamp: new Date().toISOString()
    };

    this.account.trades.push(fundingTransaction);
    await this.saveAccount();

    console.log(`💰 Funded paper account with $${amount.toFixed(2)}`);
    return {
      success: true,
      new_balance: this.account.usd_balance,
      transaction: fundingTransaction
    };
  }

  /**
   * Get account balance and positions
   */
  async getAccount() {
    const positions = [];
    let totalPositionValue = 0;

    // Calculate current position values using real Gemini prices
    for (const [symbol, position] of this.account.positions) {
      try {
        const ticker = await this.gemini.getTicker(symbol);
        const currentPrice = parseFloat(ticker.last);
        const positionValue = position.quantity * currentPrice;
        const unrealizedPnL = positionValue - position.total_cost;

        positions.push({
          symbol,
          quantity: position.quantity,
          avg_price: position.avg_price,
          current_price: currentPrice,
          total_cost: position.total_cost,
          current_value: positionValue,
          unrealized_pnl: unrealizedPnL,
          pnl_percentage: ((currentPrice - position.avg_price) / position.avg_price) * 100
        });

        totalPositionValue += positionValue;
      } catch (error) {
        console.error(`Failed to get price for ${symbol}:`, error);
        // Use last known price if available
        positions.push({
          symbol,
          quantity: position.quantity,
          avg_price: position.avg_price,
          total_cost: position.total_cost,
          error: 'Price unavailable'
        });
      }
    }

    const totalEquity = this.account.usd_balance + totalPositionValue;

    return {
      usd_balance: this.account.usd_balance,
      total_position_value: totalPositionValue,
      total_equity: totalEquity,
      positions: positions,
      orders: Array.from(this.account.orders.values()),
      order_count: this.account.orders.size,
      position_count: this.account.positions.size,
      last_updated: new Date().toISOString()
    };
  }

  /**
   * Place a paper trade order
   */
  async placeOrder(orderData) {
    const { symbol, side, quantity, price, type = 'market' } = orderData;

    if (!['buy', 'sell'].includes(side)) {
      throw new Error('Side must be "buy" or "sell"');
    }

    if (!['market', 'limit'].includes(type)) {
      throw new Error('Type must be "market" or "limit"');
    }

    if (quantity <= 0) {
      throw new Error('Quantity must be positive');
    }

    // Get current market price
    const ticker = await this.gemini.getTicker(symbol);
    const marketPrice = parseFloat(ticker.last);

    let executionPrice = marketPrice;
    let totalCost = 0;

    if (side === 'buy') {
      if (type === 'limit' && price) {
        executionPrice = Math.min(price, marketPrice);
      }

      totalCost = quantity * executionPrice;

      if (totalCost > this.account.usd_balance) {
        throw new Error(`Insufficient funds. Need $${totalCost.toFixed(2)}, have $${this.account.usd_balance.toFixed(2)}`);
      }

      // Update balance
      this.account.usd_balance -= totalCost;

      // Update or create position
      const existingPosition = this.account.positions.get(symbol);
      if (existingPosition) {
        const newQuantity = existingPosition.quantity + quantity;
        const newTotalCost = existingPosition.total_cost + totalCost;
        const newAvgPrice = newTotalCost / newQuantity;

        existingPosition.quantity = newQuantity;
        existingPosition.total_cost = newTotalCost;
        existingPosition.avg_price = newAvgPrice;
      } else {
        this.account.positions.set(symbol, {
          quantity,
          avg_price: executionPrice,
          total_cost: totalCost
        });
      }

    } else { // sell
      const position = this.account.positions.get(symbol);
      if (!position) {
        throw new Error(`No position in ${symbol}`);
      }

      if (position.quantity < quantity) {
        throw new Error(`Insufficient position. Have ${position.quantity}, trying to sell ${quantity}`);
      }

      if (type === 'limit' && price) {
        executionPrice = Math.max(price, marketPrice);
      }

      totalCost = quantity * executionPrice;
      const realizedPnL = totalCost - (quantity * position.avg_price);

      // Update balance
      this.account.usd_balance += totalCost;

      // Update position
      if (position.quantity === quantity) {
        this.account.positions.delete(symbol);
      } else {
        const remainingCost = position.total_cost - (quantity * position.avg_price);
        position.quantity -= quantity;
        position.total_cost = remainingCost;
      }

      // Record realized P&L
      const trade = {
        id: `trade_${Date.now()}`,
        symbol,
        side,
        quantity,
        price: executionPrice,
        total_value: totalCost,
        realized_pnl: realizedPnL,
        timestamp: new Date().toISOString(),
        type: 'sell'
      };

      this.account.trades.push(trade);
    }

    // Create order record
    const orderId = `paper_order_${++this.account.lastOrderId}`;
    const order = {
      order_id: orderId,
      symbol,
      side,
      quantity,
      price: executionPrice,
      total_value: totalCost,
      type,
      status: 'filled',
      timestamp: new Date().toISOString()
    };

    this.account.orders.set(orderId, order);
    this.account.orderHistory.push(order);

    await this.saveAccount();

    console.log(`📈 ${side.toUpperCase()} ${quantity} ${symbol} @ $${executionPrice.toFixed(2)} = $${totalCost.toFixed(2)}`);

    return order;
  }

  /**
   * Get order history
   */
  async getOrderHistory(limit = 50) {
    // Return orders from the orders Map, sorted by timestamp descending
    const orders = Array.from(this.account.orders.values());
    return orders
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, limit);
  }

  /**
   * Get trade history
   */
  async getTradeHistory(limit = 50) {
    return this.account.trades.slice(-limit).reverse();
  }

  /**
   * Reset paper account (for testing)
   */
  async resetAccount() {
    this.account = {
      usd_balance: 10000.00,
      positions: new Map(),
      orders: new Map(),
      orderHistory: [],
      trades: [],
      lastOrderId: 1000
    };

    await this.saveAccount();
    console.log('🔄 Paper account reset to $10,000');
  }

  /**
   * Get account statistics
   */
  async getStats() {
    const account = await this.getAccount();
    const trades = this.account.trades.filter(t => t.type !== 'funding');

    const winningTrades = trades.filter(t => t.realized_pnl > 0);
    const losingTrades = trades.filter(t => t.realized_pnl < 0);

    const totalRealizedPnL = trades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0);
    const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;

    return {
      starting_balance: 10000.00,
      current_balance: account.total_equity,
      total_pnl: account.total_equity - 10000.00,
      pnl_percentage: ((account.total_equity - 10000.00) / 10000.00) * 100,
      realized_pnl: totalRealizedPnL,
      unrealized_pnl: account.total_equity - account.usd_balance - totalRealizedPnL,
      total_trades: trades.length,
      winning_trades: winningTrades.length,
      losing_trades: losingTrades.length,
      win_rate: winRate,
      positions: account.position_count,
      open_orders: account.order_count
    };
  }
}

module.exports = { PaperTradingEngine };
