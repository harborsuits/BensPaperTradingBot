const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');

class PerformanceRecorder extends EventEmitter {
  constructor(options = {}) {
    super();
    this.dataDir = options.dataDir || path.join(__dirname, '../data/performance');
    this.maxHistorySize = options.maxHistorySize || 10000;
    
    // Storage
    this.trades = [];
    this.decisions = [];
    this.performance = {};
    this.strategyMetrics = new Map();
    
    // Ensure data directory exists
    this.ensureDataDir();
    
    // Load existing data
    this.loadData();
    
    // Auto-save interval
    this.saveInterval = setInterval(() => this.saveData(), 60000); // Save every minute
  }
  
  ensureDataDir() {
    if (!fs.existsSync(this.dataDir)) {
      fs.mkdirSync(this.dataDir, { recursive: true });
    }
  }
  
  recordDecision(decision) {
    const timestamp = new Date();
    const record = {
      id: `decision_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp,
      symbol: decision.symbol,
      strategy_id: decision.strategy_id,
      side: decision.side,
      confidence: decision.confidence || decision.score,
      price: decision.price,
      size_hint: decision.size_hint,
      context: {
        regime: decision.context?.regime || 'unknown',
        volatility: decision.context?.volatility || 'unknown',
        brain_score: decision.brain_score
      },
      meta: decision.meta || {}
    };
    
    this.decisions.push(record);
    
    // Trim if too large
    if (this.decisions.length > this.maxHistorySize) {
      this.decisions = this.decisions.slice(-this.maxHistorySize);
    }
    
    this.emit('decisionRecorded', record);
    return record;
  }
  
  recordTrade(trade) {
    const timestamp = new Date();
    const record = {
      id: trade.id || `trade_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp,
      symbol: trade.symbol,
      strategy_id: trade.strategy_id,
      side: trade.side,
      quantity: trade.quantity,
      price: trade.price,
      order_id: trade.order_id,
      status: trade.status,
      fill_price: trade.fill_price,
      fill_time: trade.fill_time,
      commission: trade.commission || 0,
      pnl: trade.pnl || null
    };
    
    this.trades.push(record);
    
    // Update strategy performance
    this.updateStrategyPerformance(trade.strategy_id, record);
    
    // Trim if too large
    if (this.trades.length > this.maxHistorySize) {
      this.trades = this.trades.slice(-this.maxHistorySize);
    }
    
    this.emit('tradeRecorded', record);
    return record;
  }
  
  updateStrategyPerformance(strategyId, trade) {
    if (!this.strategyMetrics.has(strategyId)) {
      this.strategyMetrics.set(strategyId, {
        trades: 0,
        wins: 0,
        losses: 0,
        totalPnL: 0,
        totalCommission: 0,
        winRate: 0,
        avgWin: 0,
        avgLoss: 0,
        profitFactor: 0,
        sharpe: 0,
        maxDrawdown: 0,
        equity: []
      });
    }
    
    const metrics = this.strategyMetrics.get(strategyId);
    metrics.trades++;
    
    if (trade.pnl !== null) {
      metrics.totalPnL += trade.pnl;
      metrics.totalCommission += trade.commission || 0;
      
      if (trade.pnl > 0) {
        metrics.wins++;
        metrics.avgWin = ((metrics.avgWin * (metrics.wins - 1)) + trade.pnl) / metrics.wins;
      } else if (trade.pnl < 0) {
        metrics.losses++;
        metrics.avgLoss = ((metrics.avgLoss * (metrics.losses - 1)) + Math.abs(trade.pnl)) / metrics.losses;
      }
      
      metrics.winRate = metrics.wins / metrics.trades;
      metrics.profitFactor = metrics.avgWin * metrics.wins / (metrics.avgLoss * metrics.losses || 1);
      
      // Update equity curve
      const lastEquity = metrics.equity.length > 0 ? 
        metrics.equity[metrics.equity.length - 1].value : 0;
      
      metrics.equity.push({
        timestamp: trade.timestamp,
        value: lastEquity + trade.pnl - (trade.commission || 0)
      });
      
      // Calculate max drawdown
      this.calculateMaxDrawdown(metrics);
    }
  }
  
  calculateMaxDrawdown(metrics) {
    if (metrics.equity.length < 2) return;
    
    let peak = metrics.equity[0].value;
    let maxDrawdown = 0;
    
    for (const point of metrics.equity) {
      if (point.value > peak) {
        peak = point.value;
      }
      const drawdown = (peak - point.value) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    
    metrics.maxDrawdown = maxDrawdown;
  }
  
  /**
   * Calculate Sharpe ratio (annualized)
   * @param {number} days - Number of days to calculate over (default 30)
   * @returns {number} Annualized Sharpe ratio
   */
  calculateSharpe(days = 30) {
    const returns = this.getDailyReturns(days);
    
    if (returns.length < 2) {
      return 0;
    }
    
    // Calculate average daily return
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    
    // Calculate standard deviation
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    // Avoid division by zero
    if (stdDev === 0) {
      return 0;
    }
    
    // Annualize (252 trading days)
    const sharpeRatio = (avgReturn / stdDev) * Math.sqrt(252);
    
    return sharpeRatio;
  }
  
  /**
   * Calculate Sortino ratio (downside deviation only)
   * @param {number} days - Number of days to calculate over
   * @returns {number} Annualized Sortino ratio
   */
  calculateSortino(days = 30) {
    const returns = this.getDailyReturns(days);
    
    if (returns.length < 2) {
      return 0;
    }
    
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    
    // Calculate downside deviation (only negative returns)
    const downsideReturns = returns.filter(r => r < 0);
    if (downsideReturns.length === 0) {
      return avgReturn > 0 ? 10 : 0; // Perfect score if no losses
    }
    
    const downsideVariance = downsideReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / downsideReturns.length;
    const downsideDev = Math.sqrt(downsideVariance);
    
    if (downsideDev === 0) {
      return 0;
    }
    
    // Annualize
    const sortinoRatio = (avgReturn / downsideDev) * Math.sqrt(252);
    
    return sortinoRatio;
  }
  
  /**
   * Get daily returns for a given period
   * @param {number} days - Number of days to look back
   * @returns {Array<number>} Array of daily returns as percentages
   */
  getDailyReturns(days = 30) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    // Group trades by day and calculate daily P&L
    const dailyPnL = new Map();
    
    for (const trade of this.trades) {
      const tradeDate = new Date(trade.timestamp);
      if (tradeDate >= startDate && tradeDate <= endDate) {
        const dateKey = tradeDate.toISOString().split('T')[0];
        const currentPnL = dailyPnL.get(dateKey) || 0;
        dailyPnL.set(dateKey, currentPnL + (trade.pnl || 0));
      }
    }
    
    // Convert to returns (assuming starting equity)
    const equity = 100000; // Should get from account
    const returns = Array.from(dailyPnL.values()).map(pnl => pnl / equity);
    
    return returns;
  }
  
  /**
   * Calculate profit factor (gross profit / gross loss)
   * @param {number} days - Number of days to calculate over
   * @returns {number} Profit factor
   */
  calculateProfitFactor(days = 30) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    let grossProfit = 0;
    let grossLoss = 0;
    
    for (const trade of this.trades) {
      const tradeDate = new Date(trade.timestamp);
      if (tradeDate >= startDate && tradeDate <= endDate && trade.pnl) {
        if (trade.pnl > 0) {
          grossProfit += trade.pnl;
        } else {
          grossLoss += Math.abs(trade.pnl);
        }
      }
    }
    
    if (grossLoss === 0) {
      return grossProfit > 0 ? 999 : 0; // Infinite profit factor
    }
    
    return grossProfit / grossLoss;
  }
  
  getStrategyPerformance(strategyId) {
    return this.strategyMetrics.get(strategyId) || null;
  }
  
  getAllPerformance() {
    const result = {};
    for (const [strategyId, metrics] of this.strategyMetrics) {
      result[strategyId] = metrics;
    }
    return result;
  }
  
  getRecentDecisions(limit = 100) {
    return this.decisions.slice(-limit);
  }
  
  getRecentTrades(limit = 100) {
    return this.trades.slice(-limit);
  }
  
  getDecisionAccuracy(strategyId = null) {
    // Match decisions to trades and calculate accuracy
    const relevantDecisions = strategyId ? 
      this.decisions.filter(d => d.strategy_id === strategyId) : 
      this.decisions;
    
    let correct = 0;
    let total = 0;
    
    for (const decision of relevantDecisions) {
      // Find corresponding trade within 5 minutes
      const trade = this.trades.find(t => 
        t.symbol === decision.symbol &&
        t.strategy_id === decision.strategy_id &&
        Math.abs(new Date(t.timestamp) - new Date(decision.timestamp)) < 300000
      );
      
      if (trade && trade.pnl !== null) {
        total++;
        if ((decision.side === 'buy' && trade.pnl > 0) ||
            (decision.side === 'sell' && trade.pnl > 0)) {
          correct++;
        }
      }
    }
    
    return {
      accuracy: total > 0 ? correct / total : 0,
      total: total,
      correct: correct
    };
  }
  
  loadData() {
    try {
      // Load trades
      const tradesFile = path.join(this.dataDir, 'trades.json');
      if (fs.existsSync(tradesFile)) {
        this.trades = JSON.parse(fs.readFileSync(tradesFile, 'utf8'));
      }
      
      // Load decisions
      const decisionsFile = path.join(this.dataDir, 'decisions.json');
      if (fs.existsSync(decisionsFile)) {
        this.decisions = JSON.parse(fs.readFileSync(decisionsFile, 'utf8'));
      }
      
      // Load performance metrics
      const metricsFile = path.join(this.dataDir, 'metrics.json');
      if (fs.existsSync(metricsFile)) {
        const metricsData = JSON.parse(fs.readFileSync(metricsFile, 'utf8'));
        this.strategyMetrics = new Map(Object.entries(metricsData));
      }
      
      console.log(`[PerformanceRecorder] Loaded ${this.trades.length} trades, ${this.decisions.length} decisions`);
    } catch (error) {
      console.error('[PerformanceRecorder] Error loading data:', error.message);
    }
  }
  
  saveData() {
    try {
      // Save trades
      fs.writeFileSync(
        path.join(this.dataDir, 'trades.json'),
        JSON.stringify(this.trades, null, 2)
      );
      
      // Save decisions
      fs.writeFileSync(
        path.join(this.dataDir, 'decisions.json'),
        JSON.stringify(this.decisions, null, 2)
      );
      
      // Save metrics
      const metricsData = {};
      for (const [strategyId, metrics] of this.strategyMetrics) {
        metricsData[strategyId] = metrics;
      }
      fs.writeFileSync(
        path.join(this.dataDir, 'metrics.json'),
        JSON.stringify(metricsData, null, 2)
      );
      
    } catch (error) {
      console.error('[PerformanceRecorder] Error saving data:', error.message);
    }
  }
  
  destroy() {
    if (this.saveInterval) {
      clearInterval(this.saveInterval);
      this.saveInterval = null;
    }
    this.saveData();
  }
}

module.exports = PerformanceRecorder;
