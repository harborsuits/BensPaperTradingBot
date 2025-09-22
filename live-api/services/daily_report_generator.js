/**
 * Daily Report Generator
 * Generates comprehensive daily trading performance reports
 */

const fs = require('fs').promises;
const path = require('path');

class DailyReportGenerator {
  constructor(performanceRecorder, paperBroker, autoLoop) {
    this.performanceRecorder = performanceRecorder;
    this.paperBroker = paperBroker;
    this.autoLoop = autoLoop;
    this.reportDir = path.join(__dirname, '../reports/daily');
  }
  
  async initialize() {
    // Ensure report directory exists
    await fs.mkdir(this.reportDir, { recursive: true });
  }
  
  /**
   * Generate daily report
   * @returns {Object} Report data
   */
  async generateDailyReport() {
    console.log('[DailyReport] Generating daily performance report...');
    
    const today = new Date().toISOString().split('T')[0];
    const report = {
      date: today,
      generated_at: new Date().toISOString(),
      
      // Trading Activity
      trading_activity: await this.getTradingActivity(),
      
      // Performance Metrics
      performance_metrics: await this.getPerformanceMetrics(),
      
      // Position Summary
      positions: await this.getPositionSummary(),
      
      // Risk Metrics
      risk_metrics: await this.getRiskMetrics(),
      
      // System Health
      system_health: await this.getSystemHealth(),
      
      // Strategy Performance
      strategy_performance: await this.getStrategyPerformance(),
      
      // Notable Events
      notable_events: await this.getNotableEvents()
    };
    
    // Save report
    await this.saveReport(report);
    
    // Generate summary text
    const summary = this.generateSummaryText(report);
    console.log('[DailyReport] Summary:\n' + summary);
    
    return report;
  }
  
  async getTradingActivity() {
    const trades = this.performanceRecorder.trades.filter(t => {
      const tradeDate = new Date(t.timestamp).toISOString().split('T')[0];
      const today = new Date().toISOString().split('T')[0];
      return tradeDate === today;
    });
    
    return {
      total_trades: trades.length,
      winning_trades: trades.filter(t => t.pnl > 0).length,
      losing_trades: trades.filter(t => t.pnl < 0).length,
      total_volume: trades.reduce((sum, t) => sum + (t.quantity * t.price), 0),
      symbols_traded: [...new Set(trades.map(t => t.symbol))],
      trades_by_hour: this.groupTradesByHour(trades)
    };
  }
  
  async getPerformanceMetrics() {
    const account = await this.paperBroker.getAccount();
    const dayPnL = this.calculateTodayPnL();
    
    return {
      // Daily metrics
      daily_pnl: dayPnL,
      daily_pnl_percent: (dayPnL / account.cash) * 100,
      
      // Rolling metrics
      sharpe_30d: this.performanceRecorder.calculateSharpe(30),
      sharpe_90d: this.performanceRecorder.calculateSharpe(90),
      sortino_30d: this.performanceRecorder.calculateSortino(30),
      profit_factor_30d: this.performanceRecorder.calculateProfitFactor(30),
      
      // Cumulative metrics
      total_pnl: account.equity - 100000, // Assuming starting equity
      win_rate: this.calculateWinRate(),
      average_win: this.calculateAverageWin(),
      average_loss: this.calculateAverageLoss(),
      max_drawdown: this.performanceRecorder.strategyMetrics.get('overall')?.maxDrawdown || 0
    };
  }
  
  async getPositionSummary() {
    const positions = await this.paperBroker.getPositions();
    const account = await this.paperBroker.getAccount();
    
    return {
      open_positions: positions.length,
      total_exposure: positions.reduce((sum, p) => sum + Math.abs(p.quantity * p.last), 0),
      exposure_percent: (positions.reduce((sum, p) => sum + Math.abs(p.quantity * p.last), 0) / account.equity) * 100,
      unrealized_pnl: positions.reduce((sum, p) => sum + p.pnl, 0),
      positions: positions.map(p => ({
        symbol: p.symbol,
        quantity: p.quantity,
        entry_price: p.avg_cost,
        current_price: p.last,
        pnl: p.pnl,
        pnl_percent: ((p.last - p.avg_cost) / p.avg_cost) * 100
      }))
    };
  }
  
  async getRiskMetrics() {
    const positions = await this.paperBroker.getPositions();
    const account = await this.paperBroker.getAccount();
    
    return {
      capital_at_risk: positions.reduce((sum, p) => sum + Math.abs(p.quantity * p.avg_cost), 0),
      max_position_size: Math.max(...positions.map(p => Math.abs(p.quantity * p.last)), 0),
      concentration_risk: this.calculateConcentrationRisk(positions),
      daily_var: this.calculateDailyVaR(positions),
      risk_limits: {
        max_position_percent: 10,
        max_daily_loss_percent: 2,
        max_drawdown_percent: 15
      }
    };
  }
  
  async getSystemHealth() {
    return {
      autoloop_status: this.autoLoop.status,
      autoloop_enabled: this.autoLoop.enabled,
      last_autoloop_run: this.autoLoop.lastRun,
      errors_today: 0, // Would track actual errors
      quote_quality: 'good', // Would check actual quote freshness
      broker_connection: 'active', // Would check actual connection
      api_latency_ms: 50 // Would measure actual latency
    };
  }
  
  async getStrategyPerformance() {
    const strategies = [];
    
    for (const [strategyId, metrics] of this.performanceRecorder.strategyMetrics) {
      strategies.push({
        strategy_id: strategyId,
        trades: metrics.trades,
        win_rate: metrics.winRate,
        pnl: metrics.totalPnl,
        avg_pnl: metrics.avgPnl,
        sharpe: this.calculateStrategySpecificSharpe(strategyId)
      });
    }
    
    return strategies.sort((a, b) => b.pnl - a.pnl); // Sort by P&L
  }
  
  async getNotableEvents() {
    const events = [];
    
    // Check for new highs/lows
    const account = await this.paperBroker.getAccount();
    if (account.equity > (this._lastHighWaterMark || 100000)) {
      events.push({
        type: 'new_equity_high',
        message: `New equity high: $${account.equity.toFixed(2)}`,
        timestamp: new Date().toISOString()
      });
      this._lastHighWaterMark = account.equity;
    }
    
    // Check for large trades
    const trades = this.performanceRecorder.trades.filter(t => {
      const tradeDate = new Date(t.timestamp).toISOString().split('T')[0];
      const today = new Date().toISOString().split('T')[0];
      return tradeDate === today;
    });
    
    const largeTrades = trades.filter(t => Math.abs(t.pnl) > 100);
    for (const trade of largeTrades) {
      events.push({
        type: trade.pnl > 0 ? 'large_win' : 'large_loss',
        message: `${trade.symbol}: ${trade.pnl > 0 ? '+' : ''}$${trade.pnl.toFixed(2)}`,
        timestamp: trade.timestamp
      });
    }
    
    // Check for streaks
    const recentTrades = trades.slice(-10);
    const wins = recentTrades.filter(t => t.pnl > 0).length;
    if (wins >= 7) {
      events.push({
        type: 'winning_streak',
        message: `${wins} wins in last 10 trades`,
        timestamp: new Date().toISOString()
      });
    } else if (wins <= 3) {
      events.push({
        type: 'losing_streak',
        message: `Only ${wins} wins in last 10 trades`,
        timestamp: new Date().toISOString()
      });
    }
    
    return events;
  }
  
  async saveReport(report) {
    const filename = `daily-report-${report.date}.json`;
    const filepath = path.join(this.reportDir, filename);
    
    await fs.writeFile(filepath, JSON.stringify(report, null, 2));
    console.log(`[DailyReport] Report saved to ${filepath}`);
    
    // Also save a latest.json for easy access
    const latestPath = path.join(this.reportDir, 'latest.json');
    await fs.writeFile(latestPath, JSON.stringify(report, null, 2));
  }
  
  generateSummaryText(report) {
    const { trading_activity, performance_metrics, positions, risk_metrics } = report;
    
    const summary = `
📊 DAILY TRADING REPORT - ${report.date}
====================================

💰 P&L: ${performance_metrics.daily_pnl >= 0 ? '+' : ''}$${performance_metrics.daily_pnl.toFixed(2)} (${performance_metrics.daily_pnl_percent.toFixed(2)}%)
📈 Trades: ${trading_activity.total_trades} (${trading_activity.winning_trades} wins, ${trading_activity.losing_trades} losses)
📊 Win Rate: ${((trading_activity.winning_trades / trading_activity.total_trades) * 100).toFixed(1)}%

📉 Performance Metrics:
- Sharpe (30d): ${performance_metrics.sharpe_30d.toFixed(2)}
- Sortino (30d): ${performance_metrics.sortino_30d.toFixed(2)}
- Profit Factor: ${performance_metrics.profit_factor_30d.toFixed(2)}
- Max Drawdown: ${(performance_metrics.max_drawdown * 100).toFixed(1)}%

💼 Positions:
- Open: ${positions.open_positions}
- Exposure: $${positions.total_exposure.toFixed(2)} (${positions.exposure_percent.toFixed(1)}%)
- Unrealized P&L: ${positions.unrealized_pnl >= 0 ? '+' : ''}$${positions.unrealized_pnl.toFixed(2)}

⚠️ Risk Metrics:
- Capital at Risk: $${risk_metrics.capital_at_risk.toFixed(2)}
- Daily VaR: $${risk_metrics.daily_var.toFixed(2)}

🎯 Top Performers:
${report.strategy_performance.slice(0, 3).map(s => 
  `- ${s.strategy_id}: $${s.pnl.toFixed(2)} (${s.trades} trades)`
).join('\n')}

📌 Notable Events:
${report.notable_events.map(e => `- ${e.message}`).join('\n') || '- None'}
`;
    
    return summary;
  }
  
  // Helper methods
  calculateTodayPnL() {
    const today = new Date().toISOString().split('T')[0];
    return this.performanceRecorder.trades
      .filter(t => new Date(t.timestamp).toISOString().split('T')[0] === today)
      .reduce((sum, t) => sum + (t.pnl || 0), 0);
  }
  
  calculateWinRate() {
    const recentTrades = this.performanceRecorder.trades.slice(-100);
    if (recentTrades.length === 0) return 0;
    const wins = recentTrades.filter(t => t.pnl > 0).length;
    return wins / recentTrades.length;
  }
  
  calculateAverageWin() {
    const wins = this.performanceRecorder.trades.filter(t => t.pnl > 0);
    if (wins.length === 0) return 0;
    return wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length;
  }
  
  calculateAverageLoss() {
    const losses = this.performanceRecorder.trades.filter(t => t.pnl < 0);
    if (losses.length === 0) return 0;
    return losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length;
  }
  
  calculateConcentrationRisk(positions) {
    if (positions.length === 0) return 0;
    const totalExposure = positions.reduce((sum, p) => sum + Math.abs(p.quantity * p.last), 0);
    const largestPosition = Math.max(...positions.map(p => Math.abs(p.quantity * p.last)));
    return largestPosition / totalExposure;
  }
  
  calculateDailyVaR(positions) {
    // Simplified VaR calculation (95% confidence)
    const totalExposure = positions.reduce((sum, p) => sum + Math.abs(p.quantity * p.last), 0);
    const avgVolatility = 0.02; // 2% daily volatility assumption
    return totalExposure * avgVolatility * 1.645; // 95% confidence
  }
  
  calculateStrategySpecificSharpe(strategyId) {
    // Simplified - would calculate actual Sharpe for specific strategy
    const metrics = this.performanceRecorder.strategyMetrics.get(strategyId);
    if (!metrics || metrics.trades < 10) return 0;
    return metrics.avgPnl > 0 ? 0.5 : -0.5; // Placeholder
  }
  
  groupTradesByHour(trades) {
    const hourly = {};
    for (const trade of trades) {
      const hour = new Date(trade.timestamp).getHours();
      hourly[hour] = (hourly[hour] || 0) + 1;
    }
    return hourly;
  }
}

module.exports = { DailyReportGenerator };
