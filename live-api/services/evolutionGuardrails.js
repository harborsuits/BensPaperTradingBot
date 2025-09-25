const learningConfig = require('../config/learningConfig');

class EvolutionGuardrails {
  constructor(performanceRecorder, botCompetitionService) {
    this.performanceRecorder = performanceRecorder;
    this.botCompetition = botCompetitionService;
    this.config = learningConfig.evolution;
    
    // Track strategy lineage
    this.strategyLineage = new Map(); // strategyId -> { parent1, parent2, generation, birth_date }
    this.strategyPerformance = new Map(); // strategyId -> performance_metrics
    
    // Elite strategies (top performers immune to some mutations)
    this.eliteStrategies = new Set();
    
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Listen for competition results
    if (this.botCompetition) {
      this.botCompetition.on('competition_complete', (results) => {
        this.evaluateCompetitionResults(results);
      });
      
      this.botCompetition.on('breeding_request', (parents) => {
        this.validateBreedingPair(parents);
      });
    }
  }

  // Evaluate competition results and enforce guardrails
  evaluateCompetitionResults(results) {
    const survivors = [];
    const eliminated = [];
    
    results.bots.forEach(bot => {
      const metrics = this.calculatePerformanceMetrics(bot);
      this.strategyPerformance.set(bot.id, metrics);
      
      // Check if meets minimum requirements
      if (this.meetsPerformanceGates(metrics)) {
        survivors.push(bot);
        
        // Check if elite
        if (this.isElitePerformer(metrics)) {
          this.eliteStrategies.add(bot.id);
          console.log(`[Evolution] Strategy ${bot.id} promoted to ELITE status`);
        }
      } else {
        eliminated.push({
          bot,
          reason: this.getEliminationReason(metrics)
        });
      }
    });
    
    console.log(`[Evolution] Competition results: ${survivors.length} survived, ${eliminated.length} eliminated`);
    
    // Report eliminations
    eliminated.forEach(({ bot, reason }) => {
      console.log(`[Evolution] Eliminated ${bot.id}: ${reason}`);
    });
    
    return {
      survivors,
      eliminated,
      elite_count: this.eliteStrategies.size
    };
  }

  // Calculate comprehensive performance metrics
  calculatePerformanceMetrics(bot) {
    const trades = bot.trades || [];
    if (trades.length === 0) {
      return {
        trade_count: 0,
        sharpe_ratio: 0,
        max_drawdown: 0,
        win_rate: 0,
        avg_win_loss_ratio: 0,
        profit_factor: 0
      };
    }
    
    // Calculate returns
    const returns = [];
    let equity = 100000; // Starting equity
    const equityCurve = [equity];
    
    trades.forEach(trade => {
      const returnPct = trade.pnl / equity;
      returns.push(returnPct);
      equity += trade.pnl;
      equityCurve.push(equity);
    });
    
    // Sharpe Ratio (simplified - daily)
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    );
    const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0; // Annualized
    
    // Maximum Drawdown
    let maxDrawdown = 0;
    let peak = equityCurve[0];
    
    equityCurve.forEach(equity => {
      if (equity > peak) peak = equity;
      const drawdown = (peak - equity) / peak;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    });
    
    // Win Rate
    const wins = trades.filter(t => t.pnl > 0).length;
    const winRate = wins / trades.length;
    
    // Average Win/Loss Ratio
    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl < 0);
    const avgWin = winningTrades.length > 0 ? 
      winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length : 0;
    const avgLoss = losingTrades.length > 0 ? 
      Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length) : 1;
    const winLossRatio = avgLoss > 0 ? avgWin / avgLoss : avgWin;
    
    // Profit Factor
    const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999 : 0;
    
    return {
      trade_count: trades.length,
      sharpe_ratio: sharpeRatio,
      max_drawdown: maxDrawdown,
      win_rate: winRate,
      avg_win_loss_ratio: winLossRatio,
      profit_factor: profitFactor,
      total_pnl: trades.reduce((sum, t) => sum + t.pnl, 0)
    };
  }

  // Check if strategy meets performance gates
  meetsPerformanceGates(metrics) {
    const gates = this.config.performance;
    
    // Must have minimum trades
    if (metrics.trade_count < gates.minTrades) {
      return false;
    }
    
    // Check each gate
    const checks = {
      sharpe: metrics.sharpe_ratio >= gates.minSharpe,
      drawdown: metrics.max_drawdown <= gates.maxDrawdown,
      winRate: metrics.win_rate >= gates.minWinRate
    };
    
    // Need at least 2 out of 3 gates passed
    const passedGates = Object.values(checks).filter(v => v).length;
    return passedGates >= 2;
  }

  // Determine if strategy is elite performer
  isElitePerformer(metrics) {
    return (
      metrics.sharpe_ratio > 1.0 &&
      metrics.max_drawdown < 0.1 &&
      metrics.win_rate > 0.5 &&
      metrics.profit_factor > 1.5
    );
  }

  // Get reason for elimination
  getEliminationReason(metrics) {
    const gates = this.config.performance;
    
    if (metrics.trade_count < gates.minTrades) {
      return `Insufficient trades (${metrics.trade_count}/${gates.minTrades})`;
    }
    
    const failures = [];
    
    if (metrics.sharpe_ratio < gates.minSharpe) {
      failures.push(`Low Sharpe (${metrics.sharpe_ratio.toFixed(2)})`);
    }
    
    if (metrics.max_drawdown > gates.maxDrawdown) {
      failures.push(`High drawdown (${(metrics.max_drawdown * 100).toFixed(1)}%)`);
    }
    
    if (metrics.win_rate < gates.minWinRate) {
      failures.push(`Low win rate (${(metrics.win_rate * 100).toFixed(1)}%)`);
    }
    
    return failures.join(', ');
  }

  // Validate breeding pair before crossover
  validateBreedingPair(parents) {
    const parent1Metrics = this.strategyPerformance.get(parents[0].id);
    const parent2Metrics = this.strategyPerformance.get(parents[1].id);
    
    // Both parents must have performance data
    if (!parent1Metrics || !parent2Metrics) {
      throw new Error('Cannot breed strategies without performance data');
    }
    
    // At least one parent should meet performance gates
    if (!this.meetsPerformanceGates(parent1Metrics) && 
        !this.meetsPerformanceGates(parent2Metrics)) {
      throw new Error('At least one parent must meet performance gates');
    }
    
    console.log(`[Evolution] Breeding approved:`, {
      parent1: { id: parents[0].id, sharpe: parent1Metrics.sharpe_ratio.toFixed(2) },
      parent2: { id: parents[1].id, sharpe: parent2Metrics.sharpe_ratio.toFixed(2) }
    });
    
    return true;
  }

  // Apply mutation with guardrails
  applyMutation(childParams, parentParams) {
    const mutationConfig = this.config.mutation;
    const mutatedParams = { ...childParams };
    
    Object.keys(mutatedParams).forEach(key => {
      if (typeof mutatedParams[key] === 'number' && Math.random() < mutationConfig.mutationRate) {
        const parentValue = parentParams[key] || mutatedParams[key];
        const maxChange = parentValue * mutationConfig.maxDeviation;
        
        // Gaussian mutation within bounds
        const change = (Math.random() - 0.5) * 2 * maxChange;
        mutatedParams[key] = parentValue + change;
        
        // Ensure sensible bounds
        if (key.includes('threshold')) {
          mutatedParams[key] = Math.max(0.3, Math.min(0.9, mutatedParams[key]));
        } else if (key.includes('stop')) {
          mutatedParams[key] = Math.max(0.01, Math.min(0.2, mutatedParams[key]));
        } else if (key.includes('target')) {
          mutatedParams[key] = Math.max(0.02, Math.min(0.5, mutatedParams[key]));
        }
      }
    });
    
    return mutatedParams;
  }

  // Create offspring with controlled parameters
  createOffspring(parent1, parent2) {
    const childParams = {};
    const mutationConfig = this.config.mutation;
    
    // Crossover
    Object.keys(parent1.params).forEach(key => {
      if (Math.random() < mutationConfig.crossoverRate) {
        childParams[key] = parent1.params[key];
      } else {
        childParams[key] = parent2.params[key];
      }
    });
    
    // Apply mutation (unless parents are elite)
    const parent1Elite = this.eliteStrategies.has(parent1.id);
    const parent2Elite = this.eliteStrategies.has(parent2.id);
    
    if (!parent1Elite || !parent2Elite) {
      // Only mutate if at least one parent is not elite
      return this.applyMutation(childParams, parent1.params);
    }
    
    return childParams;
  }

  // Track strategy lineage
  recordLineage(childId, parent1Id, parent2Id) {
    const parent1Lineage = this.strategyLineage.get(parent1Id) || { generation: 0 };
    const parent2Lineage = this.strategyLineage.get(parent2Id) || { generation: 0 };
    
    this.strategyLineage.set(childId, {
      parent1: parent1Id,
      parent2: parent2Id,
      generation: Math.max(parent1Lineage.generation, parent2Lineage.generation) + 1,
      birth_date: new Date(),
      elite_parent: this.eliteStrategies.has(parent1Id) || this.eliteStrategies.has(parent2Id)
    });
  }

  // Generate evolution report
  generateEvolutionReport() {
    const report = {
      timestamp: new Date(),
      total_strategies: this.strategyPerformance.size,
      elite_strategies: this.eliteStrategies.size,
      performance_distribution: this.getPerformanceDistribution(),
      top_lineages: this.getTopLineages(),
      parameter_trends: this.getParameterTrends()
    };
    
    return report;
  }

  getPerformanceDistribution() {
    const distribution = {
      high_performers: 0, // Sharpe > 1
      moderate_performers: 0, // Sharpe 0.5-1
      low_performers: 0, // Sharpe < 0.5
      average_sharpe: 0,
      average_drawdown: 0
    };
    
    let totalSharpe = 0;
    let totalDrawdown = 0;
    
    this.strategyPerformance.forEach(metrics => {
      totalSharpe += metrics.sharpe_ratio;
      totalDrawdown += metrics.max_drawdown;
      
      if (metrics.sharpe_ratio > 1) {
        distribution.high_performers++;
      } else if (metrics.sharpe_ratio > 0.5) {
        distribution.moderate_performers++;
      } else {
        distribution.low_performers++;
      }
    });
    
    const count = this.strategyPerformance.size;
    distribution.average_sharpe = count > 0 ? totalSharpe / count : 0;
    distribution.average_drawdown = count > 0 ? totalDrawdown / count : 0;
    
    return distribution;
  }

  getTopLineages() {
    // Group strategies by generation
    const generationPerformance = new Map();
    
    this.strategyLineage.forEach((lineage, strategyId) => {
      const metrics = this.strategyPerformance.get(strategyId);
      if (metrics) {
        const gen = lineage.generation;
        if (!generationPerformance.has(gen)) {
          generationPerformance.set(gen, []);
        }
        generationPerformance.set(gen, [...generationPerformance.get(gen), metrics.sharpe_ratio]);
      }
    });
    
    // Calculate average performance by generation
    const genAverages = [];
    generationPerformance.forEach((sharpes, gen) => {
      const avg = sharpes.reduce((a, b) => a + b, 0) / sharpes.length;
      genAverages.push({ generation: gen, avg_sharpe: avg, count: sharpes.length });
    });
    
    return genAverages.sort((a, b) => b.avg_sharpe - a.avg_sharpe);
  }

  getParameterTrends() {
    // This would analyze how parameters evolve over generations
    // Simplified for now
    return {
      trend: 'Parameters converging toward optimal values',
      convergence_rate: 'Moderate'
    };
  }
}

module.exports = EvolutionGuardrails;
