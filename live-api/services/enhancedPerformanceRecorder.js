const EventEmitter = require('events');
const learningConfig = require('../config/learningConfig');

class EnhancedPerformanceRecorder extends EventEmitter {
  constructor(existingRecorder) {
    super();
    this.recorder = existingRecorder;
    this.learningConfig = learningConfig;
    
    // Learning data structures
    this.observations = {
      strategies: new Map(),      // strategy -> { samples, successes, regime_performance }
      symbols: new Map(),         // symbol -> { attempts, failures, last_success, regimes }
      newsPatterns: new Map(),    // pattern -> { count, outcomes, confidence }
      exitRules: new Map(),       // rule -> { count, pnl_distribution }
      correlations: new Map()     // strategyPair -> correlation_data
    };
    
    // Current market regime
    this.currentRegime = {
      trend: 'neutral',       // bull, bear, neutral
      volatility: 'normal',   // low, normal, high
      liquidity: 'normal',    // low, normal, high
      lastUpdate: new Date()
    };
    
    // Recalibration state
    this.recalibrationMode = false;
    this.recalibrationEndTime = null;
    
    // Start regime detection
    this.startRegimeDetection();
  }

  // Record a trade with regime context
  recordTradeWithLearning(trade) {
    const regime = this.getCurrentRegime();
    const strategy = trade.strategy;
    const symbol = trade.symbol;
    
    // Update strategy observations
    if (!this.observations.strategies.has(strategy)) {
      this.observations.strategies.set(strategy, {
        samples: 0,
        successes: 0,
        regime_performance: {},
        symbols_failed: new Set(),
        symbols_succeeded: new Set()
      });
    }
    
    const stratData = this.observations.strategies.get(strategy);
    stratData.samples++;
    
    if (trade.pnl > 0) {
      stratData.successes++;
      stratData.symbols_succeeded.add(symbol);
    } else {
      stratData.symbols_failed.add(symbol);
    }
    
    // Track regime-specific performance
    const regimeKey = `${regime.trend}_${regime.volatility}`;
    if (!stratData.regime_performance[regimeKey]) {
      stratData.regime_performance[regimeKey] = { samples: 0, win_rate: 0 };
    }
    stratData.regime_performance[regimeKey].samples++;
    
    // Update symbol observations
    this.updateSymbolObservations(symbol, trade);
    
    // Check for recalibration triggers
    this.checkRecalibrationTriggers();
    
    // Learn cross-strategy correlations
    this.updateStrategyCorrelations(trade);
    
    return this.recorder.recordTrade(trade);
  }

  // Get learning-adjusted confidence
  getAdjustedConfidence(symbol, strategy, baseConfidence) {
    // Check if we have enough samples
    const stratData = this.observations.strategies.get(strategy);
    if (!stratData || stratData.samples < this.learningConfig.minimumSamples.strategy) {
      return baseConfidence; // Not enough data
    }
    
    // Get regime-specific adjustments
    const regime = this.getCurrentRegime();
    const regimeAdjustment = this.getRegimeAdjustment(regime);
    
    // Check symbol-specific history
    const symbolData = this.observations.symbols.get(symbol);
    let symbolAdjustment = 1.0;
    
    if (symbolData) {
      if (symbolData.cooling_until && new Date() < symbolData.cooling_until) {
        // Symbol is in cooldown
        return 0;
      }
      
      // Decay-adjusted failure rate
      const failureRate = this.getDecayedFailureRate(symbolData);
      symbolAdjustment = 1 - (failureRate * 0.5); // Max 50% reduction
    }
    
    // Strategy performance adjustment
    const strategyWinRate = stratData.successes / stratData.samples;
    const strategyAdjustment = 0.5 + (strategyWinRate * 0.5); // 0.5 to 1.0 range
    
    // Check if we're in recalibration mode
    const recalibrationFactor = this.recalibrationMode ? 
      this.learningConfig.recalibration.actions.resetConfidence : 1.0;
    
    // Combine all adjustments
    const adjustedConfidence = baseConfidence * 
      regimeAdjustment * 
      symbolAdjustment * 
      strategyAdjustment * 
      recalibrationFactor;
    
    // Log learning application
    if (Math.abs(adjustedConfidence - baseConfidence) > 0.1) {
      console.log(`[Learning] Adjusted confidence for ${symbol}/${strategy}:`, {
        base: baseConfidence.toFixed(3),
        adjusted: adjustedConfidence.toFixed(3),
        factors: {
          regime: regimeAdjustment.toFixed(2),
          symbol: symbolAdjustment.toFixed(2),
          strategy: strategyAdjustment.toFixed(2),
          recalibration: recalibrationFactor.toFixed(2)
        }
      });
    }
    
    return Math.max(0, Math.min(1, adjustedConfidence));
  }

  // Update symbol observations with decay
  updateSymbolObservations(symbol, trade) {
    if (!this.observations.symbols.has(symbol)) {
      this.observations.symbols.set(symbol, {
        attempts: 0,
        failures: 0,
        last_success: null,
        last_failure: null,
        cooling_until: null,
        regime_history: []
      });
    }
    
    const symbolData = this.observations.symbols.get(symbol);
    symbolData.attempts++;
    
    if (trade.pnl <= 0) {
      symbolData.failures++;
      symbolData.last_failure = new Date();
      
      // Check if needs cooldown
      const failureRate = symbolData.failures / symbolData.attempts;
      if (symbolData.attempts >= this.learningConfig.minimumSamples.symbol && 
          failureRate > 0.7) {
        // Apply cooldown with decay
        const cooldownDays = Math.min(
          this.learningConfig.decay.symbolCooldown.maxCooldown,
          Math.pow(2, symbolData.failures - 10) // Exponential backoff after 10 failures
        );
        symbolData.cooling_until = new Date(Date.now() + cooldownDays * 24 * 60 * 60 * 1000);
        console.log(`[Learning] Symbol ${symbol} entering cooldown until ${symbolData.cooling_until.toLocaleDateString()}`);
      }
    } else {
      symbolData.last_success = new Date();
      // Success can reduce cooldown
      if (symbolData.cooling_until) {
        const reduction = this.learningConfig.decay.symbolCooldown.halfLife * 24 * 60 * 60 * 1000;
        symbolData.cooling_until = new Date(Math.max(
          Date.now(),
          symbolData.cooling_until.getTime() - reduction
        ));
      }
    }
    
    // Track regime context
    symbolData.regime_history.push({
      regime: this.getCurrentRegime(),
      outcome: trade.pnl > 0 ? 'win' : 'loss',
      timestamp: new Date()
    });
  }

  // Get decay-adjusted failure rate
  getDecayedFailureRate(symbolData) {
    const now = new Date();
    let weightedFailures = 0;
    let weightedTotal = 0;
    
    // Apply exponential decay to historical data
    symbolData.regime_history.forEach(record => {
      const daysSince = (now - record.timestamp) / (24 * 60 * 60 * 1000);
      const weight = Math.pow(this.learningConfig.decay.strategyMemory.decayRate, daysSince);
      
      weightedTotal += weight;
      if (record.outcome === 'loss') {
        weightedFailures += weight;
      }
    });
    
    return weightedTotal > 0 ? weightedFailures / weightedTotal : 0;
  }

  // Update cross-strategy correlations
  updateStrategyCorrelations(trade) {
    const strategies = Array.from(this.observations.strategies.keys());
    
    strategies.forEach(otherStrategy => {
      if (otherStrategy === trade.strategy) return;
      
      const pairKey = [trade.strategy, otherStrategy].sort().join('_');
      if (!this.observations.correlations.has(pairKey)) {
        this.observations.correlations.set(pairKey, {
          overlapping_symbols: new Set(),
          performance_correlation: [],
          inverse_success: 0 // When one fails, does the other succeed?
        });
      }
      
      const correlation = this.observations.correlations.get(pairKey);
      
      // Track if both strategies have traded this symbol
      const otherStratData = this.observations.strategies.get(otherStrategy);
      if (otherStratData && 
          (otherStratData.symbols_failed.has(trade.symbol) || 
           otherStratData.symbols_succeeded.has(trade.symbol))) {
        correlation.overlapping_symbols.add(trade.symbol);
        
        // If enough overlap, start tracking inverse success
        if (correlation.overlapping_symbols.size >= this.learningConfig.correlations.minOverlap) {
          // This is simplified - in production you'd track actual correlation
          if (trade.pnl <= 0 && otherStratData.symbols_succeeded.has(trade.symbol)) {
            correlation.inverse_success++;
          }
        }
      }
    });
  }

  // Get best alternative strategy (flathead → phillips)
  getBestAlternativeStrategy(failedStrategy, symbol) {
    const correlations = new Map();
    
    this.observations.correlations.forEach((data, pairKey) => {
      if (pairKey.includes(failedStrategy) && 
          data.overlapping_symbols.has(symbol) &&
          data.overlapping_symbols.size >= this.learningConfig.correlations.minOverlap) {
        
        const otherStrategy = pairKey.split('_').find(s => s !== failedStrategy);
        const inverseScore = data.inverse_success / data.overlapping_symbols.size;
        
        if (inverseScore > 0.6) { // 60% inverse success rate
          correlations.set(otherStrategy, inverseScore);
        }
      }
    });
    
    // Return best alternative
    if (correlations.size > 0) {
      const sorted = Array.from(correlations.entries()).sort((a, b) => b[1] - a[1]);
      return {
        strategy: sorted[0][0],
        confidence: sorted[0][1],
        reason: `Inverse correlation with ${failedStrategy}`
      };
    }
    
    return null;
  }

  // Market regime detection
  async startRegimeDetection() {
    // Update regime every 30 minutes
    setInterval(() => this.detectMarketRegime(), 30 * 60 * 1000);
    await this.detectMarketRegime();
  }

  async detectMarketRegime() {
    try {
      // Fetch market data
      const marketData = await this.fetchMarketIndicators();
      
      const oldRegime = { ...this.currentRegime };
      
      // Detect trend
      if (marketData.spy_change_20d > 0.05) {
        this.currentRegime.trend = 'bull';
      } else if (marketData.spy_change_20d < -0.05) {
        this.currentRegime.trend = 'bear';
      } else {
        this.currentRegime.trend = 'neutral';
      }
      
      // Detect volatility
      if (marketData.vix > 30) {
        this.currentRegime.volatility = 'high';
      } else if (marketData.vix < 15) {
        this.currentRegime.volatility = 'low';
      } else {
        this.currentRegime.volatility = 'normal';
      }
      
      // Detect liquidity
      if (marketData.volume_ratio < 0.7) {
        this.currentRegime.liquidity = 'low';
      } else if (marketData.volume_ratio > 1.3) {
        this.currentRegime.liquidity = 'high';
      } else {
        this.currentRegime.liquidity = 'normal';
      }
      
      this.currentRegime.lastUpdate = new Date();
      
      // Check if regime changed
      if (oldRegime.trend !== this.currentRegime.trend ||
          oldRegime.volatility !== this.currentRegime.volatility) {
        console.log('[Learning] Market regime changed:', this.currentRegime);
        this.emit('regime_change', this.currentRegime);
        
        // Reduce cooldowns on regime change
        this.adjustCooldownsForRegimeChange();
      }
      
    } catch (error) {
      console.error('[Learning] Regime detection error:', error.message);
    }
  }

  async fetchMarketIndicators() {
    // Simplified - in production, fetch real data
    try {
      const response = await fetch('http://localhost:4000/api/quotes?symbols=SPY,VIX');
      const data = await response.json();
      
      return {
        spy_change_20d: 0.02, // Placeholder
        vix: data.VIX?.last || 20,
        volume_ratio: 1.0 // Placeholder
      };
    } catch (error) {
      return {
        spy_change_20d: 0,
        vix: 20,
        volume_ratio: 1.0
      };
    }
  }

  // Adjust cooldowns when regime changes
  adjustCooldownsForRegimeChange() {
    this.observations.symbols.forEach((data, symbol) => {
      if (data.cooling_until && data.cooling_until > new Date()) {
        // Reduce cooldown by regime adjustment factor
        const reduction = (data.cooling_until - new Date()) * 
          (1 - this.learningConfig.decay.symbolCooldown.regimeAdjust);
        data.cooling_until = new Date(Date.now() + reduction);
        console.log(`[Learning] Reduced cooldown for ${symbol} due to regime change`);
      }
    });
  }

  // Get regime-specific adjustment multiplier
  getRegimeAdjustment(regime) {
    let multiplier = 1.0;
    
    const adjustments = this.learningConfig.regimes.adjustments;
    
    if (regime.volatility === 'high' && adjustments.highVolatility) {
      multiplier *= adjustments.highVolatility.thresholdMultiplier;
    }
    
    if (regime.trend === 'bear' && adjustments.bearMarket) {
      multiplier *= adjustments.bearMarket.newsWeightReduction;
    }
    
    if (regime.liquidity === 'low' && adjustments.lowLiquidity) {
      multiplier *= adjustments.lowLiquidity.spreadTolerance;
    }
    
    return multiplier;
  }

  // Check and trigger recalibration if needed
  checkRecalibrationTriggers() {
    if (this.recalibrationMode) return; // Already recalibrating
    
    const recentTrades = this.getRecentTrades(50);
    if (recentTrades.length < 10) return; // Not enough data
    
    const triggers = this.learningConfig.recalibration.triggers;
    
    // Check losing streak
    let consecutiveLosses = 0;
    for (let i = recentTrades.length - 1; i >= 0; i--) {
      if (recentTrades[i].pnl <= 0) {
        consecutiveLosses++;
      } else {
        break;
      }
    }
    
    if (consecutiveLosses >= triggers.losingStreak) {
      this.triggerRecalibration('Losing streak detected');
      return;
    }
    
    // Check drawdown
    const peakEquity = Math.max(...recentTrades.map(t => t.equity_after));
    const currentEquity = recentTrades[recentTrades.length - 1].equity_after;
    const drawdown = (peakEquity - currentEquity) / peakEquity;
    
    if (drawdown >= triggers.drawdownPercent) {
      this.triggerRecalibration(`Drawdown of ${(drawdown * 100).toFixed(1)}% detected`);
      return;
    }
  }

  triggerRecalibration(reason) {
    console.log(`[Learning] Entering recalibration mode: ${reason}`);
    this.recalibrationMode = true;
    this.recalibrationEndTime = new Date(
      Date.now() + 
      this.learningConfig.recalibration.actions.reviewPeriod * 24 * 60 * 60 * 1000
    );
    
    this.emit('recalibration_triggered', {
      reason,
      endTime: this.recalibrationEndTime,
      adjustments: this.learningConfig.recalibration.actions
    });
  }

  // Get current regime
  getCurrentRegime() {
    return { ...this.currentRegime };
  }

  // Get recent trades (stub - would connect to real recorder)
  getRecentTrades(count) {
    // This would fetch from the actual performance recorder
    return [];
  }

  // Generate learning report
  generateLearningReport() {
    const report = {
      timestamp: new Date(),
      regime: this.getCurrentRegime(),
      recalibrating: this.recalibrationMode,
      learned_rules: [],
      strategy_insights: [],
      symbol_cooldowns: [],
      correlations: []
    };
    
    // Strategy insights
    this.observations.strategies.forEach((data, strategy) => {
      if (data.samples >= this.learningConfig.minimumSamples.strategy) {
        const winRate = data.successes / data.samples;
        report.strategy_insights.push({
          strategy,
          win_rate: winRate,
          sample_size: data.samples,
          regime_performance: data.regime_performance,
          confidence: winRate > 0.6 ? 'HIGH' : winRate > 0.4 ? 'MEDIUM' : 'LOW'
        });
      }
    });
    
    // Symbol cooldowns
    this.observations.symbols.forEach((data, symbol) => {
      if (data.cooling_until && data.cooling_until > new Date()) {
        report.symbol_cooldowns.push({
          symbol,
          cooling_until: data.cooling_until,
          failure_rate: data.failures / data.attempts,
          attempts: data.attempts
        });
      }
    });
    
    // Cross-strategy correlations
    this.observations.correlations.forEach((data, pairKey) => {
      if (data.overlapping_symbols.size >= this.learningConfig.correlations.minOverlap) {
        const inverseRate = data.inverse_success / data.overlapping_symbols.size;
        if (inverseRate > 0.5) {
          report.correlations.push({
            strategies: pairKey.split('_'),
            inverse_correlation: inverseRate,
            shared_symbols: data.overlapping_symbols.size,
            recommendation: `When ${pairKey.split('_')[0]} fails, try ${pairKey.split('_')[1]}`
          });
        }
      }
    });
    
    return report;
  }
}

module.exports = EnhancedPerformanceRecorder;
