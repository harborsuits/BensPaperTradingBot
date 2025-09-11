/**
 * EvoTester Optimization Enhancements
 * Performance and quality improvements for evolution process
 */

class EvoTesterOptimizations {
  constructor() {
    this.indicatorCache = new Map();
    this.noveltyTracker = new Map();
    this.adversarialHoldout = {
      segments: ['high_volatility', 'earnings_season', 'market_crash'],
      penalties: new Map(),
      performance: new Map()
    };
    this.correlationGuard = {
      activeBots: new Map(),
      correlationMatrix: new Map(),
      maxCorrelation: 0.75
    };
  }

  /**
   * INDICATOR CACHING: Cache expensive indicator calculations
   * Reduces CPU cost by reusing calculations across candidates
   */
  getCachedIndicator(symbol, indicator, timeframe, params = {}) {
    const cacheKey = `${symbol}_${indicator}_${timeframe}_${JSON.stringify(params)}`;

    if (this.indicatorCache.has(cacheKey)) {
      const cached = this.indicatorCache.get(cacheKey);
      const age = Date.now() - cached.timestamp;

      // Cache for 5 minutes during evolution
      if (age < 5 * 60 * 1000) {
        return cached.value;
      }
    }

    // Calculate and cache (placeholder - integrate with actual indicator calc)
    const value = this.calculateIndicator(symbol, indicator, timeframe, params);
    this.indicatorCache.set(cacheKey, {
      value,
      timestamp: Date.now(),
      accessCount: 0
    });

    return value;
  }

  calculateIndicator(symbol, indicator, timeframe, params) {
    // Placeholder for actual indicator calculations
    // In real implementation, this would calculate RSI, MACD, etc.
    switch (indicator) {
      case 'rsi':
        return 50 + (Math.random() - 0.5) * 20; // Mock RSI 30-70
      case 'macd':
        return (Math.random() - 0.5) * 2; // Mock MACD signal
      case 'bb_position':
        return (Math.random() - 0.5) * 2; // Mock Bollinger position
      default:
        return Math.random();
    }
  }

  /**
   * NOVELTY PRESSURE: Penalize look-alike winners
   * Prevents overfitting by reducing fitness of similar strategies
   */
  applyNoveltyPressure(population, eliteThreshold = 0.8) {
    const elites = population.filter(candidate => candidate.fitness > eliteThreshold);

    elites.forEach(elite => {
      const similarityKey = this.generateSimilarityKey(elite);

      if (this.noveltyTracker.has(similarityKey)) {
        const count = this.noveltyTracker.get(similarityKey) + 1;
        this.noveltyTracker.set(similarityKey, count);

        // Apply novelty penalty (reduce fitness by 5% per duplicate)
        elite.fitness *= Math.pow(0.95, count - 1);
        elite.noveltyPenalty = count - 1;
      } else {
        this.noveltyTracker.set(similarityKey, 1);
        elite.noveltyPenalty = 0;
      }
    });

    return population;
  }

  generateSimilarityKey(candidate) {
    // Create a similarity fingerprint based on strategy parameters
    const params = candidate.parameters || {};
    const keyParts = [
      params.entrySignal || 'none',
      params.exitSignal || 'none',
      Math.round((params.stopLoss || 0) * 10) / 10,
      Math.round((params.takeProfit || 0) * 10) / 10,
      params.timeframe || '1h'
    ];

    return keyParts.join('_');
  }

  /**
   * ADVERSARIAL HOLDOUT: Test strategies against adverse conditions
   * Uses historical segments with known poor performance
   */
  applyAdversarialHoldout(candidate, historicalData) {
    const results = {};

    this.adversarialHoldout.segments.forEach(segment => {
      const segmentData = this.getSegmentData(historicalData, segment);
      const performance = this.evaluateOnSegment(candidate, segmentData);

      results[segment] = performance;

      // Track performance in adverse conditions
      const key = `${candidate.id}_${segment}`;
      this.adversarialHoldout.performance.set(key, performance);

      // Apply penalty for poor adversarial performance
      if (performance.sharpe < 0) {
        this.adversarialHoldout.penalties.set(key, Math.abs(performance.sharpe) * 0.1);
        candidate.fitness *= (1 - Math.abs(performance.sharpe) * 0.1);
        candidate.adversarialPenalty = (candidate.adversarialPenalty || 0) + Math.abs(performance.sharpe) * 0.1;
      }
    });

    candidate.adversarialResults = results;
    return candidate;
  }

  getSegmentData(historicalData, segment) {
    // Extract relevant historical segments
    switch (segment) {
      case 'high_volatility':
        return historicalData.filter(d => d.volatility > 0.8);
      case 'earnings_season':
        return historicalData.filter(d => d.isEarningsSeason);
      case 'market_crash':
        return historicalData.filter(d => d.drawdown > 0.1);
      default:
        return historicalData;
    }
  }

  evaluateOnSegment(candidate, segmentData) {
    // Evaluate strategy performance on specific segment
    // Placeholder - would implement actual backtest on segment
    const trades = Math.floor(segmentData.length * 0.1); // Assume 10% of bars generate trades
    const wins = Math.floor(trades * (candidate.winRate || 0.55));
    const losses = trades - wins;

    const avgWin = candidate.avgWin || 0.02;
    const avgLoss = candidate.avgLoss || 0.015;

    const totalReturn = (wins * avgWin) - (losses * avgLoss);
    const volatility = Math.sqrt(trades) * 0.02; // Simplified volatility

    return {
      totalReturn,
      sharpe: totalReturn / volatility,
      maxDrawdown: Math.min(totalReturn * -0.5, -0.05),
      trades,
      winRate: wins / trades
    };
  }

  /**
   * CORRELATION GUARD: Prevent over-correlated bot allocation
   * Ensures portfolio diversification by limiting correlated strategies
   */
  checkCorrelationGuard(candidate) {
    const candidateKey = this.generateSimilarityKey(candidate);
    const correlations = [];

    // Check correlation with active bots
    for (const [botId, botKey] of this.correlationGuard.activeBots) {
      const correlation = this.calculateCorrelation(candidateKey, botKey);
      correlations.push({ botId, correlation });

      if (correlation > this.correlationGuard.maxCorrelation) {
        return {
          allowed: false,
          reason: `Too correlated with ${botId} (${(correlation * 100).toFixed(1)}% > ${(this.correlationGuard.maxCorrelation * 100).toFixed(1)}%)`,
          correlation
        };
      }
    }

    return {
      allowed: true,
      correlations,
      maxCorrelation: Math.max(...correlations.map(c => c.correlation), 0)
    };
  }

  calculateCorrelation(key1, key2) {
    // Calculate correlation between two strategy fingerprints
    // Simple implementation based on parameter similarity
    const parts1 = key1.split('_');
    const parts2 = key2.split('_');

    let similarity = 0;
    for (let i = 0; i < Math.min(parts1.length, parts2.length); i++) {
      if (parts1[i] === parts2[i]) {
        similarity += 1;
      }
    }

    // Convert similarity to correlation (0-1 scale)
    return similarity / Math.max(parts1.length, parts2.length);
  }

  registerActiveBot(botId, candidate) {
    const botKey = this.generateSimilarityKey(candidate);
    this.correlationGuard.activeBots.set(botId, botKey);

    // Update correlation matrix
    for (const [otherBotId, otherKey] of this.correlationGuard.activeBots) {
      if (otherBotId !== botId) {
        const correlation = this.calculateCorrelation(botKey, otherKey);
        this.correlationGuard.correlationMatrix.set(`${botId}_${otherBotId}`, correlation);
      }
    }
  }

  unregisterBot(botId) {
    this.correlationGuard.activeBots.delete(botId);

    // Clean up correlation matrix
    const keysToDelete = [];
    for (const key of this.correlationGuard.correlationMatrix.keys()) {
      if (key.includes(botId)) {
        keysToDelete.push(key);
      }
    }
    keysToDelete.forEach(key => this.correlationGuard.correlationMatrix.delete(key));
  }

  /**
   * PERFORMANCE MONITORING: Track optimization effectiveness
   */
  getOptimizationStats() {
    return {
      cacheStats: {
        size: this.indicatorCache.size,
        hitRate: this.calculateCacheHitRate()
      },
      noveltyStats: {
        uniqueStrategies: this.noveltyTracker.size,
        totalEvaluations: Array.from(this.noveltyTracker.values()).reduce((a, b) => a + b, 0)
      },
      adversarialStats: {
        segments: this.adversarialHoldout.segments.length,
        penalties: this.adversarialHoldout.penalties.size,
        worstPerformance: this.getWorstAdversarialPerformance()
      },
      correlationStats: {
        activeBots: this.correlationGuard.activeBots.size,
        maxCorrelation: this.getMaxCorrelation(),
        diversificationScore: this.calculateDiversificationScore()
      }
    };
  }

  calculateCacheHitRate() {
    // Placeholder - would track actual cache hits vs misses
    return 0.75; // Assume 75% hit rate
  }

  getWorstAdversarialPerformance() {
    let worst = { segment: null, sharpe: 0 };
    for (const [key, performance] of this.adversarialHoldout.performance) {
      if (performance.sharpe < worst.sharpe) {
        worst = { segment: key.split('_')[1], sharpe: performance.sharpe };
      }
    }
    return worst;
  }

  getMaxCorrelation() {
    if (this.correlationGuard.correlationMatrix.size === 0) return 0;
    return Math.max(...this.correlationGuard.correlationMatrix.values());
  }

  calculateDiversificationScore() {
    if (this.correlationGuard.correlationMatrix.size === 0) return 1;

    const correlations = Array.from(this.correlationGuard.correlationMatrix.values());
    const avgCorrelation = correlations.reduce((a, b) => a + b, 0) / correlations.length;

    // Diversification score: lower correlation = higher score
    return Math.max(0, 1 - avgCorrelation);
  }

  /**
   * CLEANUP: Periodic maintenance
   */
  cleanup() {
    // Clear old indicator cache entries (>1 hour)
    const cutoff = Date.now() - (60 * 60 * 1000);
    for (const [key, value] of this.indicatorCache) {
      if (value.timestamp < cutoff) {
        this.indicatorCache.delete(key);
      }
    }

    // Reset novelty tracker periodically
    if (this.noveltyTracker.size > 1000) {
      this.noveltyTracker.clear();
    }

    console.log('🧹 EvoTester optimizations cleaned up');
  }
}

// Export singleton instance
const evoOptimizations = new EvoTesterOptimizations();

module.exports = { EvoTesterOptimizations, evoOptimizations };
