// Learning system configuration
module.exports = {
  // Minimum sample sizes before applying learned rules
  minimumSamples: {
    strategy: 20,        // Min trades per strategy before adjusting weights
    symbol: 15,          // Min attempts per symbol before cooldown
    newsPattern: 25,     // Min news events before trusting pattern
    exitRule: 30,        // Min exits before adjusting stop/target
    regimeSpecific: 10   // Min samples within a regime
  },

  // Confidence thresholds for learning
  confidence: {
    required: 0.8,       // 80% confidence before applying rule
    recalibration: 0.6,  // Below 60% triggers recalibration
    initialWeight: 0.5   // Starting confidence for new patterns
  },

  // Decay and forgetting parameters
  decay: {
    symbolCooldown: {
      halfLife: 7,       // Days until cooldown penalty halves
      minCooldown: 1,    // Minimum days before retry
      maxCooldown: 30,   // Maximum cooldown period
      regimeAdjust: 0.5  // Multiply cooldown by this on regime change
    },
    strategyMemory: {
      windowDays: 30,    // Rolling window for performance
      decayRate: 0.95,   // Daily decay of old observations
      boostRecent: 1.5   // Weight multiplier for last 3 days
    }
  },

  // Cross-strategy correlation learning
  correlations: {
    minOverlap: 10,      // Min shared symbols to learn correlation
    inverseThreshold: -0.6, // When strategies are inversely correlated
    switchDelay: 300,    // Seconds before switching strategies on same symbol
    maxActive: 3         // Max strategies on one symbol simultaneously
  },

  // Evolution guardrails
  evolution: {
    performance: {
      minSharpe: 0.3,    // Minimum Sharpe ratio to survive
      maxDrawdown: 0.15, // Maximum 15% drawdown allowed
      minWinRate: 0.35,  // Minimum 35% win rate
      minTrades: 50      // Must complete 50 trades before breeding
    },
    mutation: {
      maxDeviation: 0.2, // Max 20% parameter change per generation
      eliteProtection: 0.1, // Top 10% immune to mutation
      crossoverRate: 0.7,   // 70% chance of crossover
      mutationRate: 0.1     // 10% chance of random mutation
    }
  },

  // Regime-specific adjustments
  regimes: {
    detection: {
      volatilityWindow: 20,  // Days for volatility calculation
      trendWindow: 50,       // Days for trend detection
      volumeWindow: 10       // Days for volume analysis
    },
    adjustments: {
      highVolatility: {
        thresholdMultiplier: 1.2,  // Raise thresholds in volatile markets
        positionSizeMultiplier: 0.7, // Smaller positions
        stopLossMultiplier: 1.5     // Wider stops
      },
      bearMarket: {
        longBiasReduction: 0.6,     // Reduce long preference
        shortBiasIncrease: 1.4,     // Increase short preference
        newsWeightReduction: 0.8    // Less trust in positive news
      },
      lowLiquidity: {
        spreadTolerance: 0.5,       // Stricter spread requirements
        minVolumeMultiplier: 2.0,   // Higher volume requirements
        exitUrgency: 1.3            // Exit faster
      }
    }
  },

  // Self-recalibration triggers
  recalibration: {
    triggers: {
      losingStreak: 7,          // Consecutive losses
      drawdownPercent: 0.08,    // 8% account drawdown
      accuracyDrop: 0.15,       // 15% drop in prediction accuracy
      regimeChange: true        // Always recalibrate on regime change
    },
    actions: {
      resetConfidence: 0.7,     // Reset overconfident rules to 70%
      expandSearch: 1.5,        // Look 50% wider for opportunities
      tightenRisk: 0.8,         // Reduce position sizes by 20%
      reviewPeriod: 3           // Days of cautious trading
    }
  },

  // Learning report config
  reporting: {
    dailySummary: true,
    includeEvidence: true,
    maxRulesPerReport: 10,
    confidenceThreshold: 0.7
  }
};
