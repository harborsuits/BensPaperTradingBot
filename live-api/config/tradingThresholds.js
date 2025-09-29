'use strict';

/**
 * Trading Thresholds Configuration
 * Centralized configuration for all trading decision thresholds
 */

module.exports = {
  // Brain scoring thresholds (0-1 scale)
  brainScore: {
    // Original thresholds: 0.94 buy, 0.88 sell (extremely restrictive)
    // Adjusted for more reasonable trading activity with news awareness
  buyThreshold: 0.35,    // TESTING MODE: Very aggressive for more trades
  sellThreshold: 0.30,   // TESTING MODE: Lower sell threshold 
  minConfidence: 0.25,   // TESTING MODE: Accept lower confidence signals
    
    // News-aware thresholds
    newsBoostThreshold: 0.02,  // If news nudge > 2%, use special thresholds
    newsBuyThreshold: 0.54,    // Buy at 54% with significant positive news
    newsSellThreshold: 0.38,   // Sell at 38% with significant negative news
    
    // Position-specific thresholds
    exitWinnerThreshold: 0.4,  // Exit winning positions if score drops below 40%
    exitLoserThreshold: 0.5,   // Exit losing positions if score drops below 50%
  },
  
  // AutoLoop specific settings
  autoLoop: {
    scanInterval: 30000,        // 30 seconds
    maxPositionsPerSymbol: 1,   // Only one position per symbol
    maxTotalPositions: 10,      // Maximum 10 concurrent positions
  },
  
  // Risk management
  risk: {
    maxPositionSizePercent: 0.05,  // 5% of equity per position
    maxDailyLossPercent: 0.02,     // 2% daily loss limit
    stopLossPercent: 0.015,        // 1.5% stop loss
    profitTargetPercent: 0.025,    // 2.5% profit target
  },
  
  // Expected value thresholds
  expectedValue: {
    minEVForEntry: 0.001,  // Minimum 0.1% expected value to enter
    minEVForHold: -0.001,  // Hold unless EV is worse than -0.1%
  },
  
  // Market conditions adjustments
  marketConditions: {
    volatileMarket: {
      buyThresholdAdjustment: 0.05,   // Require 5% higher score in volatile markets
      sellThresholdAdjustment: -0.05,  // Sell 5% sooner in volatile markets
    },
    trendingMarket: {
      buyThresholdAdjustment: -0.05,  // Buy 5% easier in trending markets
      sellThresholdAdjustment: 0.05,   // Hold 5% longer in trending markets
    },
  },
  
  // Strategy-specific overrides
  strategyOverrides: {
    'rsi_reversion': {
      buyThreshold: 0.6,   // More conservative for mean reversion
      sellThreshold: 0.5,
    },
    'ma_crossover': {
      buyThreshold: 0.65,  // Standard thresholds for trend following
      sellThreshold: 0.45,
    },
    'news_momentum_v2': {
      buyThreshold: 0.7,   // Higher threshold for news-based trades
      sellThreshold: 0.4,  // Quick exit if momentum fades
    }
  },
  
  // Gradual threshold adjustment (for testing)
  thresholdAdjustment: {
    enabled: true,
    startDate: new Date('2025-09-23'),
    daysSinceStart: Math.floor((Date.now() - new Date('2025-09-23').getTime()) / (1000 * 60 * 60 * 24)),
    // Start conservative, gradually lower thresholds
    buyThresholdDecay: 0.01,  // Lower buy threshold by 1% per day
    maxBuyThresholdReduction: 0.15, // Max 15% reduction (0.94 -> 0.79)
  },
  
  // Get adjusted thresholds based on current conditions
  getAdjustedThresholds: function(strategyId = null, marketCondition = 'neutral', newsNudge = 0) {
    let buyThreshold = this.brainScore.buyThreshold;
    let sellThreshold = this.brainScore.sellThreshold;
    
    // Check if this is an evolved strategy with proven track record
    const isEvolvedStrategy = strategyId && strategyId.includes('evo_');
    const isLiveStrategy = strategyId && strategyId.includes('_live');
    
    // Evolved strategies that made it to live have proven themselves
    if (isEvolvedStrategy && isLiveStrategy) {
      buyThreshold = 0.52; // Trust evolved live strategies more
      sellThreshold = 0.48;
      console.log(`[Thresholds] Using evolved LIVE strategy ${strategyId}, adjusted thresholds: buy=${buyThreshold}, sell=${sellThreshold}`);
    } else if (isEvolvedStrategy) {
      // Still in tournament, be more cautious
      buyThreshold = 0.56;
      sellThreshold = 0.44;
      console.log(`[Thresholds] Using evolved tournament strategy ${strategyId}, adjusted thresholds: buy=${buyThreshold}, sell=${sellThreshold}`);
    }
    
    // Apply news-aware thresholds if news nudge is significant
    if (Math.abs(newsNudge) >= this.brainScore.newsBoostThreshold) {
      if (newsNudge > 0) {
        // Positive news - lower buy threshold to be more aggressive
        buyThreshold = this.brainScore.newsBuyThreshold;
        console.log(`[Thresholds] Positive news detected (nudge: ${newsNudge}), lowering buy threshold to ${buyThreshold}`);
      } else {
        // Negative news - lower sell threshold to exit faster
        sellThreshold = this.brainScore.newsSellThreshold;
        console.log(`[Thresholds] Negative news detected (nudge: ${newsNudge}), lowering sell threshold to ${sellThreshold}`);
      }
    }
    
    // Apply strategy overrides
    if (strategyId && this.strategyOverrides[strategyId]) {
      buyThreshold = this.strategyOverrides[strategyId].buyThreshold;
      sellThreshold = this.strategyOverrides[strategyId].sellThreshold;
    }
    
    // Apply market condition adjustments
    if (marketCondition === 'volatile' && this.marketConditions.volatileMarket) {
      buyThreshold += this.marketConditions.volatileMarket.buyThresholdAdjustment;
      sellThreshold += this.marketConditions.volatileMarket.sellThresholdAdjustment;
    } else if (marketCondition === 'trending' && this.marketConditions.trendingMarket) {
      buyThreshold += this.marketConditions.trendingMarket.buyThresholdAdjustment;
      sellThreshold += this.marketConditions.trendingMarket.sellThresholdAdjustment;
    }
    
    // Apply gradual adjustment if enabled
    if (this.thresholdAdjustment.enabled && this.thresholdAdjustment.daysSinceStart > 0) {
      const reduction = Math.min(
        this.thresholdAdjustment.buyThresholdDecay * this.thresholdAdjustment.daysSinceStart,
        this.thresholdAdjustment.maxBuyThresholdReduction
      );
      buyThreshold -= reduction;
    }
    
    // Kickstart feature: Lower threshold for first trade of the day to start learning cycle
    const today = new Date().toDateString();
    const tradesKey = `trades_${today}`;
    if (!global[tradesKey]) {
      global[tradesKey] = 0;
    }
    if (global[tradesKey] === 0) {
      buyThreshold = Math.min(buyThreshold, 0.48); // Lower to 48% for first trade
      console.log(`[Thresholds] First trade of the day - lowering buy threshold to ${buyThreshold} to kickstart learning`);
    }
    
    // Ensure thresholds stay within bounds (TESTING MODE: allow lower)
    buyThreshold = Math.max(0.25, Math.min(0.95, buyThreshold)); // Allow down to 25% for testing
    sellThreshold = Math.max(0.20, Math.min(0.8, sellThreshold)); // Allow down to 20% for testing
    
    return {
      buyThreshold,
      sellThreshold,
      minConfidence: this.brainScore.minConfidence
    };
  }
};

