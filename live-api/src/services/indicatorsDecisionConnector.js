/**
 * Indicators Decision Connector
 *
 * Connects market indicators with decision-making processes.
 * Provides indicator-based signals and integrates with the brain service.
 */

const { MarketIndicatorsService } = require('./marketIndicators');
const { BrainService } = require('./brainService');

class IndicatorsDecisionConnector {
  constructor(config = {}) {
    this.config = {
      signalThresholds: {
        rsi: { oversold: 30, overbought: 70 },
        macd: { signalThreshold: 0.1 },
        bollinger: { deviationThreshold: 2.0 },
        volume: { ratioThreshold: 1.5 }
      },
      regimeAdjustments: {
        bull: { rsi: -5, macd: 0.05 },
        bear: { rsi: 5, macd: -0.05 },
        high_vol: { rsi: 10, macd: 0.1 }
      },
      confidenceWeights: {
        rsi: 0.25,
        macd: 0.30,
        bollinger: 0.20,
        volume: 0.15,
        trend: 0.10
      },
      ...config
    };

    this.indicatorsService = new MarketIndicatorsService();
    this.brainService = new BrainService();
    this.signalCache = new Map();
  }

  /**
   * Generate trading signals based on technical indicators
   */
  async generateIndicatorSignals(symbol, timeframe = '1h') {
    try {
      const cacheKey = `${symbol}_${timeframe}`;

      // Check cache
      const cached = this.signalCache.get(cacheKey);
      if (cached && (Date.now() - cached.timestamp) < 30000) { // 30 seconds
        return cached.signals;
      }

      // Get indicators
      const indicators = await this.indicatorsService.getIndicators(symbol, 100);
      if (!indicators) {
        return this.createEmptySignals(symbol);
      }

      // Get market regime
      const regime = await this.indicatorsService.getMarketRegime(symbol);

      // Generate signals from different indicators
      const signals = {
        symbol,
        timestamp: new Date().toISOString(),
        regime: regime.regime,
        signals: {
          rsi: this.generateRSISignal(indicators.rsi, regime),
          macd: this.generateMACDSignal(indicators.macd, regime),
          bollinger: this.generateBollingerSignal(indicators, regime),
          volume: this.generateVolumeSignal(indicators.volume, regime),
          trend: this.generateTrendSignal(indicators, regime)
        },
        composite: this.generateCompositeSignal(signals.signals, regime)
      };

      // Cache results
      this.signalCache.set(cacheKey, {
        timestamp: Date.now(),
        signals
      });

      return signals;

    } catch (error) {
      console.error(`Failed to generate indicator signals for ${symbol}:`, error);
      return this.createEmptySignals(symbol);
    }
  }

  /**
   * Generate RSI-based signal
   */
  generateRSISignal(rsi, regime) {
    if (!rsi) return { signal: 'neutral', confidence: 0.5, reason: 'No RSI data' };

    const thresholds = this.config.signalThresholds.rsi;
    const regimeAdjustment = this.config.regimeAdjustments[regime.regime.split('_')[0]]?.rsi || 0;

    const adjustedOversold = thresholds.oversold + regimeAdjustment;
    const adjustedOverbought = thresholds.overbought + regimeAdjustment;

    let signal = 'neutral';
    let confidence = 0.6;
    let reason = '';

    if (rsi <= adjustedOversold) {
      signal = 'buy';
      confidence = Math.min(0.9, 1 - (rsi / adjustedOversold) * 0.4);
      reason = `RSI oversold at ${rsi.toFixed(1)} (threshold: ${adjustedOversold})`;
    } else if (rsi >= adjustedOverbought) {
      signal = 'sell';
      confidence = Math.min(0.9, (rsi - adjustedOverbought) / (100 - adjustedOverbought) * 0.4 + 0.6);
      reason = `RSI overbought at ${rsi.toFixed(1)} (threshold: ${adjustedOverbought})`;
    } else {
      confidence = 0.5;
      reason = `RSI neutral at ${rsi.toFixed(1)}`;
    }

    return { signal, confidence, reason };
  }

  /**
   * Generate MACD-based signal
   */
  generateMACDSignal(macd, regime) {
    if (!macd || !macd.macd || !macd.signal) {
      return { signal: 'neutral', confidence: 0.5, reason: 'No MACD data' };
    }

    const regimeAdjustment = this.config.regimeAdjustments[regime.regime.split('_')[0]]?.macd || 0;
    const histogram = macd.histogram || (macd.macd - macd.signal);
    const threshold = this.config.signalThresholds.macd.signalThreshold + regimeAdjustment;

    let signal = 'neutral';
    let confidence = 0.6;
    let reason = '';

    if (histogram > threshold) {
      signal = 'buy';
      confidence = Math.min(0.85, 0.6 + (histogram / threshold) * 0.25);
      reason = `MACD bullish histogram ${histogram.toFixed(4)}`;
    } else if (histogram < -threshold) {
      signal = 'sell';
      confidence = Math.min(0.85, 0.6 + Math.abs(histogram / threshold) * 0.25);
      reason = `MACD bearish histogram ${histogram.toFixed(4)}`;
    } else {
      confidence = 0.5;
      reason = `MACD neutral histogram ${histogram.toFixed(4)}`;
    }

    return { signal, confidence, reason };
  }

  /**
   * Generate Bollinger Bands-based signal
   */
  generateBollingerSignal(indicators, regime) {
    const bollinger = indicators.bollinger;
    const latest = indicators.latest;

    if (!bollinger || !latest) {
      return { signal: 'neutral', confidence: 0.5, reason: 'No Bollinger data' };
    }

    const price = latest.price;
    const upper = bollinger.upper;
    const lower = bollinger.lower;
    const middle = bollinger.middle;

    let signal = 'neutral';
    let confidence = 0.6;
    let reason = '';

    // Calculate position within bands
    const upperDistance = (upper - price) / (upper - middle);
    const lowerDistance = (price - lower) / (middle - lower);

    if (price <= lower) {
      signal = 'buy';
      confidence = Math.min(0.8, 0.7 - lowerDistance * 0.2);
      reason = `Price at lower Bollinger band (${lowerDistance.toFixed(2)} below middle)`;
    } else if (price >= upper) {
      signal = 'sell';
      confidence = Math.min(0.8, 0.7 - upperDistance * 0.2);
      reason = `Price at upper Bollinger band (${upperDistance.toFixed(2)} above middle)`;
    } else if (price < middle && lowerDistance < 0.3) {
      signal = 'buy';
      confidence = 0.65;
      reason = `Price approaching lower band`;
    } else if (price > middle && upperDistance < 0.3) {
      signal = 'sell';
      confidence = 0.65;
      reason = `Price approaching upper band`;
    } else {
      confidence = 0.5;
      reason = `Price within normal Bollinger range`;
    }

    return { signal, confidence, reason };
  }

  /**
   * Generate volume-based signal
   */
  generateVolumeSignal(volume, regime) {
    if (!volume || !volume.volumeRatio) {
      return { signal: 'neutral', confidence: 0.5, reason: 'No volume data' };
    }

    const ratio = volume.volumeRatio;
    const threshold = this.config.signalThresholds.volume.ratioThreshold;

    let signal = 'neutral';
    let confidence = 0.6;
    let reason = '';

    if (ratio > threshold * 1.5) {
      signal = 'strong';
      confidence = Math.min(0.8, 0.6 + (ratio / threshold - 1) * 0.2);
      reason = `High volume: ${ratio.toFixed(1)}x average`;
    } else if (ratio > threshold) {
      signal = 'moderate';
      confidence = 0.65;
      reason = `Above average volume: ${ratio.toFixed(1)}x average`;
    } else if (ratio < 0.7) {
      signal = 'weak';
      confidence = 0.55;
      reason = `Low volume: ${ratio.toFixed(1)}x average`;
    } else {
      confidence = 0.5;
      reason = `Normal volume: ${ratio.toFixed(1)}x average`;
    }

    return { signal, confidence, reason, ratio };
  }

  /**
   * Generate trend-based signal
   */
  generateTrendSignal(indicators, regime) {
    const sma = indicators.sma;
    const ema = indicators.ema;

    if (!sma || !ema) {
      return { signal: 'neutral', confidence: 0.5, reason: 'No trend data' };
    }

    // Use multiple timeframes for trend confirmation
    const shortTrend = ema.ema12 > ema.ema26 ? 'bull' : 'bear';
    const mediumTrend = sma.sma20 > sma.sma50 ? 'bull' : 'bear';
    const longTrend = sma.sma50 > sma.sma200 ? 'bull' : 'bear';

    // Trend strength based on agreement
    const trends = [shortTrend, mediumTrend, longTrend];
    const bullishCount = trends.filter(t => t === 'bull').length;

    let signal = 'neutral';
    let confidence = 0.6;
    let reason = '';

    if (bullishCount >= 2) {
      signal = 'bull';
      confidence = 0.7 + (bullishCount - 2) * 0.1;
      reason = `Bullish trend (${bullishCount}/3 timeframes)`;
    } else if (bullishCount <= 1) {
      signal = 'bear';
      confidence = 0.7 + (2 - bullishCount) * 0.1;
      reason = `Bearish trend (${3 - bullishCount}/3 timeframes)`;
    } else {
      confidence = 0.5;
      reason = `Mixed trend signals (${bullishCount}/3 bullish)`;
    }

    return { signal, confidence, reason, trendStrength: bullishCount / 3 };
  }

  /**
   * Generate composite signal from all indicators
   */
  generateCompositeSignal(individualSignals, regime) {
    const weights = this.config.confidenceWeights;

    let buyScore = 0;
    let sellScore = 0;
    let totalWeight = 0;

    const reasons = [];

    // Aggregate signals
    for (const [indicator, signal] of Object.entries(individualSignals)) {
      const weight = weights[indicator] || 0.1;
      totalWeight += weight;

      if (signal.signal === 'buy' || signal.signal === 'bull') {
        buyScore += signal.confidence * weight;
      } else if (signal.signal === 'sell' || signal.signal === 'bear') {
        sellScore += signal.confidence * weight;
      }

      if (signal.reason) {
        reasons.push(`${indicator.toUpperCase()}: ${signal.reason}`);
      }
    }

    // Normalize scores
    if (totalWeight > 0) {
      buyScore /= totalWeight;
      sellScore /= totalWeight;
    }

    // Determine overall signal
    let signal = 'neutral';
    let confidence = 0.5;

    if (buyScore > sellScore + 0.1) {
      signal = 'buy';
      confidence = Math.min(0.9, buyScore);
    } else if (sellScore > buyScore + 0.1) {
      signal = 'sell';
      confidence = Math.min(0.9, sellScore);
    }

    // Adjust for regime
    const regimeMultiplier = this.getRegimeMultiplier(regime.regime);
    confidence *= regimeMultiplier;

    return {
      signal,
      confidence,
      buyScore: buyScore.toFixed(3),
      sellScore: sellScore.toFixed(3),
      reason: reasons.join('; '),
      regime: regime.regime,
      regimeMultiplier: regimeMultiplier.toFixed(2)
    };
  }

  /**
   * Get regime-based confidence multiplier
   */
  getRegimeMultiplier(regime) {
    const multipliers = {
      'bull_low': 1.1,    // Stronger signals in bull markets
      'bull_medium': 1.0,
      'bull_high': 0.9,   // Weaker signals in high volatility bull markets
      'bear_low': 1.1,    // Stronger signals in bear markets
      'bear_medium': 1.0,
      'bear_high': 0.9,   // Weaker signals in high volatility bear markets
      'neutral_medium': 0.95, // Slightly weaker signals in neutral markets
      'neutral_low': 1.0,
      'neutral_high': 0.85  // Much weaker signals in high volatility neutral markets
    };

    return multipliers[regime] || 1.0;
  }

  /**
   * Create empty signals structure
   */
  createEmptySignals(symbol) {
    return {
      symbol,
      timestamp: new Date().toISOString(),
      regime: 'unknown',
      signals: {
        rsi: { signal: 'neutral', confidence: 0.5, reason: 'No data' },
        macd: { signal: 'neutral', confidence: 0.5, reason: 'No data' },
        bollinger: { signal: 'neutral', confidence: 0.5, reason: 'No data' },
        volume: { signal: 'neutral', confidence: 0.5, reason: 'No data' },
        trend: { signal: 'neutral', confidence: 0.5, reason: 'No data' }
      },
      composite: {
        signal: 'neutral',
        confidence: 0.5,
        buyScore: '0.000',
        sellScore: '0.000',
        reason: 'Insufficient data for analysis',
        regime: 'unknown',
        regimeMultiplier: '1.00'
      }
    };
  }

  /**
   * Get decision context enriched with indicators
   */
  async getDecisionContext(symbol, additionalContext = {}) {
    const signals = await this.generateIndicatorSignals(symbol);
    const regime = await this.indicatorsService.getMarketRegime(symbol);

    return {
      symbol,
      timestamp: new Date().toISOString(),
      indicators: signals,
      regime: regime,
      marketContext: {
        trend: regime.regime.split('_')[0],
        volatility: regime.regime.split('_')[1] || 'medium'
      },
      ...additionalContext
    };
  }

  /**
   * Integrate with brain service for decision-making
   */
  async makeIndicatorEnhancedDecision(decisionContext) {
    // Enrich context with indicators
    const enrichedContext = await this.getDecisionContext(
      decisionContext.symbol,
      decisionContext
    );

    // Get brain decision
    const brainDecision = await this.brainService.makeDecision(enrichedContext);

    // Add indicator analysis to reasoning
    if (enrichedContext.indicators?.composite) {
      brainDecision.reasoning.push(
        `Indicators: ${enrichedContext.indicators.composite.reason}`
      );

      // Adjust confidence based on indicator agreement
      const indicatorSignal = enrichedContext.indicators.composite.signal;
      const indicatorConfidence = enrichedContext.indicators.composite.confidence;

      if (brainDecision.action !== 'no_trade' &&
          indicatorSignal !== 'neutral' &&
          ((brainDecision.action === 'enter' && indicatorSignal === 'buy') ||
           (brainDecision.action === 'exit' && indicatorSignal === 'sell'))) {
        // Agreement - boost confidence
        brainDecision.confidence = Math.min(0.95, brainDecision.confidence + indicatorConfidence * 0.1);
        brainDecision.reasoning.push(`Indicator agreement: +${(indicatorConfidence * 10).toFixed(0)}% confidence`);
      } else if (brainDecision.action !== 'no_trade' && indicatorSignal !== 'neutral') {
        // Disagreement - reduce confidence
        brainDecision.confidence *= 0.9;
        brainDecision.reasoning.push(`Indicator disagreement: -10% confidence`);
      }
    }

    return brainDecision;
  }

  /**
   * Get performance metrics for indicators
   */
  async getIndicatorPerformance(timeframe = '1d') {
    // This would analyze historical indicator signals vs actual price movements
    // For now, return basic health metrics
    return {
      indicatorsHealth: {
        rsi: 'active',
        macd: 'active',
        bollinger: 'active',
        volume: 'active',
        trend: 'active'
      },
      signalsGenerated: this.signalCache.size,
      lastUpdate: new Date().toISOString(),
      cacheHitRate: 0.85 // Would be calculated from actual cache hits
    };
  }

  /**
   * Clear signal cache
   */
  clearCache() {
    this.signalCache.clear();
    this.indicatorsService.clearCache();
  }
}

module.exports = { IndicatorsDecisionConnector };
