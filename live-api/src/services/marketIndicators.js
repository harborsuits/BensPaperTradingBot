/**
 * Market Indicators Service
 *
 * Computes technical indicators and market regime detection from real market data.
 * Provides indicators for trading decisions and market analysis.
 */

const { recorder: marketRecorder } = require('./marketRecorder');

class MarketIndicatorsService {
  constructor(config = {}) {
    this.config = {
      cacheExpiryMs: 30000, // 30 seconds
      minDataPoints: 50,
      ...config
    };

    this.indicatorsCache = new Map();
    this.regimeCache = new Map();
  }

  /**
   * Get technical indicators for a symbol
   */
  async getIndicators(symbol, lookback = 100) {
    const cacheKey = `${symbol}_${lookback}`;

    // Check cache
    const cached = this.indicatorsCache.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp) < this.config.cacheExpiryMs) {
      return cached.data;
    }

    try {
      // Get recent quotes from MarketRecorder
      const quotes = await marketRecorder.getRecentQuotes(symbol, lookback);

      if (!quotes || quotes.length < this.config.minDataPoints) {
        throw new Error(`Insufficient data for ${symbol}: ${quotes?.length || 0} points`);
      }

      // Compute indicators
      const indicators = this.computeAllIndicators(quotes);

      // Cache result
      this.indicatorsCache.set(cacheKey, {
        timestamp: Date.now(),
        data: indicators
      });

      return indicators;

    } catch (error) {
      console.error(`Failed to compute indicators for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Detect current market regime
   */
  async getMarketRegime(symbol = 'SPY') {
    const cacheKey = `regime_${symbol}`;

    // Check cache
    const cached = this.regimeCache.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp) < this.config.cacheExpiryMs) {
      return cached.data;
    }

    try {
      // Get longer lookback for regime detection
      const quotes = await marketRecorder.getRecentQuotes(symbol, 200);

      if (!quotes || quotes.length < 100) {
        throw new Error(`Insufficient data for regime detection: ${quotes?.length || 0} points`);
      }

      const regime = this.detectMarketRegime(quotes);

      // Cache result
      this.regimeCache.set(cacheKey, {
        timestamp: Date.now(),
        data: regime
      });

      return regime;

    } catch (error) {
      console.error(`Failed to detect market regime for ${symbol}:`, error);
      // Return neutral regime as fallback
      return {
        trend: 'neutral',
        volatility: 'medium',
        regime: 'neutral_medium',
        confidence: 0.5,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Compute all technical indicators
   */
  computeAllIndicators(quotes) {
    const closes = quotes.map(q => q.close);
    const highs = quotes.map(q => q.high);
    const lows = quotes.map(q => q.low);
    const volumes = quotes.map(q => q.volume);

    return {
      symbol: quotes[0]?.symbol || 'unknown',
      timestamp: new Date().toISOString(),

      // Trend Indicators
      sma: {
        sma20: this.calculateSMA(closes, 20),
        sma50: this.calculateSMA(closes, 50),
        sma200: this.calculateSMA(closes, 200)
      },

      ema: {
        ema12: this.calculateEMA(closes, 12),
        ema26: this.calculateEMA(closes, 26),
        ema50: this.calculateEMA(closes, 50)
      },

      // Momentum Indicators
      rsi: this.calculateRSI(closes, 14),
      macd: this.calculateMACD(closes),
      stoch: this.calculateStochastic(highs, lows, closes, 14),

      // Volatility Indicators
      bollinger: this.calculateBollingerBands(closes, 20, 2),
      atr: this.calculateATR(highs, lows, closes, 14),

      // Volume Indicators
      volume: {
        volumeSMA20: this.calculateSMA(volumes, 20),
        volumeRatio: this.calculateVolumeRatio(volumes),
        obv: this.calculateOBV(closes, volumes)
      },

      // Support/Resistance
      pivot: this.calculatePivotPoints(highs[highs.length - 1], lows[lows.length - 1], closes[closes.length - 1]),

      // Market Breadth (if we have multiple symbols)
      breadth: this.calculateMarketBreadth(quotes),

      // Raw data for additional calculations
      latest: {
        price: closes[closes.length - 1],
        volume: volumes[volumes.length - 1],
        high: highs[highs.length - 1],
        low: lows[lows.length - 1]
      }
    };
  }

  /**
   * Detect market regime based on trend and volatility
   */
  detectMarketRegime(quotes) {
    const closes = quotes.map(q => q.close);
    const highs = quotes.map(q => q.high);
    const lows = quotes.map(q => q.low);

    // Calculate trend using EMA slope
    const ema50 = this.calculateEMA(closes, 50);
    const ema200 = this.calculateEMA(closes, 200);

    if (!ema50 || !ema200) {
      return this.createRegimeResult('neutral', 'medium', 0.3);
    }

    // Trend detection: compare recent EMA50 vs EMA200
    const recentEMA50 = ema50.slice(-10).filter(x => x !== null);
    const recentEMA200 = ema200.slice(-10).filter(x => x !== null);

    if (recentEMA50.length < 5 || recentEMA200.length < 5) {
      return this.createRegimeResult('neutral', 'medium', 0.3);
    }

    const avgEMA50 = recentEMA50.reduce((a, b) => a + b, 0) / recentEMA50.length;
    const avgEMA200 = recentEMA200.reduce((a, b) => a + b, 0) / recentEMA200.length;

    const trend = avgEMA50 > avgEMA200 ? 'bull' : avgEMA50 < avgEMA200 ? 'bear' : 'neutral';

    // Volatility detection using ATR percentile
    const atr = this.calculateATR(highs, lows, closes, 14);
    if (atr && atr.length > 0) {
      const recentATR = atr.slice(-20).filter(x => x !== null);
      if (recentATR.length > 0) {
        const avgATR = recentATR.reduce((a, b) => a + b, 0) / recentATR.length;
        const atrPercentile = this.calculateATRPercentile(avgATR, closes);

        let volatility = 'low';
        if (atrPercentile > 80) volatility = 'high';
        else if (atrPercentile > 60) volatility = 'medium';
        else if (atrPercentile > 40) volatility = 'medium';
        else volatility = 'low';

        return this.createRegimeResult(trend, volatility, 0.8);
      }
    }

    return this.createRegimeResult(trend, 'medium', 0.6);
  }

  createRegimeResult(trend, volatility, confidence) {
    return {
      trend,
      volatility,
      regime: `${trend}_${volatility}`,
      confidence,
      timestamp: new Date().toISOString()
    };
  }

  // Technical Indicator Calculations

  calculateSMA(prices, period) {
    if (prices.length < period) return null;

    const result = [];
    for (let i = period - 1; i < prices.length; i++) {
      const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
    return result[result.length - 1]; // Return latest value
  }

  calculateEMA(prices, period) {
    if (prices.length < period) return [];

    const multiplier = 2 / (period + 1);
    const result = [];
    let ema = prices.slice(0, period).reduce((a, b) => a + b, 0) / period;

    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] - ema) * multiplier + ema;
      result.push(ema);
    }

    return result;
  }

  calculateRSI(prices, period = 14) {
    if (prices.length < period + 1) return null;

    const gains = [];
    const losses = [];

    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? -change : 0);
    }

    const avgGain = gains.slice(-period).reduce((a, b) => a + b, 0) / period;
    const avgLoss = losses.slice(-period).reduce((a, b) => a + b, 0) / period;

    if (avgLoss === 0) return 100;

    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const fastEMA = this.calculateEMA(prices, fastPeriod);
    const slowEMA = this.calculateEMA(prices, slowPeriod);

    if (fastEMA.length === 0 || slowEMA.length === 0) return null;

    const macdLine = fastEMA[fastEMA.length - 1] - slowEMA[slowEMA.length - 1];
    const signalLine = this.calculateEMA(fastEMA.slice(-signalPeriod * 2), signalPeriod);
    const histogram = signalLine ? macdLine - signalLine[signalLine.length - 1] : 0;

    return {
      macd: macdLine,
      signal: signalLine ? signalLine[signalLine.length - 1] : null,
      histogram: histogram
    };
  }

  calculateStochastic(highs, lows, closes, period = 14) {
    if (closes.length < period) return null;

    const recentHighs = highs.slice(-period);
    const recentLows = lows.slice(-period);
    const currentClose = closes[closes.length - 1];

    const highest = Math.max(...recentHighs);
    const lowest = Math.min(...recentLows);

    const k = ((currentClose - lowest) / (highest - lowest)) * 100;

    return {
      k: k,
      d: k // Simplified - in practice you'd smooth this
    };
  }

  calculateBollingerBands(prices, period = 20, stdDev = 2) {
    if (prices.length < period) return null;

    const recentPrices = prices.slice(-period);
    const sma = recentPrices.reduce((a, b) => a + b, 0) / period;

    const variance = recentPrices.reduce((sum, price) => {
      return sum + Math.pow(price - sma, 2);
    }, 0) / period;

    const std = Math.sqrt(variance);

    return {
      upper: sma + (stdDev * std),
      middle: sma,
      lower: sma - (stdDev * std),
      bandwidth: (std * stdDev * 2) / sma
    };
  }

  calculateATR(highs, lows, closes, period = 14) {
    if (closes.length < period + 1) return [];

    const trueRanges = [];

    for (let i = 1; i < closes.length; i++) {
      const tr1 = highs[i] - lows[i];
      const tr2 = Math.abs(highs[i] - closes[i - 1]);
      const tr3 = Math.abs(lows[i] - closes[i - 1]);

      trueRanges.push(Math.max(tr1, tr2, tr3));
    }

    // Simple ATR calculation (could be improved with Wilder's smoothing)
    const result = [];
    for (let i = period - 1; i < trueRanges.length; i++) {
      const avg = trueRanges.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
      result.push(avg);
    }

    return result;
  }

  calculateVolumeRatio(volumes) {
    if (volumes.length < 20) return null;

    const recentVolume = volumes[volumes.length - 1];
    const avgVolume = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;

    return recentVolume / avgVolume;
  }

  calculateOBV(closes, volumes) {
    if (closes.length !== volumes.length || closes.length < 2) return null;

    let obv = 0;
    for (let i = 1; i < closes.length; i++) {
      if (closes[i] > closes[i - 1]) {
        obv += volumes[i];
      } else if (closes[i] < closes[i - 1]) {
        obv -= volumes[i];
      }
      // If price unchanged, OBV remains the same
    }

    return obv;
  }

  calculatePivotPoints(high, low, close) {
    const pivot = (high + low + close) / 3;
    return {
      pivot,
      r1: (2 * pivot) - low,
      r2: pivot + (high - low),
      s1: (2 * pivot) - high,
      s2: pivot - (high - low)
    };
  }

  calculateMarketBreadth(quotes) {
    // This would be more sophisticated with multiple symbols
    // For now, return basic breadth metrics
    return {
      advancing: 0, // Would count symbols above their SMA
      declining: 0, // Would count symbols below their SMA
      unchanged: 0,
      advanceDeclineRatio: 1.0
    };
  }

  calculateATRPercentile(atr, closes) {
    // Simplified percentile calculation
    // In practice, you'd compare against historical ATR distribution
    const currentPrice = closes[closes.length - 1];
    const atrPercent = (atr / currentPrice) * 100;

    // Rough percentile mapping based on typical ATR ranges
    if (atrPercent > 3) return 90;
    if (atrPercent > 2) return 75;
    if (atrPercent > 1.5) return 60;
    if (atrPercent > 1) return 45;
    if (atrPercent > 0.5) return 25;
    return 10;
  }

  /**
   * Get bulk indicators for multiple symbols
   */
  async getBulkIndicators(symbols, lookback = 100) {
    const results = {};

    for (const symbol of symbols) {
      try {
        results[symbol] = await this.getIndicators(symbol, lookback);
      } catch (error) {
        console.error(`Failed to get indicators for ${symbol}:`, error);
        results[symbol] = null;
      }
    }

    return results;
  }

  /**
   * Clear caches (useful for testing or memory management)
   */
  clearCache() {
    this.indicatorsCache.clear();
    this.regimeCache.clear();
  }
}

module.exports = { MarketIndicatorsService };
