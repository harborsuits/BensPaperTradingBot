/**
 * Market Indicators Service
 *
 * Computes technical indicators and market regime detection from real market data.
 * Provides indicators for trading decisions and market analysis.
 */

const axios = require('axios');
const rateLimiter = require('../../services/rateLimiter');

class MarketIndicatorsService {
  constructor(config = {}) {
    this.config = {
      cacheExpiryMs: 30000, // 30 seconds
      minDataPoints: 50,
      apiBaseUrl: process.env.API_BASE_URL || 'http://localhost:4000',
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
      // Get real historical bars from Tradier with rate limiting
      const response = await rateLimiter.executeWithLimit(
        'tradier',
        'bars',
        async () => axios.get(`${this.config.apiBaseUrl}/api/bars`, {
          params: {
            symbol,
            timeframe: '1Day',
            limit: lookback
          }
        })
      );
      
      const bars = response.data;
      
      if (!bars || bars.length < this.config.minDataPoints) {
        throw new Error(`Insufficient data for ${symbol}: ${bars?.length || 0} bars`);
      }

      // Compute real indicators from bars
      const indicators = this.computeAllIndicators(bars);

      // Cache result
      this.indicatorsCache.set(cacheKey, {
        timestamp: Date.now(),
        data: indicators
      });

      return indicators;

    } catch (error) {
      console.error(`Failed to compute indicators for ${symbol}:`, error.message);
      
      // Return neutral indicators on error
      return {
        rsi: 50,
        macd: { macd: 0, signal: 0, histogram: 0 },
        ma_20: null,
        ma_50: null,
        volume_ratio: 1.0,
        bb_position: 0.5,
        latest: { price: null, volume: null }
      };
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
      const response = await rateLimiter.executeWithLimit(
        'tradier',
        'bars',
        async () => axios.get(`${this.config.apiBaseUrl}/api/bars`, {
          params: {
            symbol,
            timeframe: '1Day',
            limit: 200
          }
        })
      );
      
      const bars = response.data;

      if (!bars || bars.length < 100) {
        throw new Error(`Insufficient data for regime detection: ${bars?.length || 0} bars`);
      }

      const regime = this.detectMarketRegime(bars);

      // Cache result
      this.regimeCache.set(cacheKey, {
        timestamp: Date.now(),
        data: regime
      });

      return regime;

    } catch (error) {
      console.error(`Failed to detect market regime for ${symbol}:`, error.message);
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
  computeAllIndicators(bars) {
    const closes = bars.map(b => b.c);
    const highs = bars.map(b => b.h);
    const lows = bars.map(b => b.l);
    const volumes = bars.map(b => b.v);

    // RSI
    const rsi = this.calculateRSI(closes, 14);
    
    // MACD
    const macd = this.calculateMACD(closes);
    
    // Moving Averages
    const ma_20 = this.calculateSMA(closes, 20);
    const ma_50 = this.calculateSMA(closes, 50);
    
    // Bollinger Bands
    const bb = this.calculateBollingerBands(closes, 20, 2);
    const lastClose = closes[closes.length - 1];
    const bb_position = bb.upper > bb.lower ? 
      (lastClose - bb.lower) / (bb.upper - bb.lower) : 0.5;
    
    // Volume Ratio
    const avgVolume = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;
    const lastVolume = volumes[volumes.length - 1];
    const volume_ratio = avgVolume > 0 ? lastVolume / avgVolume : 1.0;

    return {
      rsi: rsi[rsi.length - 1] || 50,
      macd: {
        macd: macd.macd[macd.macd.length - 1] || 0,
        signal: macd.signal[macd.signal.length - 1] || 0,
        histogram: macd.histogram[macd.histogram.length - 1] || 0
      },
      ma_20: ma_20[ma_20.length - 1] || null,
      ma_50: ma_50[ma_50.length - 1] || null,
      volume_ratio,
      bb_position,
      latest: {
        price: lastClose,
        volume: lastVolume
      }
    };
  }

  /**
   * Detect market regime from historical data
   */
  detectMarketRegime(bars) {
    const closes = bars.map(b => b.c);
    const volumes = bars.map(b => b.v);
    
    // Calculate returns
    const returns = [];
    for (let i = 1; i < closes.length; i++) {
      returns.push((closes[i] - closes[i-1]) / closes[i-1]);
    }
    
    // Trend detection using 50-day SMA
    const sma50 = this.calculateSMA(closes, 50);
    const lastClose = closes[closes.length - 1];
    const lastSMA50 = sma50[sma50.length - 1];
    const sma50_10daysAgo = sma50[sma50.length - 11];
    
    let trend = 'neutral';
    if (lastClose > lastSMA50 * 1.02 && lastSMA50 > sma50_10daysAgo) {
      trend = 'bull';
    } else if (lastClose < lastSMA50 * 0.98 && lastSMA50 < sma50_10daysAgo) {
      trend = 'bear';
    }
    
    // Volatility detection using standard deviation of returns
    const recentReturns = returns.slice(-20);
    const avgReturn = recentReturns.reduce((a, b) => a + b, 0) / recentReturns.length;
    const variance = recentReturns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / recentReturns.length;
    const stdDev = Math.sqrt(variance);
    const annualizedVol = stdDev * Math.sqrt(252) * 100; // Annualized volatility percentage
    
    let volatility = 'medium';
    if (annualizedVol < 15) {
      volatility = 'low';
    } else if (annualizedVol > 30) {
      volatility = 'high';
    }
    
    // Confidence based on data quality and trend strength
    const trendStrength = Math.abs((lastClose - lastSMA50) / lastSMA50);
    const confidence = Math.min(0.9, 0.5 + trendStrength * 2);
    
    return {
      trend,
      volatility,
      regime: `${trend}_${volatility}`,
      confidence,
      metrics: {
        annualizedVol,
        trendStrength,
        sma50: lastSMA50
      },
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Calculate Simple Moving Average
   */
  calculateSMA(values, period) {
    const result = [];
    for (let i = 0; i < values.length; i++) {
      if (i < period - 1) {
        result.push(null);
      } else {
        const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        result.push(sum / period);
      }
    }
    return result;
  }

  /**
   * Calculate Exponential Moving Average
   */
  calculateEMA(values, period) {
    const result = [];
    const multiplier = 2 / (period + 1);
    
    // Start with SMA for first value
    let ema = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
    result.push(ema);
    
    // Calculate EMA for remaining values
    for (let i = period; i < values.length; i++) {
      ema = (values[i] - ema) * multiplier + ema;
      result.push(ema);
    }
    
    // Pad beginning with nulls
    return new Array(period - 1).fill(null).concat(result);
  }

  /**
   * Calculate RSI (Relative Strength Index)
   */
  calculateRSI(values, period = 14) {
    if (values.length < period + 1) return [50];
    
    const gains = [];
    const losses = [];
    
    for (let i = 1; i < values.length; i++) {
      const diff = values[i] - values[i - 1];
      gains.push(diff > 0 ? diff : 0);
      losses.push(diff < 0 ? -diff : 0);
    }
    
    let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
    
    const result = [];
    
    for (let i = period; i < gains.length; i++) {
      avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
      avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
      
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      const rsi = avgLoss === 0 ? 100 : 100 - (100 / (1 + rs));
      result.push(rsi);
    }
    
    return result;
  }

  /**
   * Calculate MACD
   */
  calculateMACD(values, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const emaFast = this.calculateEMA(values, fastPeriod);
    const emaSlow = this.calculateEMA(values, slowPeriod);
    
    const macdLine = [];
    for (let i = 0; i < values.length; i++) {
      if (emaFast[i] !== null && emaSlow[i] !== null) {
        macdLine.push(emaFast[i] - emaSlow[i]);
      } else {
        macdLine.push(null);
      }
    }
    
    const signalLine = this.calculateEMA(macdLine.filter(v => v !== null), signalPeriod);
    const histogram = [];
    
    let signalIndex = 0;
    for (let i = 0; i < macdLine.length; i++) {
      if (macdLine[i] !== null && signalIndex < signalLine.length) {
        histogram.push(macdLine[i] - signalLine[signalIndex]);
        signalIndex++;
      } else {
        histogram.push(null);
      }
    }
    
    return {
      macd: macdLine,
      signal: signalLine,
      histogram
    };
  }

  /**
   * Calculate Bollinger Bands
   */
  calculateBollingerBands(values, period = 20, stdDev = 2) {
    const sma = this.calculateSMA(values, period);
    const lastSMA = sma[sma.length - 1];
    
    if (!lastSMA) return { upper: 0, middle: 0, lower: 0 };
    
    // Calculate standard deviation
    const recentValues = values.slice(-period);
    const variance = recentValues.reduce((sum, val) => sum + Math.pow(val - lastSMA, 2), 0) / period;
    const std = Math.sqrt(variance);
    
    return {
      upper: lastSMA + (std * stdDev),
      middle: lastSMA,
      lower: lastSMA - (std * stdDev)
    };
  }
}

module.exports = { MarketIndicatorsService };