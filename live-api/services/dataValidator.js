/**
 * Market Data Validator
 * 
 * Ensures all market data is fresh, valid, and safe to trade on.
 * Prevents trading on stale quotes, invalid prices, or suspicious data.
 */

class DataValidator {
  constructor(config = {}) {
    this.config = {
      // Freshness thresholds
      maxQuoteAgeMs: config.maxQuoteAgeMs || 5000, // 5 seconds
      maxBarAgeMs: config.maxBarAgeMs || 300000, // 5 minutes
      maxNewsAgeMs: config.maxNewsAgeMs || 3600000, // 1 hour
      
      // Price validation
      maxSpreadPercent: config.maxSpreadPercent || 0.05, // 5%
      minPrice: config.minPrice || 0.01,
      maxPrice: config.maxPrice || 100000,
      maxPriceChangePercent: config.maxPriceChangePercent || 0.20, // 20% in one update
      
      // Volume validation
      minVolume: config.minVolume || 100,
      minAvgVolume: config.minAvgVolume || 10000,
      
      // Halt detection
      haltDetectionWindowMs: config.haltDetectionWindowMs || 300000, // 5 minutes
      noTradeThresholdMs: config.noTradeThresholdMs || 60000, // 1 minute
      
      ...config
    };
    
    // Track validation failures
    this.validationStats = {
      totalChecks: 0,
      totalFailures: 0,
      failureReasons: {},
      lastFailure: null
    };
    
    console.log('[DataValidator] Initialized with config:', this.config);
  }
  
  /**
   * Validate quote data
   */
  validateQuote(quote, symbol) {
    this.validationStats.totalChecks++;
    
    const errors = [];
    const warnings = [];
    const now = Date.now();
    
    // Check if quote exists
    if (!quote) {
      this.recordFailure('MISSING_QUOTE');
      return { valid: false, errors: ['No quote data provided'], warnings };
    }
    
    // Check quote age
    const timestampValue = quote.timestamp || quote.last_updated || quote.ts;
    let quoteTime = null;
    
    if (timestampValue) {
      quoteTime = new Date(timestampValue).getTime();
    }
    
    // If no timestamp or invalid timestamp, use current time as fallback
    if (!quoteTime || isNaN(quoteTime)) {
      console.warn(`[DataValidator] Invalid or missing timestamp for ${symbol}, using current time`);
      quoteTime = now;
    }
    
    const age = now - quoteTime;
    
    if (age > this.config.maxQuoteAgeMs) {
      errors.push(`Quote too stale: ${age}ms old (max ${this.config.maxQuoteAgeMs}ms)`);
      this.recordFailure('STALE_QUOTE');
    }
    
    // Validate bid/ask
    const bid = parseFloat(quote.bid || quote.bidPrice || 0);
    const ask = parseFloat(quote.ask || quote.askPrice || 0);
    const last = parseFloat(quote.last || quote.lastPrice || 0);
    
    if (bid <= 0 || ask <= 0 || last <= 0) {
      errors.push(`Invalid prices: bid=${bid}, ask=${ask}, last=${last}`);
      this.recordFailure('INVALID_PRICE');
    }
    
    if (bid > ask) {
      errors.push(`Crossed market: bid ${bid} > ask ${ask}`);
      this.recordFailure('CROSSED_MARKET');
    }
    
    // Check spread
    const spread = ask - bid;
    const spreadPercent = spread / ((bid + ask) / 2);
    
    if (spreadPercent > this.config.maxSpreadPercent) {
      warnings.push(`Wide spread: ${(spreadPercent * 100).toFixed(2)}% (max ${this.config.maxSpreadPercent * 100}%)`);
      this.recordFailure('WIDE_SPREAD');
    }
    
    // Check price bounds
    if (last < this.config.minPrice || last > this.config.maxPrice) {
      errors.push(`Price out of bounds: ${last} (allowed ${this.config.minPrice}-${this.config.maxPrice})`);
      this.recordFailure('PRICE_OUT_OF_BOUNDS');
    }
    
    // Check volume
    const volume = parseInt(quote.volume || quote.dayVolume || 0);
    if (volume < this.config.minVolume) {
      warnings.push(`Low volume: ${volume} (min ${this.config.minVolume})`);
    }
    
    // Check for halt indicators
    const bidSize = parseInt(quote.bidSize || quote.bid_size || 0);
    const askSize = parseInt(quote.askSize || quote.ask_size || 0);
    
    if (bidSize === 0 && askSize === 0) {
      errors.push('No market depth - possible halt');
      this.recordFailure('NO_MARKET_DEPTH');
    }
    
    const valid = errors.length === 0;
    
    return {
      valid,
      errors,
      warnings,
      metadata: {
        symbol,
        age,
        bid,
        ask,
        last,
        spread,
        spreadPercent,
        volume,
        timestamp: quoteTime ? new Date(quoteTime).toISOString() : new Date().toISOString()
      }
    };
  }
  
  /**
   * Validate historical bars
   */
  validateBars(bars, symbol) {
    if (!bars || !Array.isArray(bars) || bars.length === 0) {
      return { valid: false, errors: ['No bar data provided'], warnings: [] };
    }
    
    const errors = [];
    const warnings = [];
    const now = Date.now();
    
    // Check most recent bar
    const latestBar = bars[bars.length - 1];
    const barTimestampValue = latestBar.time || latestBar.timestamp || latestBar.t;
    let barTime = null;
    
    if (barTimestampValue) {
      barTime = new Date(barTimestampValue).getTime();
    }
    
    if (!barTime || isNaN(barTime)) {
      console.warn('[DataValidator] Invalid or missing bar timestamp, using current time');
      barTime = now;
    }
    
    const barAge = now - barTime;
    
    if (barAge > this.config.maxBarAgeMs) {
      warnings.push(`Latest bar is stale: ${Math.round(barAge / 60000)} minutes old`);
    }
    
    // Check for data gaps
    let previousTime = null;
    let gaps = 0;
    
    for (const bar of bars) {
      const barTimestamp = bar.time || bar.timestamp || bar.t;
      let time = null;
      
      if (barTimestamp) {
        time = new Date(barTimestamp).getTime();
      }
      
      if (!time || isNaN(time)) {
        console.warn('[DataValidator] Invalid bar timestamp in bar data');
        continue; // Skip this bar
      }
      
      // Validate OHLCV
      const open = parseFloat(bar.open || bar.o);
      const high = parseFloat(bar.high || bar.h);
      const low = parseFloat(bar.low || bar.l);
      const close = parseFloat(bar.close || bar.c);
      const volume = parseInt(bar.volume || bar.v);
      
      if (open <= 0 || high <= 0 || low <= 0 || close <= 0) {
        errors.push(`Invalid OHLC data in bar at ${new Date(time).toISOString()}`);
      }
      
      if (low > high) {
        errors.push(`Invalid bar: low ${low} > high ${high}`);
      }
      
      if (open > high || open < low || close > high || close < low) {
        errors.push(`OHLC inconsistency in bar at ${new Date(time).toISOString()}`);
      }
      
      // Check for gaps (assuming 1-minute bars)
      if (previousTime && time - previousTime > 120000) { // More than 2 minutes
        gaps++;
      }
      
      previousTime = time;
    }
    
    if (gaps > bars.length * 0.1) { // More than 10% gaps
      warnings.push(`Too many data gaps: ${gaps} gaps in ${bars.length} bars`);
    }
    
    return {
      valid: errors.length === 0,
      errors,
      warnings,
      metadata: {
        symbol,
        barCount: bars.length,
        latestBarAge: barAge,
        gaps
      }
    };
  }
  
  /**
   * Validate order parameters before submission
   */
  validateOrderParams(order) {
    const errors = [];
    const warnings = [];
    
    // Required fields
    if (!order.symbol) errors.push('Missing symbol');
    if (!order.side || !['buy', 'sell'].includes(order.side)) errors.push('Invalid side');
    if (!order.quantity || order.quantity <= 0) errors.push('Invalid quantity');
    
    // Price validation for limit orders
    if (order.type === 'limit') {
      if (!order.limitPrice || order.limitPrice <= 0) {
        errors.push('Invalid limit price for limit order');
      }
      
      if (order.limitPrice < this.config.minPrice || order.limitPrice > this.config.maxPrice) {
        errors.push(`Limit price out of bounds: ${order.limitPrice}`);
      }
    }
    
    // Stop price validation
    if (order.stopPrice) {
      if (order.stopPrice <= 0) {
        errors.push('Invalid stop price');
      }
      
      // Check stop price relative to side
      if (order.side === 'buy' && order.limitPrice && order.stopPrice <= order.limitPrice) {
        warnings.push('Buy stop price should be above limit price');
      }
      
      if (order.side === 'sell' && order.limitPrice && order.stopPrice >= order.limitPrice) {
        warnings.push('Sell stop price should be below limit price');
      }
    }
    
    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }
  
  /**
   * Check if symbol is likely halted
   */
  isLikelyHalted(symbol, recentQuotes) {
    if (!recentQuotes || recentQuotes.length === 0) return false;
    
    const now = Date.now();
    const oldestQuote = recentQuotes[0];
    const newestQuote = recentQuotes[recentQuotes.length - 1];
    
    // Check if we have quotes spanning enough time
    const newestTime = newestQuote.timestamp ? new Date(newestQuote.timestamp).getTime() : null;
    const oldestTime = oldestQuote.timestamp ? new Date(oldestQuote.timestamp).getTime() : null;
    
    if (!newestTime || !oldestTime || isNaN(newestTime) || isNaN(oldestTime)) {
      return false; // Can't determine halt status without valid timestamps
    }
    
    const timeSpan = newestTime - oldestTime;
    if (timeSpan < this.config.noTradeThresholdMs) return false;
    
    // Check if price hasn't moved
    const priceChange = Math.abs(newestQuote.last - oldestQuote.last);
    if (priceChange === 0) {
      // Price hasn't moved in the time window
      return true;
    }
    
    // Check if no volume
    const totalVolume = recentQuotes.reduce((sum, q) => sum + (q.volume || 0), 0);
    if (totalVolume === 0) {
      return true;
    }
    
    return false;
  }
  
  /**
   * Comprehensive pre-trade validation
   */
  validatePreTrade(data) {
    const validations = {
      quote: null,
      bars: null,
      order: null,
      overall: true,
      errors: [],
      warnings: []
    };
    
    // Validate quote if provided
    if (data.quote) {
      validations.quote = this.validateQuote(data.quote, data.symbol);
      if (!validations.quote.valid) {
        validations.overall = false;
        validations.errors.push(...validations.quote.errors);
      }
      validations.warnings.push(...validations.quote.warnings);
    }
    
    // Validate bars if provided
    if (data.bars) {
      validations.bars = this.validateBars(data.bars, data.symbol);
      if (!validations.bars.valid) {
        validations.overall = false;
        validations.errors.push(...validations.bars.errors);
      }
      validations.warnings.push(...validations.bars.warnings);
    }
    
    // Validate order parameters if provided
    if (data.order) {
      validations.order = this.validateOrderParams(data.order);
      if (!validations.order.valid) {
        validations.overall = false;
        validations.errors.push(...validations.order.errors);
      }
      validations.warnings.push(...validations.order.warnings);
    }
    
    // Check for halt
    if (data.recentQuotes && this.isLikelyHalted(data.symbol, data.recentQuotes)) {
      validations.overall = false;
      validations.errors.push('Symbol appears to be halted');
      this.recordFailure('LIKELY_HALT');
    }
    
    return validations;
  }
  
  /**
   * Record validation failure
   */
  recordFailure(reason) {
    this.validationStats.totalFailures++;
    this.validationStats.failureReasons[reason] = (this.validationStats.failureReasons[reason] || 0) + 1;
    this.validationStats.lastFailure = {
      reason,
      timestamp: new Date().toISOString()
    };
  }
  
  /**
   * Get validation statistics
   */
  getStats() {
    return {
      ...this.validationStats,
      successRate: this.validationStats.totalChecks > 0 
        ? (this.validationStats.totalChecks - this.validationStats.totalFailures) / this.validationStats.totalChecks
        : 1,
      topFailureReasons: Object.entries(this.validationStats.failureReasons)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([reason, count]) => ({ reason, count }))
    };
  }
}

module.exports = DataValidator;
