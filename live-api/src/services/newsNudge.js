/**
 * News Nudge Module - Statistical confidence adjustment for trading plans
 * Applies validated event signals with strict caps and regime awareness
 */

class NewsNudge {
  constructor(reactionStatsBuilder) {
    this.reactionStats = reactionStatsBuilder;
    this.circuitBreaker = {
      active: false,
      lastReset: Date.now(),
      errorRate: 0,
      latencyP95: 0,
      missRate: 0
    };

    // Hard caps and thresholds
    this.CONFIDENCE_CAP = 0.05; // ±5% max nudge
    this.EFFECT_SIZE_CAP = 0.6; // Max z-score effect size
    this.VIX_SHRINK_THRESHOLD = 20; // VIX level for shrink activation
    this.VIX_SHRINK_FACTOR = 0.04; // Shrink per point above threshold
    this.MIN_REGIME_SHRINK = 0.5; // Minimum shrink factor

    // Performance tracking
    this.performance = {
      nudgesApplied: 0,
      avgLatency: 0,
      circuitBreakerTriggers: 0,
      validationFailures: 0
    };
  }

  /**
   * Calculate news-based confidence nudge for a trading plan
   * @param {EventSignal[]} events - Validated event signals
   * @param {Object} marketContext - Current market context (VIX, liquidity, etc.)
   * @param {string} sector - Target sector
   * @param {string} symbol - Target symbol
   * @returns {number} Confidence nudge (-0.05 to +0.05)
   */
  calculateNudge(events, marketContext, sector, symbol) {
    const startTime = Date.now();

    try {
      // Check circuit breaker
      if (this.circuitBreaker.active) {
        console.log('🔌 News nudge circuit breaker active - returning 0');
        return 0;
      }

      // Filter to validated events only
      const validatedEvents = events.filter(event => event.validated && event.effectZ !== undefined);

      if (validatedEvents.length === 0) {
        return 0; // No validated events = no nudge
      }

      // Calculate regime-aware effect size
      let totalEffect = 0;
      let totalConfidence = 0;

      for (const event of validatedEvents) {
        // Get reaction stats for this event type and sector
        const stats = this.reactionStats.getReactionStats(event.type, sector);

        if (!stats || !stats.passesValidation) {
          this.performance.validationFailures++;
          continue; // Skip unvalidated event types
        }

        // Apply effect size cap
        const cappedEffectZ = Math.max(-this.EFFECT_SIZE_CAP,
                                      Math.min(this.EFFECT_SIZE_CAP, event.effectZ));

        // Apply confidence weighting
        const weightedEffect = cappedEffectZ * event.confidence;

        totalEffect += weightedEffect;
        totalConfidence += event.confidence;
      }

      if (totalConfidence === 0) {
        return 0;
      }

      // Average effect across events
      const avgEffect = totalEffect / totalConfidence;

      // Apply regime shrink based on VIX and market conditions
      const regimeShrink = this._calculateRegimeShrink(marketContext);

      // Apply final confidence cap
      let nudge = avgEffect * regimeShrink;
      nudge = Math.max(-this.CONFIDENCE_CAP, Math.min(this.CONFIDENCE_CAP, nudge));

      // Update performance tracking
      this._updatePerformance(startTime, nudge !== 0);

      return nudge;

    } catch (error) {
      console.error('NewsNudge error:', error);
      this._triggerCircuitBreaker('error', error.message);
      return 0;
    }
  }

  /**
   * Calculate regime-aware shrink factor
   * Reduces influence in high-volatility or low-liquidity conditions
   */
  _calculateRegimeShrink(marketContext) {
    let shrink = 1.0;

    // VIX-based shrink
    if (marketContext.vix && marketContext.vix > this.VIX_SHRINK_THRESHOLD) {
      const vixExcess = marketContext.vix - this.VIX_SHRINK_THRESHOLD;
      shrink *= Math.max(this.MIN_REGIME_SHRINK, 1 - (vixExcess * this.VIX_SHRINK_FACTOR));
    }

    // Liquidity-based shrink (simplified)
    if (marketContext.spreadPercent && marketContext.spreadPercent > 0.5) {
      shrink *= 0.8; // Reduce influence in wide spreads
    }

    // Market trend shrink (reduce in strong trends to avoid momentum chasing)
    if (marketContext.trendStrength && marketContext.trendStrength > 2.0) {
      shrink *= 0.7;
    }

    return Math.max(this.MIN_REGIME_SHRINK, shrink);
  }

  /**
   * Validate event signal against statistical gates
   * @param {EventSignal} event - Event signal to validate
   * @param {string} sector - Target sector
   * @returns {boolean} Whether event passes validation
   */
  validateEvent(event, sector) {
    try {
      const stats = this.reactionStats.getReactionStats(event.type, sector);

      if (!stats) {
        return false;
      }

      // Check validation gates
      const passesGates = (
        stats.passesValidation &&
        stats.sampleSize5m >= 100 &&
        stats.last12mThreshold &&
        Math.abs(stats.effectSize5m) >= 0.2 &&
        (!stats.orthogonalityScore || Math.abs(stats.orthogonalityScore) <= 0.3)
      );

      if (passesGates) {
        // Populate effect size from stats
        event.effectZ = stats.effectSize5m;
        event.validated = true;
        event.expectedReturn5m = stats.avgReturn5m;
        event.hitRate = stats.hitRate5m;
      }

      return passesGates;

    } catch (error) {
      console.error('Event validation error:', error);
      return false;
    }
  }

  /**
   * Update performance tracking
   */
  _updatePerformance(startTime, nudgeApplied) {
    const latency = Date.now() - startTime;

    // Update rolling average latency
    const alpha = 0.1;
    this.performance.avgLatency = (1 - alpha) * this.performance.avgLatency + alpha * latency;

    if (nudgeApplied) {
      this.performance.nudgesApplied++;
    }

    // Check for circuit breaker conditions
    if (this.performance.avgLatency > 8000) { // 8ms p95 threshold
      this._triggerCircuitBreaker('latency', `High latency: ${this.performance.avgLatency.toFixed(1)}ms`);
    }
  }

  /**
   * Trigger circuit breaker
   */
  _triggerCircuitBreaker(reason, details) {
    this.circuitBreaker.active = true;
    this.circuitBreaker.lastTrigger = Date.now();
    this.circuitBreaker.triggerReason = reason;
    this.circuitBreaker.triggerDetails = details;
    this.performance.circuitBreakerTriggers++;

    console.log(`🔌 News nudge circuit breaker triggered: ${reason} - ${details}`);

    // Auto-reset after 5 minutes
    setTimeout(() => {
      this.circuitBreaker.active = false;
      console.log('🔄 News nudge circuit breaker auto-reset');
    }, 5 * 60 * 1000);
  }

  /**
   * Get circuit breaker status
   */
  getCircuitBreakerStatus() {
    return {
      active: this.circuitBreaker.active,
      lastTrigger: this.circuitBreaker.lastTrigger,
      reason: this.circuitBreaker.triggerReason,
      details: this.circuitBreaker.triggerDetails,
      timeRemaining: this.circuitBreaker.active ?
        Math.max(0, 5 * 60 * 1000 - (Date.now() - this.circuitBreaker.lastTrigger)) : 0
    };
  }

  /**
   * Get performance statistics
   */
  getPerformanceStats() {
    const validatedEvents = this.reactionStats.getValidatedEventTypes();

    return {
      ...this.performance,
      circuitBreaker: this.getCircuitBreakerStatus(),
      validatedEventTypes: validatedEvents.length,
      validatedEvents: validatedEvents,
      activeEventTypes: validatedEvents.filter(e => e.sampleSize >= 300).length
    };
  }

  /**
   * Force circuit breaker reset (for testing/admin)
   */
  resetCircuitBreaker() {
    this.circuitBreaker.active = false;
    this.circuitBreaker.lastTrigger = null;
    this.circuitBreaker.triggerReason = null;
    this.circuitBreaker.triggerDetails = null;
    console.log('🔄 News nudge circuit breaker manually reset');
  }

  /**
   * Get nudge explanation for UI display
   * @param {number} nudge - The calculated nudge value
   * @param {EventSignal[]} events - Events that contributed
   * @param {Object} marketContext - Market context used
   * @returns {Object} Explanation object for UI
   */
  getNudgeExplanation(nudge, events, marketContext) {
    if (Math.abs(nudge) < 0.001) {
      return {
        nudge: 0,
        reason: 'No validated news events',
        confidence: 0,
        factors: []
      };
    }

    const validatedEvents = events.filter(e => e.validated);
    const regimeShrink = this._calculateRegimeShrink(marketContext);

    const factors = validatedEvents.map(event => ({
      eventType: event.type,
      direction: event.direction > 0 ? 'positive' : 'negative',
      confidence: event.confidence,
      effectSize: event.effectZ,
      expectedReturn: event.expectedReturn5m,
      hitRate: event.hitRate
    }));

    return {
      nudge: Math.round(nudge * 10000) / 100, // Convert to basis points
      reason: nudge > 0 ? 'Positive news reaction expected' : 'Negative news reaction expected',
      confidence: Math.abs(nudge) / this.CONFIDENCE_CAP, // 0-1 scale
      factors,
      regimeShrink,
      vixLevel: marketContext.vix,
      circuitBreakerActive: this.circuitBreaker.active,
      lastUpdated: new Date().toISOString()
    };
  }
}

module.exports = { NewsNudge };
