/**
 * Circuit Breaker Service
 * 
 * Provides system-wide protection against cascading failures.
 * Monitors error rates, API failures, and system health to automatically
 * halt trading when dangerous conditions are detected.
 */

const EventEmitter = require('events');

class CircuitBreaker extends EventEmitter {
  constructor(config = {}) {
    super();
    
    // Circuit breaker states
    this.states = {
      CLOSED: 'CLOSED',     // Normal operation
      OPEN: 'OPEN',         // Circuit tripped, blocking all operations
      HALF_OPEN: 'HALF_OPEN' // Testing if system has recovered
    };
    
    // Current state
    this.state = this.states.CLOSED;
    this.openedAt = null;
    this.lastFailureTime = null;
    
    // Configuration
    this.config = {
      // Error thresholds
      maxFailuresPerWindow: config.maxFailuresPerWindow || 5,
      windowSizeMs: config.windowSizeMs || 60000, // 1 minute
      
      // Recovery settings
      cooldownPeriodMs: config.cooldownPeriodMs || 300000, // 5 minutes
      halfOpenTestLimit: config.halfOpenTestLimit || 3,
      
      // Specific thresholds
      maxApiErrors: config.maxApiErrors || 3,
      maxOrderFailures: config.maxOrderFailures || 5,
      maxDataStaleness: config.maxDataStaleness || 30000, // 30 seconds
      
      // Emergency thresholds
      maxDrawdownPercent: config.maxDrawdownPercent || 0.05, // 5%
      maxDailyLossPercent: config.maxDailyLossPercent || 0.02, // 2%
      maxPositionLossPercent: config.maxPositionLossPercent || 0.10, // 10%
      
      ...config
    };
    
    // Tracking windows
    this.failures = [];
    this.apiErrors = [];
    this.orderFailures = [];
    this.halfOpenTests = 0;
    
    // System metrics
    this.metrics = {
      totalFailures: 0,
      totalApiErrors: 0,
      totalOrderFailures: 0,
      totalCircuitBreaks: 0,
      lastCircuitBreak: null,
      currentDrawdown: 0,
      dailyPnL: 0
    };
    
    // Start monitoring
    this.monitoringInterval = setInterval(() => this.checkRecovery(), 10000); // Check every 10s
    
    console.log('[CircuitBreaker] Initialized with config:', this.config);
  }
  
  /**
   * Record a failure event
   */
  recordFailure(type, error, context = {}) {
    const now = Date.now();
    const failure = {
      type,
      error: error.message || error,
      context,
      timestamp: now
    };
    
    // Add to appropriate tracking window
    this.failures.push(failure);
    
    switch (type) {
      case 'API_ERROR':
        this.apiErrors.push(failure);
        this.metrics.totalApiErrors++;
        break;
      case 'ORDER_FAILURE':
        this.orderFailures.push(failure);
        this.metrics.totalOrderFailures++;
        break;
    }
    
    this.metrics.totalFailures++;
    
    // Clean old failures outside window
    this.cleanOldFailures(now);
    
    // Check if we should trip the circuit
    this.evaluateCircuit();
    
    // Emit event for monitoring
    this.emit('failure_recorded', failure);
  }
  
  /**
   * Check if an operation should be allowed
   */
  canExecute(operation = 'trade') {
    if (this.state === this.states.CLOSED) {
      return { allowed: true, reason: 'Circuit breaker closed' };
    }
    
    if (this.state === this.states.OPEN) {
      return { 
        allowed: false, 
        reason: `Circuit breaker OPEN since ${this.openedAt}. Cooling down.` 
      };
    }
    
    // Half-open: allow limited testing
    if (this.state === this.states.HALF_OPEN) {
      if (this.halfOpenTests < this.config.halfOpenTestLimit) {
        this.halfOpenTests++;
        return { 
          allowed: true, 
          reason: 'Circuit breaker half-open, testing recovery' 
        };
      } else {
        return { 
          allowed: false, 
          reason: 'Half-open test limit reached' 
        };
      }
    }
    
    return { allowed: false, reason: 'Unknown circuit state' };
  }
  
  /**
   * Record a successful operation
   */
  recordSuccess(operation = 'trade') {
    if (this.state === this.states.HALF_OPEN) {
      // If we're in half-open and operations are succeeding, move to closed
      if (this.halfOpenTests >= this.config.halfOpenTestLimit) {
        this.close();
      }
    }
  }
  
  /**
   * Evaluate if circuit should be tripped
   */
  evaluateCircuit() {
    if (this.state === this.states.OPEN) return;
    
    const now = Date.now();
    let shouldTrip = false;
    let reason = '';
    
    // Check failure rates
    if (this.failures.length >= this.config.maxFailuresPerWindow) {
      shouldTrip = true;
      reason = `Too many failures: ${this.failures.length} in window`;
    }
    
    // Check API errors
    if (this.apiErrors.length >= this.config.maxApiErrors) {
      shouldTrip = true;
      reason = `Too many API errors: ${this.apiErrors.length}`;
    }
    
    // Check order failures
    if (this.orderFailures.length >= this.config.maxOrderFailures) {
      shouldTrip = true;
      reason = `Too many order failures: ${this.orderFailures.length}`;
    }
    
    if (shouldTrip) {
      this.trip(reason);
    }
  }
  
  /**
   * Trip the circuit breaker
   */
  trip(reason) {
    if (this.state === this.states.OPEN) return;
    
    this.state = this.states.OPEN;
    this.openedAt = new Date();
    this.lastFailureTime = Date.now();
    this.metrics.totalCircuitBreaks++;
    this.metrics.lastCircuitBreak = this.openedAt;
    
    console.error(`[CircuitBreaker] 🚨 CIRCUIT BREAKER TRIPPED! Reason: ${reason}`);
    console.error(`[CircuitBreaker] All trading operations halted. Will attempt recovery in ${this.config.cooldownPeriodMs / 1000}s`);
    
    // Emit critical event
    this.emit('circuit_tripped', {
      reason,
      openedAt: this.openedAt,
      metrics: this.getMetrics()
    });
  }
  
  /**
   * Move to half-open state
   */
  halfOpen() {
    this.state = this.states.HALF_OPEN;
    this.halfOpenTests = 0;
    
    console.log('[CircuitBreaker] Moving to HALF-OPEN state, testing recovery...');
    
    this.emit('circuit_half_open', {
      wasOpenFor: Date.now() - this.openedAt.getTime(),
      testLimit: this.config.halfOpenTestLimit
    });
  }
  
  /**
   * Close the circuit (resume normal operation)
   */
  close() {
    const wasOpen = this.state !== this.states.CLOSED;
    this.state = this.states.CLOSED;
    this.halfOpenTests = 0;
    
    // Clear failure windows on recovery
    this.failures = [];
    this.apiErrors = [];
    this.orderFailures = [];
    
    if (wasOpen) {
      console.log('[CircuitBreaker] ✅ Circuit CLOSED - System recovered, resuming normal operation');
      
      this.emit('circuit_closed', {
        recoveryTime: this.openedAt ? Date.now() - this.openedAt.getTime() : 0
      });
    }
  }
  
  /**
   * Check if system has recovered
   */
  checkRecovery() {
    if (this.state !== this.states.OPEN) return;
    
    const now = Date.now();
    const timeSinceOpen = now - this.openedAt.getTime();
    
    if (timeSinceOpen >= this.config.cooldownPeriodMs) {
      this.halfOpen();
    }
  }
  
  /**
   * Update system metrics (called by external monitors)
   */
  updateMetrics(metrics) {
    if (metrics.drawdown !== undefined) {
      this.metrics.currentDrawdown = metrics.drawdown;
      
      // Check emergency drawdown limit
      if (Math.abs(metrics.drawdown) > this.config.maxDrawdownPercent) {
        this.trip(`Drawdown exceeded limit: ${(metrics.drawdown * 100).toFixed(2)}%`);
      }
    }
    
    if (metrics.dailyPnL !== undefined) {
      this.metrics.dailyPnL = metrics.dailyPnL;
      
      // Check daily loss limit
      if (metrics.dailyPnL < -this.config.maxDailyLossPercent) {
        this.trip(`Daily loss exceeded limit: ${(metrics.dailyPnL * 100).toFixed(2)}%`);
      }
    }
  }
  
  /**
   * Check data freshness
   */
  checkDataFreshness(lastQuoteTime, lastBrokerTime) {
    const now = Date.now();
    const quoteStaleness = now - lastQuoteTime;
    const brokerStaleness = now - lastBrokerTime;
    
    if (quoteStaleness > this.config.maxDataStaleness || 
        brokerStaleness > this.config.maxDataStaleness) {
      this.trip(`Data too stale - Quotes: ${quoteStaleness}ms, Broker: ${brokerStaleness}ms`);
    }
  }
  
  /**
   * Clean old failures outside the window
   */
  cleanOldFailures(now) {
    const cutoff = now - this.config.windowSizeMs;
    
    this.failures = this.failures.filter(f => f.timestamp > cutoff);
    this.apiErrors = this.apiErrors.filter(f => f.timestamp > cutoff);
    this.orderFailures = this.orderFailures.filter(f => f.timestamp > cutoff);
  }
  
  /**
   * Get current metrics
   */
  getMetrics() {
    return {
      state: this.state,
      ...this.metrics,
      currentFailures: this.failures.length,
      currentApiErrors: this.apiErrors.length,
      currentOrderFailures: this.orderFailures.length,
      openedAt: this.openedAt,
      uptime: this.state === this.states.CLOSED ? 
        (Date.now() - (this.metrics.lastCircuitBreak || Date.now())) : 0
    };
  }
  
  /**
   * Get current status
   */
  getStatus() {
    return {
      healthy: this.state === this.states.CLOSED,
      state: this.state,
      canTrade: this.canExecute().allowed,
      metrics: this.getMetrics()
    };
  }
  
  /**
   * Force close circuit (admin override)
   */
  forceClose() {
    console.warn('[CircuitBreaker] Force closing circuit breaker (admin override)');
    this.close();
  }
  
  /**
   * Force trip circuit (emergency stop)
   */
  forceTrip(reason = 'Manual emergency stop') {
    console.warn('[CircuitBreaker] Force tripping circuit breaker:', reason);
    this.trip(reason);
  }
  
  /**
   * Cleanup
   */
  stop() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    console.log('[CircuitBreaker] Stopped');
  }
}

module.exports = CircuitBreaker;
