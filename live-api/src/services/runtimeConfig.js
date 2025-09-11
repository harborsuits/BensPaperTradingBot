/**
 * Runtime Configuration for Safety-Critical Operations
 * Guards all options endpoints with data mode validation
 */

class RuntimeConfig {
  constructor() {
    // Core data mode - production-ready real-only validation
    this.DATA_MODE = 'real_only';

    // Force real data sources in production
    this.FORCE_REAL_QUOTES = true;
    this.FORCE_REAL_TRADES = true;
    this.FORCE_REAL_PORTFOLIO = true;

    // Freshness requirements for real data
    this.FRESH_QUOTE_MS = 5000;  // 5 seconds
    this.FRESH_CHAIN_MS = 5000;  // 5 seconds

    // Clock discipline and audit trail
    this.CLOCK_DISCIPLINE = {
      MAX_OUT_OF_ORDER_MS: 5000,  // Reject snapshots >5s out of order
      REQUIRE_AUDIT_STAMPS: true, // All proofs must include audit stamps
      SERVER_TIME_DRIFT_MAX: 1000, // Max server time drift (1s)
      WORM_MODE: true // Write-Once-Read-Many - no mutations after recording
    };

    // Production safety invariants
    this.SAFETY = {
      ALLOWED_MODES: ['real_only'],
      FORBIDDEN_PATTERNS: ['Math.random', 'FAKE', 'mock', 'synthetic'],
      REQUIRED_FRESHNESS: true,
      NET_DEBIT_ONLY: true,
      CASH_BUFFER_MIN_PCT: 0.05
    };
  }

  /**
   * Validate data mode for safety-critical operations
   */
  validateDataMode(operation = 'unknown') {
    if (this.DATA_MODE !== 'real_only') {
      const error = `INVALID_DATA_MODE: ${this.DATA_MODE} for operation ${operation}. Only 'real_only' allowed in production.`;
      console.error(error);
      throw new Error(error);
    }
    return true;
  }

  /**
   * Check if data is fresh enough for trading decisions
   */
  isFresh(timestamp, type = 'quote') {
    const now = Date.now();
    const age = now - new Date(timestamp).getTime();

    const maxAge = type === 'chain' ? this.FRESH_CHAIN_MS : this.FRESH_QUOTE_MS;

    if (age > maxAge) {
      console.warn(`${type.toUpperCase()}_STALE: age=${age}ms, max=${maxAge}ms`);
      return false;
    }

    return true;
  }

  /**
   * Get HTTP status for stale data (422 Unprocessable Entity)
   */
  getStaleDataResponse(type, age, maxAge) {
    return {
      error: `${type.toUpperCase()}_STALE`,
      message: `${type} data is ${age}ms old, maximum allowed is ${maxAge}ms`,
      code: 'DATA_STALE',
      status: 422
    };
  }

  /**
   * Production safety check for all option endpoints
   */
  enforceProductionSafety(operation, data = {}) {
    this.validateDataMode(operation);

    // Check for forbidden patterns in data
    const dataStr = JSON.stringify(data);
    for (const pattern of this.SAFETY.FORBIDDEN_PATTERNS) {
      if (dataStr.includes(pattern)) {
        throw new Error(`FORBIDDEN_PATTERN_DETECTED: ${pattern} found in ${operation} data`);
      }
    }

    return true;
  }

  /**
   * Generate WORM audit stamp for all proofs and recordings
   */
  generateAuditStamp(ts_feed = null, ts_recv = null) {
    const server_ts = new Date().toISOString();

    // Validate clock discipline
    if (ts_recv && ts_feed) {
      const recvTime = new Date(ts_recv).getTime();
      const feedTime = new Date(ts_feed).getTime();
      const serverTime = new Date(server_ts).getTime();

      // Reject out-of-order snapshots
      if (recvTime < feedTime) {
        throw new Error(`OUT_OF_ORDER_SNAPSHOT: ts_recv(${ts_recv}) < ts_feed(${ts_feed})`);
      }

      // Reject stale snapshots
      const age = serverTime - recvTime;
      if (age > this.CLOCK_DISCIPLINE.MAX_OUT_OF_ORDER_MS) {
        throw new Error(`STALE_SNAPSHOT: age(${age}ms) > max(${this.CLOCK_DISCIPLINE.MAX_OUT_OF_ORDER_MS}ms)`);
      }

      // Check server time drift
      const drift = Math.abs(serverTime - recvTime);
      if (drift > this.CLOCK_DISCIPLINE.SERVER_TIME_DRIFT_MAX) {
        console.warn(`SERVER_TIME_DRIFT: ${drift}ms deviation from received time`);
      }
    }

    return {
      ts_feed: ts_feed || server_ts,
      ts_recv: ts_recv || server_ts,
      server_ts: server_ts,
      commit_hash: process.env.GIT_COMMIT || 'dev',
      policy_hash: this.generatePolicyHash(),
      environment: process.env.NODE_ENV || 'development',
      worm_mode: this.CLOCK_DISCIPLINE.WORM_MODE
    };
  }

  /**
   * Generate policy hash for audit trail
   */
  generatePolicyHash() {
    const crypto = require('crypto');
    const policy = JSON.stringify({
      data_mode: this.DATA_MODE,
      freshness: { quotes: this.FRESH_QUOTE_MS, chains: this.FRESH_CHAIN_MS },
      clock_discipline: this.CLOCK_DISCIPLINE,
      safety: this.SAFETY
    });
    return crypto.createHash('sha256').update(policy).digest('hex').substring(0, 16);
  }
}

// Singleton instance
const runtime = new RuntimeConfig();

module.exports = { RuntimeConfig, runtime };
