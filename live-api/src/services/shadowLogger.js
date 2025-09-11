/**
 * Shadow Logging Service for Attribution Tracking
 *
 * Tracks all decision-making inputs and outputs for performance analysis,
 * attribution, and continuous improvement without affecting live trading.
 */

const fs = require('fs').promises;
const path = require('path');

class ShadowLogger {
  constructor(config = {}) {
    this.config = {
      enabled: true,
      logDir: path.join(__dirname, '../../logs/shadow'),
      maxLogAge: 30 * 24 * 60 * 60 * 1000, // 30 days
      maxLogSize: 100 * 1024 * 1024, // 100MB
      ...config
    };

    this.sessionId = this.generateSessionId();
    this.decisionLog = [];
    this.performanceMetrics = new Map();
    this.attributionTracker = new Map();

    this.initialize();
  }

  async initialize() {
    if (!this.config.enabled) return;

    try {
      await fs.mkdir(this.config.logDir, { recursive: true });
      await this.cleanupOldLogs();
      console.log('🤫 Shadow Logger initialized');
    } catch (error) {
      console.error('Failed to initialize Shadow Logger:', error);
    }
  }

  generateSessionId() {
    return `shadow_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Log a decision with full context for attribution
   */
  async logDecision(decisionContext) {
    if (!this.config.enabled) return;

    const logEntry = {
      sessionId: this.sessionId,
      timestamp: new Date().toISOString(),
      type: 'decision',
      data: {
        ...decisionContext,
        attributionId: this.generateAttributionId(decisionContext)
      }
    };

    this.decisionLog.push(logEntry);
    this.updateAttributionTracker(logEntry);

    // Async write to avoid blocking
    this.writeToFile(logEntry).catch(error =>
      console.error('Shadow log write failed:', error)
    );

    return logEntry.data.attributionId;
  }

  /**
   * Log decision outcomes for performance analysis
   */
  async logOutcome(attributionId, outcome) {
    if (!this.config.enabled) return;

    const outcomeEntry = {
      sessionId: this.sessionId,
      timestamp: new Date().toISOString(),
      type: 'outcome',
      attributionId,
      data: outcome
    };

    this.updatePerformanceMetrics(attributionId, outcome);
    this.writeToFile(outcomeEntry).catch(error =>
      console.error('Shadow outcome log failed:', error)
    );
  }

  /**
   * Generate unique attribution ID for tracking decision lineage
   */
  generateAttributionId(decisionContext) {
    const components = [
      decisionContext.symbol || 'unknown',
      decisionContext.strategy || 'unknown',
      decisionContext.timestamp || Date.now(),
      Math.random().toString(36).substr(2, 6)
    ];

    return components.join('_');
  }

  /**
   * Update attribution tracker with decision metadata
   */
  updateAttributionTracker(logEntry) {
    const { attributionId, symbol, strategy, confidence } = logEntry.data;

    if (!this.attributionTracker.has(attributionId)) {
      this.attributionTracker.set(attributionId, {
        symbol,
        strategy,
        confidence,
        timestamp: logEntry.timestamp,
        status: 'pending',
        outcomes: []
      });
    }
  }

  /**
   * Update performance metrics for analysis
   */
  updatePerformanceMetrics(attributionId, outcome) {
    if (!this.attributionTracker.has(attributionId)) return;

    const decision = this.attributionTracker.get(attributionId);
    decision.status = 'completed';
    decision.outcomes.push(outcome);

    // Calculate performance metrics
    const metrics = this.calculatePerformanceMetrics(decision);
    this.performanceMetrics.set(attributionId, metrics);
  }

  /**
   * Calculate performance metrics for a decision
   */
  calculatePerformanceMetrics(decision) {
    const outcomes = decision.outcomes;
    if (outcomes.length === 0) return null;

    const latestOutcome = outcomes[outcomes.length - 1];

    return {
      attributionId: decision.attributionId,
      symbol: decision.symbol,
      strategy: decision.strategy,
      confidence: decision.confidence,
      pnl: latestOutcome.pnl || 0,
      pnlPercent: latestOutcome.pnlPercent || 0,
      duration: new Date() - new Date(decision.timestamp),
      successful: (latestOutcome.pnl || 0) > 0,
      timestamp: decision.timestamp
    };
  }

  /**
   * Get performance summary for analysis
   */
  getPerformanceSummary(timeframe = '1h') {
    const now = Date.now();
    const timeframeMs = this.parseTimeframe(timeframe);

    const relevantMetrics = Array.from(this.performanceMetrics.values())
      .filter(metric => (now - new Date(metric.timestamp)) <= timeframeMs);

    if (relevantMetrics.length === 0) {
      return {
        totalDecisions: 0,
        winRate: 0,
        avgPnl: 0,
        avgConfidence: 0,
        timeframe
      };
    }

    const successful = relevantMetrics.filter(m => m.successful).length;
    const totalPnl = relevantMetrics.reduce((sum, m) => sum + m.pnl, 0);
    const avgConfidence = relevantMetrics.reduce((sum, m) => sum + m.confidence, 0) / relevantMetrics.length;

    return {
      totalDecisions: relevantMetrics.length,
      winRate: successful / relevantMetrics.length,
      avgPnl: totalPnl / relevantMetrics.length,
      avgConfidence: avgConfidence,
      timeframe
    };
  }

  /**
   * Get attribution analysis for specific strategies/symbols
   */
  getAttributionAnalysis(filter = {}) {
    const { strategy, symbol, successful } = filter;

    return Array.from(this.attributionTracker.values())
      .filter(decision => {
        if (strategy && decision.strategy !== strategy) return false;
        if (symbol && decision.symbol !== symbol) return false;
        if (successful !== undefined && decision.successful !== successful) return false;
        return true;
      })
      .map(decision => ({
        attributionId: decision.attributionId,
        ...decision,
        performanceMetrics: this.performanceMetrics.get(decision.attributionId)
      }));
  }

  /**
   * Parse timeframe string to milliseconds
   */
  parseTimeframe(timeframe) {
    const unit = timeframe.slice(-1);
    const value = parseInt(timeframe.slice(0, -1));

    switch (unit) {
      case 'h': return value * 60 * 60 * 1000;
      case 'd': return value * 24 * 60 * 60 * 1000;
      case 'w': return value * 7 * 24 * 60 * 60 * 1000;
      case 'm': return value * 30 * 24 * 60 * 60 * 1000;
      default: return 60 * 60 * 1000; // default 1h
    }
  }

  /**
   * Write log entry to file
   */
  async writeToFile(entry) {
    const date = new Date().toISOString().split('T')[0];
    const logFile = path.join(this.config.logDir, `shadow_${date}.jsonl`);

    const line = JSON.stringify(entry) + '\n';

    await fs.appendFile(logFile, line);
    await this.checkLogRotation(logFile);
  }

  /**
   * Check if log file needs rotation
   */
  async checkLogRotation(logFile) {
    try {
      const stats = await fs.stat(logFile);
      if (stats.size > this.config.maxLogSize) {
        const backupFile = logFile.replace('.jsonl', `_${Date.now()}.jsonl`);
        await fs.rename(logFile, backupFile);
      }
    } catch (error) {
      // File doesn't exist yet, no rotation needed
    }
  }

  /**
   * Clean up old log files
   */
  async cleanupOldLogs() {
    try {
      const files = await fs.readdir(this.config.logDir);
      const now = Date.now();

      for (const file of files) {
        if (!file.startsWith('shadow_') || !file.endsWith('.jsonl')) continue;

        const filePath = path.join(this.config.logDir, file);
        const stats = await fs.stat(filePath);

        if (now - stats.mtime.getTime() > this.config.maxLogAge) {
          await fs.unlink(filePath);
        }
      }
    } catch (error) {
      console.error('Log cleanup failed:', error);
    }
  }

  /**
   * Export shadow logs for analysis
   */
  async exportLogs(startDate, endDate) {
    const logs = [];

    try {
      const files = await fs.readdir(this.config.logDir);

      for (const file of files) {
        if (!file.startsWith('shadow_') || !file.endsWith('.jsonl')) continue;

        const fileDate = file.replace('shadow_', '').replace('.jsonl', '');
        if (fileDate < startDate || fileDate > endDate) continue;

        const filePath = path.join(this.config.logDir, file);
        const content = await fs.readFile(filePath, 'utf8');

        const lines = content.trim().split('\n');
        logs.push(...lines.map(line => JSON.parse(line)));
      }
    } catch (error) {
      console.error('Log export failed:', error);
    }

    return logs.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  }
}

module.exports = { ShadowLogger };
