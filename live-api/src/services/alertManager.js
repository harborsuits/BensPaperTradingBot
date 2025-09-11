/**
 * Alert Manager - Enterprise-grade alerting system for critical thresholds
 * Integrates with observability endpoints and triggers alerts on violations
 */

const { recorder } = require('./marketRecorder');

class AlertManager {
  constructor() {
    this.alerts = new Map();
    this.thresholds = {
      nbboFreshness: 0.95,      // 95% freshness required
      frictionP90: 0.06,        // $0.06 max friction at 90th percentile
      recorderLag: 2000,        // 2 seconds max lag
      proofPassRate: 0.95       // 95% proof pass rate required
    };

    this.alertCooldown = 5 * 60 * 1000; // 5 minutes between duplicate alerts
    this.activeAlerts = new Set();
  }

  /**
   * Check all critical thresholds and trigger alerts
   */
  async checkCriticalThresholds() {
    const alerts = [];

    try {
      // Check NBBO freshness
      const nbboAlert = await this.checkNBBOFreshness();
      if (nbboAlert) alerts.push(nbboAlert);

      // Check friction compliance
      const frictionAlert = await this.checkFrictionCompliance();
      if (frictionAlert) alerts.push(frictionAlert);

      // Check recorder lag
      const lagAlert = await this.checkRecorderLag();
      if (lagAlert) alerts.push(lagAlert);

      // Check proof health
      const proofAlert = await this.checkProofHealth();
      if (proofAlert) alerts.push(proofAlert);

    } catch (error) {
      console.error('❌ Alert Manager error:', error);
      alerts.push({
        level: 'critical',
        type: 'alert_system_error',
        message: `Alert system error: ${error.message}`,
        timestamp: new Date().toISOString()
      });
    }

    // Process alerts
    alerts.forEach(alert => this.processAlert(alert));

    return alerts;
  }

  /**
   * Check NBBO freshness threshold
   */
  async checkNBBOFreshness() {
    try {
      // Get freshness stats from last hour
      const freshnessStats = await this.getNBBOFreshnessStats();

      if (freshnessStats.percentage < this.thresholds.nbboFreshness) {
        return {
          level: 'warning',
          type: 'nbbo_freshness_low',
          message: `NBBO freshness below threshold: ${(freshnessStats.percentage * 100).toFixed(1)}% < ${(this.thresholds.nbboFreshness * 100).toFixed(1)}%`,
          details: freshnessStats,
          timestamp: new Date().toISOString()
        };
      }
    } catch (error) {
      console.error('NBBO freshness check error:', error);
    }
    return null;
  }

  /**
   * Check friction compliance threshold
   */
  async checkFrictionCompliance() {
    try {
      const frictionStats = await this.getFrictionStats();

      if (frictionStats.p90 > this.thresholds.frictionP90) {
        return {
          level: 'warning',
          type: 'friction_high',
          message: `Friction p90 above threshold: $${frictionStats.p90.toFixed(4)} > $${this.thresholds.frictionP90.toFixed(4)}`,
          details: frictionStats,
          timestamp: new Date().toISOString()
        };
      }
    } catch (error) {
      console.error('Friction compliance check error:', error);
    }
    return null;
  }

  /**
   * Check recorder lag threshold
   */
  async checkRecorderLag() {
    try {
      const lagStats = await this.getRecorderLagStats();

      if (lagStats.averageLag > this.thresholds.recorderLag) {
        return {
          level: 'critical',
          type: 'recorder_lag_high',
          message: `Recorder lag above threshold: ${lagStats.averageLag}ms > ${this.thresholds.recorderLag}ms`,
          details: lagStats,
          timestamp: new Date().toISOString()
        };
      }
    } catch (error) {
      console.error('Recorder lag check error:', error);
    }
    return null;
  }

  /**
   * Check proof health threshold
   */
  async checkProofHealth() {
    try {
      const proofStats = await this.getProofStats();

      if (proofStats.passRate < this.thresholds.proofPassRate) {
        return {
          level: 'critical',
          type: 'proof_failure_rate_high',
          message: `Proof pass rate below threshold: ${(proofStats.passRate * 100).toFixed(1)}% < ${(this.thresholds.proofPassRate * 100).toFixed(1)}%`,
          details: proofStats,
          timestamp: new Date().toISOString()
        };
      }
    } catch (error) {
      console.error('Proof health check error:', error);
    }
    return null;
  }

  /**
   * Process and emit alerts
   */
  processAlert(alert) {
    const alertKey = `${alert.type}_${alert.level}`;

    // Check cooldown to prevent alert spam
    if (this.activeAlerts.has(alertKey)) {
      const lastAlertTime = this.alerts.get(alertKey);
      if (lastAlertTime && (Date.now() - lastAlertTime) < this.alertCooldown) {
        return; // Still in cooldown
      }
    }

    // Record alert
    this.activeAlerts.add(alertKey);
    this.alerts.set(alertKey, Date.now());

    // Log alert
    const logLevel = alert.level === 'critical' ? 'error' : 'warn';
    console[logLevel](`🚨 ALERT [${alert.level.toUpperCase()}]: ${alert.message}`);

    // In production, this would:
    // - Send to monitoring system (DataDog, New Relic, etc.)
    // - Send email/SMS notifications
    // - Update dashboard status
    // - Trigger automated responses

    // Emit alert event for real-time dashboard updates
    if (global.alertsBus) {
      global.alertsBus.emit('alert', alert);
    }
  }

  /**
   * Get NBBO freshness statistics
   */
  async getNBBOFreshnessStats() {
    // Mock implementation - in production would query recorder
    return {
      percentage: 0.98,
      fresh: 95,
      stale: 3,
      veryStale: 2,
      total: 100,
      window: '1h'
    };
  }

  /**
   * Get friction statistics
   */
  async getFrictionStats() {
    // Mock implementation - in production would query fills_snapshot
    return {
      p90: 0.045,
      p95: 0.062,
      average: 0.032,
      totalFills: 150,
      window: '24h'
    };
  }

  /**
   * Get recorder lag statistics
   */
  async getRecorderLagStats() {
    // Mock implementation - in production would measure actual lag
    return {
      averageLag: 1500,
      maxLag: 2800,
      minLag: 800,
      samples: 100
    };
  }

  /**
   * Get proof statistics
   */
  async getProofStats() {
    // Mock implementation - in production would query proof results
    return {
      passRate: 0.97,
      passed: 485,
      failed: 15,
      total: 500,
      window: '24h'
    };
  }

  /**
   * Clear resolved alerts
   */
  clearResolvedAlerts() {
    // In production, this would check if alerts are still active
    // and clear them if resolved
    this.activeAlerts.clear();
  }

  /**
   * Get current alert status
   */
  getAlertStatus() {
    return {
      activeAlerts: Array.from(this.activeAlerts),
      thresholds: this.thresholds,
      lastCheck: new Date().toISOString(),
      alertCount: this.activeAlerts.size
    };
  }
}

module.exports = { AlertManager };
