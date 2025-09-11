/**
 * AI Model Monitor Service
 *
 * Monitors AI/ML model performance, detects drift, and provides health metrics.
 * Tracks model accuracy, latency, and provides alerts for performance degradation.
 */

const fs = require('fs').promises;
const path = require('path');

class AIModelMonitor {
  constructor(config = {}) {
    this.config = {
      models: ['python_brain', 'ai_scorer', 'news_classifier'],
      driftThreshold: 0.05, // 5% performance drop
      latencyThreshold: 1000, // 1 second
      healthCheckInterval: 300000, // 5 minutes
      metricsRetention: 30 * 24 * 60 * 60 * 1000, // 30 days
      ...config
    };

    this.modelMetrics = new Map();
    this.performanceHistory = new Map();
    this.alerts = [];
    this.lastHealthCheck = 0;

    // Initialize metrics for each model
    for (const model of this.config.models) {
      this.modelMetrics.set(model, {
        status: 'unknown',
        lastUpdate: null,
        accuracy: null,
        latency: null,
        requests: 0,
        errors: 0,
        drift: 0,
        baselineAccuracy: null
      });

      this.performanceHistory.set(model, []);
    }

    this.initialize();
  }

  async initialize() {
    console.log('🤖 AI Model Monitor initialized');

    // Load persisted metrics if available
    await this.loadPersistedMetrics();

    // Start health check loop
    setInterval(() => this.performHealthCheck(), this.config.healthCheckInterval);
  }

  /**
   * Record model prediction and outcome for accuracy tracking
   */
  async recordPrediction(modelName, prediction, actualOutcome, metadata = {}) {
    if (!this.config.models.includes(modelName)) {
      console.warn(`Unknown model: ${modelName}`);
      return;
    }

    const metrics = this.modelMetrics.get(modelName);
    const timestamp = Date.now();

    // Update request count
    metrics.requests++;

    // Calculate accuracy if we have actual outcome
    if (actualOutcome !== undefined) {
      const isCorrect = this.evaluatePrediction(prediction, actualOutcome);
      const accuracy = isCorrect ? 1 : 0;

      // Update rolling accuracy
      const history = this.performanceHistory.get(modelName);
      history.push({
        timestamp,
        prediction,
        actual: actualOutcome,
        accuracy,
        metadata
      });

      // Keep only recent history (last 1000 predictions)
      if (history.length > 1000) {
        history.splice(0, history.length - 1000);
      }

      // Calculate current accuracy
      const recentHistory = history.slice(-100); // Last 100 predictions
      const recentAccuracy = recentHistory.reduce((sum, h) => sum + h.accuracy, 0) / recentHistory.length;

      metrics.accuracy = recentAccuracy;
      metrics.lastUpdate = timestamp;

      // Detect accuracy drift
      if (metrics.baselineAccuracy && metrics.baselineAccuracy > 0) {
        const drift = metrics.baselineAccuracy - recentAccuracy;
        metrics.drift = drift;

        if (Math.abs(drift) > this.config.driftThreshold) {
          await this.createAlert(modelName, 'accuracy_drift', {
            drift: drift.toFixed(4),
            currentAccuracy: recentAccuracy.toFixed(4),
            baselineAccuracy: metrics.baselineAccuracy.toFixed(4)
          });
        }
      } else if (recentHistory.length >= 50) {
        // Set baseline after 50 predictions
        metrics.baselineAccuracy = recentAccuracy;
      }
    }
  }

  /**
   * Record model latency
   */
  async recordLatency(modelName, latencyMs) {
    const metrics = this.modelMetrics.get(modelName);
    if (metrics) {
      metrics.latency = latencyMs;
      metrics.lastUpdate = Date.now();

      // Check latency threshold
      if (latencyMs > this.config.latencyThreshold) {
        await this.createAlert(modelName, 'high_latency', {
          latency: latencyMs,
          threshold: this.config.latencyThreshold
        });
      }
    }
  }

  /**
   * Record model error
   */
  async recordError(modelName, error, metadata = {}) {
    const metrics = this.modelMetrics.get(modelName);
    if (metrics) {
      metrics.errors++;
      metrics.lastUpdate = Date.now();

      await this.createAlert(modelName, 'model_error', {
        error: error.message,
        metadata
      });
    }
  }

  /**
   * Evaluate if prediction was correct
   */
  evaluatePrediction(prediction, actual) {
    // Simple accuracy evaluation - can be customized per model
    if (typeof prediction === 'number' && typeof actual === 'number') {
      // For numerical predictions (e.g., confidence scores)
      return Math.abs(prediction - actual) < 0.1; // Within 10%
    }

    if (typeof prediction === 'string' && typeof actual === 'string') {
      // For categorical predictions
      return prediction.toLowerCase() === actual.toLowerCase();
    }

    if (prediction && typeof prediction === 'object' && actual && typeof actual === 'object') {
      // For object predictions (e.g., decision objects)
      return prediction.action === actual.action;
    }

    // Default: assume correct if prediction exists and actual is truthy
    return Boolean(prediction) && Boolean(actual);
  }

  /**
   * Perform health check on all models
   */
  async performHealthCheck() {
    const now = Date.now();
    if (now - this.lastHealthCheck < this.config.healthCheckInterval) {
      return;
    }

    this.lastHealthCheck = now;

    for (const modelName of this.config.models) {
      try {
        const isHealthy = await this.checkModelHealth(modelName);
        const metrics = this.modelMetrics.get(modelName);

        if (!isHealthy && metrics.status !== 'unhealthy') {
          metrics.status = 'unhealthy';
          await this.createAlert(modelName, 'health_check_failed', {});
        } else if (isHealthy && metrics.status !== 'healthy') {
          metrics.status = 'healthy';
        }

      } catch (error) {
        console.error(`Health check failed for ${modelName}:`, error);
        await this.createAlert(modelName, 'health_check_error', {
          error: error.message
        });
      }
    }

    // Persist metrics
    await this.persistMetrics();
  }

  /**
   * Check if a specific model is healthy
   */
  async checkModelHealth(modelName) {
    try {
      const metrics = this.modelMetrics.get(modelName);
      const now = Date.now();

      // Check if model has been updated recently
      if (!metrics.lastUpdate || (now - metrics.lastUpdate) > 3600000) { // 1 hour
        return false;
      }

      // Check error rate (last 100 requests)
      const history = this.performanceHistory.get(modelName);
      const recentHistory = history.slice(-100);

      if (recentHistory.length > 0) {
        const errorRate = recentHistory.filter(h => h.accuracy === 0).length / recentHistory.length;
        if (errorRate > 0.1) { // 10% error rate
          return false;
        }
      }

      // Model-specific health checks
      switch (modelName) {
        case 'python_brain':
          return await this.checkPythonBrainHealth();
        case 'ai_scorer':
          return await this.checkAIScorerHealth();
        case 'news_classifier':
          return await this.checkNewsClassifierHealth();
        default:
          return true;
      }

    } catch (error) {
      return false;
    }
  }

  /**
   * Check Python brain service health
   */
  async checkPythonBrainHealth() {
    try {
      const response = await fetch('http://localhost:8001/health', {
        timeout: 5000
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Check AI scorer service health
   */
  async checkAIScorerHealth() {
    try {
      const response = await fetch('http://localhost:8009/health', {
        timeout: 5000
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Check news classifier health
   */
  async checkNewsClassifierHealth() {
    // News classifier is part of the main Node.js service
    // Check if news endpoints are responding
    try {
      const response = await fetch('http://localhost:4000/api/news/status', {
        timeout: 5000
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Create an alert for monitoring issues
   */
  async createAlert(modelName, alertType, details) {
    const alert = {
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      model: modelName,
      type: alertType,
      timestamp: new Date().toISOString(),
      details,
      acknowledged: false
    };

    this.alerts.push(alert);

    // Keep only recent alerts (last 100)
    if (this.alerts.length > 100) {
      this.alerts.splice(0, this.alerts.length - 100);
    }

    console.warn(`🚨 AI Model Alert: ${modelName} - ${alertType}`, details);

    // In a real system, this would send notifications
    // await this.sendNotification(alert);
  }

  /**
   * Get current model status and metrics
   */
  getModelStatus(modelName = null) {
    if (modelName) {
      return this.modelMetrics.get(modelName) || null;
    }

    const status = {};
    for (const [name, metrics] of this.modelMetrics) {
      status[name] = {
        status: metrics.status,
        accuracy: metrics.accuracy,
        latency: metrics.latency,
        requests: metrics.requests,
        errors: metrics.errors,
        drift: metrics.drift,
        lastUpdate: metrics.lastUpdate ? new Date(metrics.lastUpdate).toISOString() : null
      };
    }
    return status;
  }

  /**
   * Get performance history for analysis
   */
  getPerformanceHistory(modelName, hours = 24) {
    const history = this.performanceHistory.get(modelName) || [];
    const cutoff = Date.now() - (hours * 60 * 60 * 1000);

    return history.filter(h => h.timestamp > cutoff);
  }

  /**
   * Get active alerts
   */
  getActiveAlerts() {
    return this.alerts.filter(alert => !alert.acknowledged);
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId) {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      alert.acknowledgedAt = new Date().toISOString();
    }
  }

  /**
   * Get comprehensive health report
   */
  getHealthReport() {
    const now = Date.now();
    const report = {
      timestamp: new Date().toISOString(),
      overallHealth: 'healthy',
      models: {},
      alerts: this.getActiveAlerts(),
      summary: {
        totalRequests: 0,
        totalErrors: 0,
        avgAccuracy: 0,
        modelsWithDrift: 0
      }
    };

    let totalAccuracy = 0;
    let accuracyCount = 0;

    for (const [modelName, metrics] of this.modelMetrics) {
      report.models[modelName] = {
        status: metrics.status,
        healthScore: this.calculateHealthScore(metrics),
        accuracy: metrics.accuracy,
        latency: metrics.latency,
        requests: metrics.requests,
        errorRate: metrics.requests > 0 ? metrics.errors / metrics.requests : 0,
        drift: metrics.drift,
        lastUpdate: metrics.lastUpdate ? new Date(metrics.lastUpdate).toISOString() : null,
        timeSinceUpdate: metrics.lastUpdate ? now - metrics.lastUpdate : null
      };

      report.summary.totalRequests += metrics.requests;
      report.summary.totalErrors += metrics.errors;

      if (metrics.accuracy !== null) {
        totalAccuracy += metrics.accuracy;
        accuracyCount++;
      }

      if (Math.abs(metrics.drift || 0) > this.config.driftThreshold) {
        report.summary.modelsWithDrift++;
      }

      // Update overall health
      if (metrics.status === 'unhealthy') {
        report.overallHealth = 'unhealthy';
      } else if (metrics.status === 'unknown' && report.overallHealth === 'healthy') {
        report.overallHealth = 'degraded';
      }
    }

    if (accuracyCount > 0) {
      report.summary.avgAccuracy = totalAccuracy / accuracyCount;
    }

    return report;
  }

  /**
   * Calculate health score for a model (0-100)
   */
  calculateHealthScore(metrics) {
    let score = 100;

    // Deduct for errors
    if (metrics.requests > 0) {
      const errorRate = metrics.errors / metrics.requests;
      score -= errorRate * 50; // Max 50 point deduction for errors
    }

    // Deduct for drift
    if (metrics.drift) {
      score -= Math.abs(metrics.drift) * 200; // 5% drift = 10 point deduction
    }

    // Deduct for high latency
    if (metrics.latency && metrics.latency > this.config.latencyThreshold) {
      const latencyRatio = metrics.latency / this.config.latencyThreshold;
      score -= (latencyRatio - 1) * 20; // Additional deduction for latency
    }

    // Deduct for stale data
    const now = Date.now();
    if (metrics.lastUpdate) {
      const hoursSinceUpdate = (now - metrics.lastUpdate) / (60 * 60 * 1000);
      if (hoursSinceUpdate > 1) {
        score -= Math.min(hoursSinceUpdate * 5, 30); // Max 30 point deduction
      }
    } else {
      score -= 20; // No updates = unhealthy
    }

    return Math.max(0, Math.min(100, score));
  }

  /**
   * Persist metrics to disk
   */
  async persistMetrics() {
    try {
      const metricsDir = path.join(__dirname, '../../data/metrics');
      await fs.mkdir(metricsDir, { recursive: true });

      const metricsData = {
        timestamp: new Date().toISOString(),
        models: Object.fromEntries(this.modelMetrics),
        alerts: this.alerts,
        performanceHistory: Object.fromEntries(this.performanceHistory)
      };

      const filePath = path.join(metricsDir, 'ai_monitor_metrics.json');
      await fs.writeFile(filePath, JSON.stringify(metricsData, null, 2));
    } catch (error) {
      console.error('Failed to persist metrics:', error);
    }
  }

  /**
   * Load persisted metrics from disk
   */
  async loadPersistedMetrics() {
    try {
      const filePath = path.join(__dirname, '../../data/metrics/ai_monitor_metrics.json');

      const data = await fs.readFile(filePath, 'utf8');
      const metrics = JSON.parse(data);

      // Restore metrics
      for (const [modelName, modelMetrics] of Object.entries(metrics.models)) {
        if (this.modelMetrics.has(modelName)) {
          Object.assign(this.modelMetrics.get(modelName), modelMetrics);
        }
      }

      // Restore alerts (only recent ones)
      const recentAlerts = metrics.alerts.filter(alert => {
        const alertTime = new Date(alert.timestamp);
        const cutoff = new Date(Date.now() - this.config.metricsRetention);
        return alertTime > cutoff;
      });
      this.alerts.push(...recentAlerts);

      console.log('📊 Loaded persisted AI monitoring metrics');
    } catch (error) {
      // File doesn't exist or is corrupted - start fresh
      console.log('📊 No persisted metrics found, starting fresh');
    }
  }

  /**
   * Reset model baseline (useful after model updates)
   */
  resetBaseline(modelName) {
    const metrics = this.modelMetrics.get(modelName);
    if (metrics) {
      metrics.baselineAccuracy = null;
      metrics.drift = 0;
      console.log(`🔄 Reset baseline for model: ${modelName}`);
    }
  }

  /**
   * Get model performance trends
   */
  getPerformanceTrends(modelName, hours = 24) {
    const history = this.getPerformanceHistory(modelName, hours);

    if (history.length < 2) {
      return { trend: 'insufficient_data', change: 0 };
    }

    // Calculate accuracy trend
    const firstHalf = history.slice(0, Math.floor(history.length / 2));
    const secondHalf = history.slice(Math.floor(history.length / 2));

    const firstHalfAccuracy = firstHalf.reduce((sum, h) => sum + h.accuracy, 0) / firstHalf.length;
    const secondHalfAccuracy = secondHalf.reduce((sum, h) => sum + h.accuracy, 0) / secondHalf.length;

    const change = secondHalfAccuracy - firstHalfAccuracy;
    const trend = change > 0.01 ? 'improving' :
                 change < -0.01 ? 'degrading' : 'stable';

    return {
      trend,
      change: change.toFixed(4),
      firstHalfAccuracy: firstHalfAccuracy.toFixed(4),
      secondHalfAccuracy: secondHalfAccuracy.toFixed(4),
      dataPoints: history.length
    };
  }
}

module.exports = { AIModelMonitor };
