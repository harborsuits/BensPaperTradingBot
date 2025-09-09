import { evoTesterApi } from './api';

export interface MarketCondition {
  volatility: number;
  regime: 'bull' | 'bear' | 'sideways';
  sentiment: number;
  volume: number;
  newsImpact: number;
  timestamp: string;
}

export interface TriggerRule {
  id: string;
  name: string;
  condition: (marketData: MarketCondition) => boolean;
  description: string;
  active: boolean;
  priority: number;
  cooldownMinutes: number;
  lastTriggered?: Date;
  triggerCount: number;
  successRate: number;
}

export interface AutoTriggerConfig {
  enabled: boolean;
  maxConcurrentExperiments: number;
  capitalPerExperiment: number;
  riskLevel: 'low' | 'medium' | 'high';
}

class AutoTriggerService {
  private rules: Map<string, TriggerRule> = new Map();
  private config: AutoTriggerConfig;
  private monitoringInterval: NodeJS.Timeout | null = null;
  private lastCheck: Date = new Date();
  private activeTriggers: Set<string> = new Set();

  constructor() {
    this.config = {
      enabled: true,
      maxConcurrentExperiments: 3,
      capitalPerExperiment: 500,
      riskLevel: 'low'
    };

    this.initializeRules();
  }

  private initializeRules() {
    // Volatility Spike Trigger
    this.addRule({
      id: 'volatility_spike',
      name: 'Volatility Spike',
      condition: (marketData) => marketData.volatility > 2.5,
      description: 'Triggers when market volatility exceeds 2.5 standard deviations',
      active: true,
      priority: 1,
      cooldownMinutes: 60,
      triggerCount: 0,
      successRate: 0
    });

    // Market Regime Change Trigger
    this.addRule({
      id: 'regime_change',
      name: 'Market Regime Change',
      condition: (marketData) => {
        // This would need to track previous regime and detect changes
        // For now, we'll simulate with a simple condition
        return Math.random() > 0.95; // 5% chance to trigger for demo
      },
      description: 'Triggers when market regime shifts significantly',
      active: true,
      priority: 2,
      cooldownMinutes: 120,
      triggerCount: 0,
      successRate: 0
    });

    // High News Impact Trigger
    this.addRule({
      id: 'high_news_impact',
      name: 'High News Impact',
      condition: (marketData) => marketData.newsImpact > 0.7,
      description: 'Triggers when news sentiment impact exceeds threshold',
      active: true,
      priority: 3,
      cooldownMinutes: 30,
      triggerCount: 0,
      successRate: 0
    });

    // Volume Anomaly Trigger
    this.addRule({
      id: 'volume_anomaly',
      name: 'Volume Anomaly',
      condition: (marketData) => marketData.volume > 1.8,
      description: 'Triggers when trading volume shows unusual patterns',
      active: false,
      priority: 4,
      cooldownMinutes: 45,
      triggerCount: 0,
      successRate: 0
    });

    // Sentiment Reversal Trigger
    this.addRule({
      id: 'sentiment_reversal',
      name: 'Sentiment Reversal',
      condition: (marketData) => {
        // Would need to track sentiment trend and detect reversals
        return marketData.sentiment < -0.6 || marketData.sentiment > 0.6;
      },
      description: 'Triggers when market sentiment shows extreme reversal',
      active: true,
      priority: 5,
      cooldownMinutes: 90,
      triggerCount: 0,
      successRate: 0
    });

    // Research Window Trigger (specific times)
    this.addRule({
      id: 'research_window',
      name: 'Research Window',
      condition: (marketData) => {
        const hour = new Date(marketData.timestamp).getHours();
        // Trigger during low volatility periods (e.g., 2-4 AM ET)
        return hour >= 2 && hour <= 4;
      },
      description: 'Triggers during low-volatility research windows',
      active: true,
      priority: 6,
      cooldownMinutes: 240, // 4 hours
      triggerCount: 0,
      successRate: 0
    });
  }

  addRule(rule: TriggerRule) {
    this.rules.set(rule.id, rule);
  }

  removeRule(ruleId: string) {
    this.rules.delete(ruleId);
  }

  updateRule(ruleId: string, updates: Partial<TriggerRule>) {
    const rule = this.rules.get(ruleId);
    if (rule) {
      this.rules.set(ruleId, { ...rule, ...updates });
    }
  }

  updateConfig(config: Partial<AutoTriggerConfig>) {
    this.config = { ...this.config, ...config };
  }

  getRules(): TriggerRule[] {
    return Array.from(this.rules.values());
  }

  getConfig(): AutoTriggerConfig {
    return this.config;
  }

  startMonitoring() {
    if (this.monitoringInterval) {
      return; // Already monitoring
    }

    this.monitoringInterval = setInterval(() => {
      this.checkTriggers();
    }, 30000); // Check every 30 seconds

    console.log('Auto-trigger monitoring started');
  }

  stopMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
      console.log('Auto-trigger monitoring stopped');
    }
  }

  private async checkTriggers() {
    if (!this.config.enabled) {
      return;
    }

    try {
      // Get current market conditions
      const marketData = await this.getMarketConditions();

      if (!marketData) {
        return;
      }

      // Check each active rule
      const activeRules = Array.from(this.rules.values())
        .filter(rule => rule.active)
        .sort((a, b) => a.priority - b.priority);

      for (const rule of activeRules) {
        if (this.shouldTrigger(rule, marketData)) {
          await this.executeTrigger(rule, marketData);
        }
      }

      this.lastCheck = new Date();
    } catch (error) {
      console.error('Error checking auto-triggers:', error);
    }
  }

  private shouldTrigger(rule: TriggerRule, marketData: MarketCondition): boolean {
    // Check cooldown period
    if (rule.lastTriggered) {
      const timeSinceLastTrigger = Date.now() - rule.lastTriggered.getTime();
      const cooldownMs = rule.cooldownMinutes * 60 * 1000;
      if (timeSinceLastTrigger < cooldownMs) {
        return false;
      }
    }

    // Check maximum concurrent experiments
    if (this.activeTriggers.size >= this.config.maxConcurrentExperiments) {
      return false;
    }

    // Check if this trigger is already active
    if (this.activeTriggers.has(rule.id)) {
      return false;
    }

    // Evaluate the trigger condition
    return rule.condition(marketData);
  }

  private async executeTrigger(rule: TriggerRule, marketData: MarketCondition) {
    try {
      console.log(`Auto-trigger activated: ${rule.name}`);

      // Mark trigger as active
      this.activeTriggers.add(rule.id);

      // Create EvoTester configuration based on trigger
      const config = this.createEvoConfig(rule, marketData);

      // Start the evolution experiment
      const result = await evoTesterApi.startEvoTest(config);

      if (result.success) {
        // Update rule statistics
        const updatedRule = {
          ...rule,
          lastTriggered: new Date(),
          triggerCount: rule.triggerCount + 1
        };
        this.rules.set(rule.id, updatedRule);

        console.log(`Evolution experiment started for trigger: ${rule.name}`);

        // Emit event for UI updates
        this.emitTriggerEvent('trigger_activated', {
          ruleId: rule.id,
          sessionId: result.sessionId,
          marketData,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      console.error(`Failed to execute auto-trigger ${rule.name}:`, error);
    } finally {
      // Remove from active triggers after a delay
      setTimeout(() => {
        this.activeTriggers.delete(rule.id);
      }, 60000); // 1 minute cooldown
    }
  }

  private createEvoConfig(rule: TriggerRule, marketData: MarketCondition) {
    // Base configuration
    const baseConfig = {
      population_size: 100,
      generations: 50,
      mutation_rate: 0.1,
      crossover_rate: 0.8,
      target_asset: 'SPY',
      optimization_metric: 'sharpe',
      symbols: ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA'],
      sentiment_weight: 0.3,
      news_impact_weight: 0.2,
      intelligence_snowball: true,
      auto_triggered: true,
      trigger_rule: rule.id,
      market_context: marketData
    };

    // Customize based on trigger type
    switch (rule.id) {
      case 'volatility_spike':
        return {
          ...baseConfig,
          population_size: 150,
          optimization_metric: 'sortino', // Better for volatile conditions
          volatility_focus: true
        };

      case 'regime_change':
        return {
          ...baseConfig,
          population_size: 120,
          generations: 75,
          regime_adaptation: true
        };

      case 'high_news_impact':
        return {
          ...baseConfig,
          news_impact_weight: 0.5,
          sentiment_weight: 0.4,
          news_focus: true
        };

      case 'volume_anomaly':
        return {
          ...baseConfig,
          optimization_metric: 'calmar',
          volume_focus: true
        };

      case 'sentiment_reversal':
        return {
          ...baseConfig,
          sentiment_weight: 0.6,
          reversal_focus: true
        };

      case 'research_window':
        return {
          ...baseConfig,
          population_size: 200,
          generations: 100,
          research_mode: true
        };

      default:
        return baseConfig;
    }
  }

  private async getMarketConditions(): Promise<MarketCondition | null> {
    try {
      // This would integrate with your market data APIs
      // For now, we'll simulate market data
      const simulatedData: MarketCondition = {
        volatility: Math.random() * 3 + 0.5, // 0.5 to 3.5
        regime: Math.random() > 0.6 ? 'bull' : Math.random() > 0.3 ? 'bear' : 'sideways',
        sentiment: (Math.random() - 0.5) * 2, // -1 to 1
        volume: Math.random() * 2 + 0.5, // 0.5 to 2.5
        newsImpact: Math.random(), // 0 to 1
        timestamp: new Date().toISOString()
      };

      return simulatedData;
    } catch (error) {
      console.error('Error fetching market conditions:', error);
      return null;
    }
  }

  private emitTriggerEvent(eventType: string, data: any) {
    // Emit custom event for UI updates
    const event = new CustomEvent('evo_trigger_event', {
      detail: { type: eventType, data }
    });
    window.dispatchEvent(event);
  }

  // Public methods for external control
  getActiveTriggers(): string[] {
    return Array.from(this.activeTriggers);
  }

  getTriggerStats(ruleId?: string) {
    if (ruleId) {
      return this.rules.get(ruleId);
    }
    return this.getRules().map(rule => ({
      id: rule.id,
      name: rule.name,
      triggerCount: rule.triggerCount,
      successRate: rule.successRate,
      lastTriggered: rule.lastTriggered,
      active: rule.active
    }));
  }

  resetTriggerCooldown(ruleId: string) {
    const rule = this.rules.get(ruleId);
    if (rule) {
      const updatedRule = { ...rule, lastTriggered: undefined };
      this.rules.set(ruleId, updatedRule);
    }
  }

  // Cleanup
  destroy() {
    this.stopMonitoring();
    this.rules.clear();
    this.activeTriggers.clear();
  }
}

// Export singleton instance
export const autoTriggerService = new AutoTriggerService();
export default autoTriggerService;
