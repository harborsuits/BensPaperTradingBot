/**
 * Policy Engine Service
 *
 * Implements autonomous trigger logic for AI decision making.
 * Evaluates market conditions, strategy performance, and capacity constraints
 * to determine when to spawn, promote, demote, or reseed strategies.
 */

class PolicyEngine {
    constructor(policy) {
        this.policy = policy;
        this.triggerStates = {
            lastRegime: null,
            lastRegimeChange: null,
            consecutiveUnderperformers: 0,
            capacityTriggerCount: 0,
            eventTriggerCount: 0
        };
    }

    /**
     * Check for catastrophic risk conditions (circuit breakers)
     */
    checkCircuitBreakers(context, roster) {
        const breakers = [];

        // VIX spike breaker (>35 = extreme volatility)
        if (context.vix_level && context.vix_level > 35) {
            breakers.push({
                type: 'vix_spike',
                severity: 'high',
                reason: `VIX at ${context.vix_level} (>35 threshold)`,
                action: 'halt_trading'
            });
        }

        // Extreme drawdown breaker (>5% daily loss)
        const totalPnL = roster.reduce((sum, s) => sum + (s.performance?.total_return || 0), 0);
        if (totalPnL < -0.05) { // 5% daily loss
            breakers.push({
                type: 'extreme_drawdown',
                severity: 'critical',
                reason: `Portfolio down ${(totalPnL * 100).toFixed(1)}%`,
                action: 'emergency_stop'
            });
        }

        // High failure rate breaker (>50% strategies failing)
        const failingStrategies = roster.filter(s =>
            s.performance && (s.performance.sharpe_ratio < 0.5 || s.performance.max_drawdown > 0.10)
        ).length;
        const failureRate = failingStrategies / roster.length;

        if (failureRate > 0.5) {
            breakers.push({
                type: 'mass_failure',
                severity: 'high',
                reason: `${(failureRate * 100).toFixed(0)}% strategies failing`,
                action: 'reduce_exposure'
            });
        }

        return breakers;
    }

    /**
     * Evaluate all triggers and compute orchestration needs
     */
    evaluateTriggers(context, roster, capacity) {
        const needs = {
            spawnR1: 0,
            promoteR1ToR2: 0,
            promoteR2ToR3: 0,
            promoteR3ToLive: 0,
            demoteLive: 0,
            demoteR3: 0,
            reseedFamilies: false,
            triggers: []
        };

        // 1. Capacity Trigger - Free budget or low roster utilization
        const capacityNeeds = this.evaluateCapacityTrigger(roster, capacity);
        if (capacityNeeds.spawnR1 > 0) {
            needs.spawnR1 = capacityNeeds.spawnR1;
            needs.triggers.push('capacity');
        }

        // 2. Decay Trigger - Underperforming strategies
        const decayNeeds = this.evaluateDecayTrigger(roster);
        if (decayNeeds.demoteLive > 0) {
            needs.demoteLive = decayNeeds.demoteLive;
            needs.triggers.push('decay');
        }

        // 3. Regime Change Trigger - Market regime shift
        const regimeNeeds = this.evaluateRegimeTrigger(context);
        if (regimeNeeds.reseedFamilies) {
            needs.reseedFamilies = true;
            needs.triggers.push('regime_change');
        }

        // 4. Event Trigger - Earnings, FOMC, CPI events
        const eventNeeds = this.evaluateEventTrigger(context);
        if (eventNeeds.spawnR1 > 0) {
            needs.spawnR1 = Math.max(needs.spawnR1, eventNeeds.spawnR1);
            needs.triggers.push('event_driven');
        }

        // 5. Drift Detection Trigger - Feature or alpha drift
        const driftNeeds = this.evaluateDriftTrigger(roster);
        if (driftNeeds.reseedFamilies) {
            needs.reseedFamilies = true;
            needs.triggers.push('drift_detected');
        }

        // 6. Novelty Trigger - Periodic exploration
        const noveltyNeeds = this.evaluateNoveltyTrigger(roster);
        if (noveltyNeeds.spawnR1 > 0) {
            needs.spawnR1 += noveltyNeeds.spawnR1;
            needs.triggers.push('exploration_quota');
        }

        console.log(`[POLICY_ENGINE] Evaluated triggers: ${needs.triggers.join(', ')}`);
        return needs;
    }

    /**
     * Evaluate capacity trigger
     */
    evaluateCapacityTrigger(roster, capacity) {
        const triggers = this.policy.triggers?.capacity_management || {};
        const budgetThreshold = triggers.paper_budget_threshold || 0.3;
        const rosterThreshold = triggers.roster_gap_threshold || 0.2;

        const budgetUtilization = capacity.paperBudget.used / capacity.paperBudget.max;
        const totalSlotsUsed = Object.values(capacity.slots).reduce((sum, s) => sum + s.used, 0);
        const totalSlotsMax = Object.values(capacity.slots).reduce((sum, s) => sum + s.max, 0);
        const rosterUtilization = totalSlotsUsed / totalSlotsMax;

        let spawnR1 = 0;

        // Check budget threshold
        if (budgetUtilization < (1 - budgetThreshold)) {
            const availableBudgetSlots = Math.floor(capacity.paperBudget.available / 100); // Assume $100 per strategy
            spawnR1 = Math.max(spawnR1, availableBudgetSlots);
        }

        // Check roster utilization threshold
        if (rosterUtilization < (1 - rosterThreshold)) {
            const availableSlots = capacity.slots.R1.available;
            spawnR1 = Math.max(spawnR1, Math.min(availableSlots, 10)); // Cap at 10 per cycle
        }

        return { spawnR1: Math.min(spawnR1, capacity.slots.R1.available) };
    }

    /**
     * Evaluate decay trigger
     */
    evaluateDecayTrigger(roster) {
        const triggers = this.policy.triggers?.decay_detection || {};
        const sharpeThreshold = triggers.sharpe_decay_threshold || -0.25;
        const edgeThreshold = triggers.edge_decay_threshold || -0.25;
        const lookbackDays = triggers.lookback_days || 10;

        let demoteLive = 0;

        // Count strategies that have decayed significantly
        roster.forEach(strategy => {
            if (strategy.status === 'live') {
                const perf = strategy.performance || {};
                const currentSharpe = perf.sharpe_ratio || 0;

                // Check if strategy has decayed below threshold
                if (currentSharpe < sharpeThreshold) {
                    demoteLive++;
                }
            }
        });

        return { demoteLive };
    }

    /**
     * Evaluate regime change trigger
     */
    evaluateRegimeTrigger(context) {
        const triggers = this.policy.triggers?.regime_change || {};
        const persistenceReq = triggers.trend_change_persistence || 5;

        const currentRegime = context.regime;
        const previousRegime = this.triggerStates.lastRegime;

        let reseedFamilies = false;

        if (currentRegime !== previousRegime) {
            const now = Date.now();
            const timeSinceLastChange = previousRegime ?
                now - (this.triggerStates.lastRegimeChange || 0) : 0;

            // Check if regime change has persisted
            if (timeSinceLastChange > (persistenceReq * 60 * 1000)) {
                reseedFamilies = true;
                this.triggerStates.lastRegime = currentRegime;
                this.triggerStates.lastRegimeChange = now;

                console.log(`[POLICY_ENGINE] Regime changed: ${previousRegime} → ${currentRegime}`);
            }
        }

        return { reseedFamilies };
    }

    /**
     * Evaluate event-driven trigger
     */
    evaluateEventTrigger(context) {
        const triggers = this.policy.triggers?.event_drivers || {};
        const earningsWindow = triggers.earnings_window_days || 7;
        const fomcWindow = triggers.fomc_window_days || 3;
        const cpiWindow = triggers.cpi_window_days || 2;

        const events = context.calendar_events || [];
        let spawnR1 = 0;

        // Check for upcoming high-impact events
        const highImpactEvents = events.filter(event => {
            const daysUntil = this.daysUntilEvent(event.date);
            return daysUntil <= Math.max(earningsWindow, fomcWindow, cpiWindow) &&
                   daysUntil >= 0;
        });

        if (highImpactEvents.length > 0) {
            // Spawn additional strategies for event response
            spawnR1 = Math.min(15, highImpactEvents.length * 3);
            console.log(`[POLICY_ENGINE] Event trigger: ${highImpactEvents.length} events detected`);
        }

        return { spawnR1 };
    }

    /**
     * Evaluate drift detection trigger
     */
    evaluateDriftTrigger(roster) {
        const triggers = this.policy.triggers?.drift_detection || {};
        const ksThreshold = triggers.feature_drift_ks_threshold || 0.1;
        const alphaThreshold = triggers.alpha_drift_threshold || 0.15;

        let reseedFamilies = false;
        let driftDetected = false;

        // Simplified drift detection - check for performance degradation patterns
        const liveStrategies = roster.filter(s => s.status === 'live');

        if (liveStrategies.length > 5) {
            const avgSharpe = liveStrategies.reduce((sum, s) =>
                sum + (s.performance?.sharpe_ratio || 0), 0) / liveStrategies.length;

            const underperformers = liveStrategies.filter(s =>
                (s.performance?.sharpe_ratio || 0) < alphaThreshold).length;

            const underperformerRatio = underperformers / liveStrategies.length;

            if (underperformerRatio > 0.4) { // 40% of strategies underperforming
                driftDetected = true;
                reseedFamilies = true;
                console.log(`[POLICY_ENGINE] Drift detected: ${underperformerRatio * 100}% strategies underperforming`);
            }
        }

        return { reseedFamilies };
    }

    /**
     * Evaluate novelty/exploration trigger
     */
    evaluateNoveltyTrigger(roster) {
        const explorationQuota = this.policy.ai_policy?.exploration_quota || 0.1;
        const totalStrategies = roster.length;
        const explorationStrategies = roster.filter(s =>
            s.config?.source === 'exploration' || s.config?.is_novelty).length;

        const currentExplorationRatio = explorationStrategies / totalStrategies;
        let spawnR1 = 0;

        if (currentExplorationRatio < explorationQuota) {
            const targetExploration = Math.floor(totalStrategies * explorationQuota);
            const deficit = targetExploration - explorationStrategies;
            spawnR1 = Math.min(deficit, 5); // Cap exploration spawns

            if (spawnR1 > 0) {
                console.log(`[POLICY_ENGINE] Exploration trigger: ${currentExplorationRatio * 100}% < ${explorationQuota * 100}% target`);
            }
        }

        return { spawnR1 };
    }

    /**
     * Select appropriate families for current context
     */
    selectFamiliesForContext(context, roster) {
        const regime = context.regime?.toLowerCase() || 'neutral';
        const volatility = context.volatility?.toLowerCase() || 'medium';

        const regimeConfig = this.policy.regimes?.[regime] ||
                           this.policy.regimes?.[`${regime}_${volatility}`] ||
                           this.policy.regimes?.neutral_medium ||
                           { families: ['trend', 'meanrev', 'breakout'] };

        const families = {};

        // Get current family distribution in roster
        const currentDistribution = this.getCurrentFamilyDistribution(roster);

        regimeConfig.families.forEach(family => {
            const baseWeight = this.policy.families?.[family]?.weight || 0.33;
            const currentWeight = currentDistribution[family] || 0;

            // Boost underrepresented families
            const adjustment = Math.max(0, 0.3 - currentWeight);
            families[family] = Math.min(1.0, baseWeight + adjustment);
        });

        return families;
    }

    /**
     * Get current family distribution in roster
     */
    getCurrentFamilyDistribution(roster) {
        const distribution = {};
        const total = roster.length;

        if (total === 0) return distribution;

        roster.forEach(strategy => {
            const family = strategy.config?.family || 'unknown';
            distribution[family] = (distribution[family] || 0) + 1;
        });

        // Convert to ratios
        Object.keys(distribution).forEach(family => {
            distribution[family] = distribution[family] / total;
        });

        return distribution;
    }

    /**
     * Get gene bounds for current regime
     */
    getGeneBoundsForRegime(regime) {
        const bounds = {};

        Object.keys(this.policy.families || {}).forEach(family => {
            const familyConfig = this.policy.families[family];

            if (familyConfig.genes) {
                // Apply regime-specific adaptations if configured
                const regimeConfig = this.policy.regimes?.[regime];
                if (regimeConfig?.adapt_gene_bounds) {
                    bounds[family] = this.adaptGeneBoundsForRegime(familyConfig.genes, regime);
                } else {
                    bounds[family] = familyConfig.genes;
                }
            }
        });

        return bounds;
    }

    /**
     * Adapt gene bounds based on regime
     */
    adaptGeneBoundsForRegime(baseBounds, regime) {
        const adapted = { ...baseBounds };

        // Example adaptations based on regime
        switch (regime?.toLowerCase()) {
            case 'trending':
                // Increase trend-following parameters
                if (adapted.adx_min) adapted.adx_min = [25, 35]; // Higher ADX for trending
                if (adapted.sl_atr) adapted.sl_atr = [1.2, 2.2]; // Tighter stops
                break;

            case 'choppy':
                // Favor mean reversion parameters
                if (adapted.rsi_buy) adapted.rsi_buy = [25, 40]; // More aggressive RSI levels
                if (adapted.bb_dev) adapted.bb_dev = [2.0, 2.8]; // Wider bands
                break;

            case 'volatile':
                // Increase risk management parameters
                if (adapted.sl_atr) adapted.sl_atr = [1.5, 2.5]; // Wider stops
                if (adapted.sl_pct) adapted.sl_pct = [1.0, 2.5]; // Higher stop percentages
                break;
        }

        return adapted;
    }

    /**
     * Calculate days until event
     */
    daysUntilEvent(eventDate) {
        try {
            const event = new Date(eventDate);
            const now = new Date();
            const diffTime = event.getTime() - now.getTime();
            return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        } catch (error) {
            return 999; // Far future if date parsing fails
        }
    }

    /**
     * Check if strategy meets promotion criteria
     */
    meetsPromotionCriteria(strategy, targetStage) {
        const perf = strategy.performance || {};
        const criteria = this.getPromotionCriteria(targetStage);

        // Basic performance checks
        if (perf.sharpe_ratio < criteria.minSharpe) return false;
        if (perf.profit_factor < criteria.minPf) return false;
        if (perf.max_drawdown > criteria.maxDd) return false;

        // Trade count requirement
        if ((perf.total_trades || 0) < criteria.minTrades) return false;

        // Breach limits
        if ((perf.gate_breaches || 0) > criteria.maxBreaches) return false;

        return true;
    }

    /**
     * Get promotion criteria for a stage
     */
    getPromotionCriteria(stage) {
        const stageConfig = this.policy.ai_policy?.rounds?.[stage];
        if (!stageConfig) return {};

        return {
            minTrades: stageConfig.criteria?.trades_min || 40,
            minSharpe: stageConfig.criteria?.sharpe_min || 0.9,
            minPf: stageConfig.criteria?.pf_min || 1.15,
            maxDd: stageConfig.criteria?.dd_max || 0.08,
            maxBreaches: stageConfig.criteria?.breaches_max || 0,
            maxSlippage: stageConfig.criteria?.slip_bps_over_model_max || 5
        };
    }

    /**
     * Update trigger states
     */
    updateTriggerStates(newStates) {
        Object.assign(this.triggerStates, newStates);
    }

    /**
     * Get trigger statistics for monitoring
     */
    getTriggerStats() {
        return {
            ...this.triggerStates,
            timestamp: new Date().toISOString()
        };
    }
}

module.exports = PolicyEngine;
