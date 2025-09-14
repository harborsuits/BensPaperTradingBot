/**
 * AI Orchestrator Service
 *
 * Autonomous AI Brain that orchestrates EvoTester seeding, tournament management,
 * and strategy lifecycle end-to-end without human intervention.
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const SeedGenerator = require('./seed_generator');
const PolicyEngine = require('./policy_engine');

class AIOrchestrator {
    constructor(strategyManager, tournamentController, marketIndicators, configPath = './ai_policy.yaml') {
        this.strategyManager = strategyManager;
        this.tournamentController = tournamentController;
        this.marketIndicators = marketIndicators;
        this.configPath = configPath;

        // Load AI policy configuration
        this.policy = this.loadPolicy();

        // Orchestrator state
        this.isActive = false;
        this.lastRun = null;
        this.decisionHistory = [];
        this.triggerStates = {};

        // Market context cache
        this.marketContext = null;
        this.contextExpiry = 5 * 60 * 1000; // 5 minutes

        // Initialize seed generator
        this.seedGenerator = new SeedGenerator(this.policy);

        // Initialize policy engine
        this.policyEngine = new PolicyEngine(this.policy);

        console.log('[AI_ORCHESTRATOR] Initialized with policy version:', this.policy.version || 'latest');
    }

    /**
     * Load AI policy configuration from YAML
     */
    loadPolicy() {
        try {
            const configPath = path.resolve(this.configPath);
            const configData = fs.readFileSync(configPath, 'utf8');
            const policy = yaml.load(configData);

            console.log('[AI_ORCHESTRATOR] Policy loaded:', Object.keys(policy));
            return policy;
        } catch (error) {
            console.error('[AI_ORCHESTRATOR] Error loading policy:', error);
            return this.getDefaultPolicy();
        }
    }

    /**
     * Default policy if YAML fails to load
     */
    getDefaultPolicy() {
        return {
            ai_policy: {
                paper_cap_max: 20000,
                exploration_quota: 0.1,
                rounds: {
                    R1: { max_slots: 50, cap_total: 5000 },
                    R2: { max_slots: 20, cap_total: 8000 },
                    R3: { max_slots: 8, cap_total: 7000 }
                },
                cadence: { decisions_minutes: 15 }
            },
            families: {
                trend: { weight: 0.5 },
                meanrev: { weight: 0.3 },
                breakout: { weight: 0.2 }
            }
        };
    }

    /**
     * Start the autonomous orchestrator
     */
    start() {
        if (this.isActive) return;

        console.log('[AI_ORCHESTRATOR] Starting autonomous operation...');
        this.isActive = true;

        // Run initial cycle
        this.runOrchestratorCycle();

        // Start the main orchestration loop (market hours only)
        const intervalMs = (this.policy.ai_policy?.cadence?.decisions_minutes || 15) * 60 * 1000;
        this.intervalId = setInterval(() => {
            if (this.isMarketHours()) {
                this.runOrchestratorCycle();
            }
        }, intervalMs);

        console.log(`[AI_ORCHESTRATOR] Orchestration loop started (${intervalMs/1000}s intervals)`);
    }

    /**
     * Stop the autonomous orchestrator
     */
    stop() {
        if (!this.isActive) return;

        console.log('[AI_ORCHESTRATOR] Stopping autonomous operation...');
        this.isActive = false;

        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    /**
     * Main orchestration cycle - runs every 15 minutes
     */
    async runOrchestratorCycle() {
        try {
            console.log('[AI_ORCHESTRATOR] Starting orchestration cycle...');
            const cycleStart = new Date();

            // Step 1: Get current market context
            const context = await this.getMarketContext();
            this.marketContext = context;

            // Step 2: Get current roster and capacity
            const allStrategies = this.strategyManager.getAllStrategies();
            const roster = this.getRosterSnapshot();
            const capacity = this.getCapacitySnapshot();

            // Step 3: Check for catastrophic conditions first
            const circuitBreakers = this.policyEngine.checkCircuitBreakers(context, allStrategies);
            if (circuitBreakers.length > 0) {
                console.log(`[AI_ORCHESTRATOR] 🚨 CIRCUIT BREAKERS ACTIVE: ${circuitBreakers.length}`);
                circuitBreakers.forEach(cb => {
                    console.log(`  ${cb.severity.toUpperCase()}: ${cb.reason} → ${cb.action}`);
                });

                // Emergency halt - don't spawn new strategies or make changes
                const needs = {
                    spawnR1: 0,
                    promoteR1ToR2: 0,
                    promoteR2ToR3: 0,
                    promoteR3ToLive: 0,
                    demoteLive: 0,
                    demoteR3: 0,
                    reseedFamilies: false,
                    triggers: ['circuit_breaker'],
                    circuitBreakers: circuitBreakers
                };

                // Record emergency state
                this.recordDecisionCycle(cycleStart, context, needs, []);
                return;
            }

            // Step 4: Evaluate normal triggers and compute needs
            const needs = await this.computeOrchestrationNeeds(context, roster, capacity);

            // Step 5: Execute orchestration decisions
            const decisions = await this.executeOrchestrationDecisions(needs, context, roster, capacity);

            // Step 6: Record decision history
            this.recordDecisionCycle(cycleStart, context, needs, decisions);

            console.log(`[AI_ORCHESTRATOR] Cycle complete. Triggered: ${Object.keys(needs).filter(k => needs[k] > 0).join(', ')}`);

        } catch (error) {
            console.error('[AI_ORCHESTRATOR] Error in orchestration cycle:', error);
        }
    }

    /**
     * Get current market context (regime, volatility, calendar events)
     */
    async getMarketContext() {
        try {
            // Check cache first
            if (this.marketContext &&
                (Date.now() - this.marketContext.timestamp) < this.contextExpiry) {
                return this.marketContext;
            }

            const context = {
                timestamp: Date.now(),
                regime: 'neutral',
                volatility: 'medium',
                sentiment: 'neutral',
                calendar_events: [],
                vix_level: null,
                market_trend: 'sideways'
            };

            // Get market regime from indicators service
            try {
                const regimeData = await this.marketIndicators.getMarketRegime('SPY');
                if (regimeData) {
                    context.regime = regimeData.regime;
                    context.volatility = regimeData.volatility;
                    context.market_trend = regimeData.trend;
                }
            } catch (error) {
                console.warn('[AI_ORCHESTRATOR] Could not get market regime:', error.message);
            }

            // Get VIX level (volatility indicator)
            try {
                const vixIndicators = await this.marketIndicators.getIndicators('VIX', 50);
                if (vixIndicators && vixIndicators.latest) {
                    context.vix_level = vixIndicators.latest.price;
                }
            } catch (error) {
                console.warn('[AI_ORCHESTRATOR] Could not get VIX data:', error.message);
            }

            // Check for upcoming events (earnings, FOMC, etc.)
            context.calendar_events = this.checkCalendarEvents();

            console.log(`[AI_ORCHESTRATOR] Market context: ${context.regime} regime, ${context.volatility} volatility`);
            return context;

        } catch (error) {
            console.error('[AI_ORCHESTRATOR] Error getting market context:', error);
            return {
                timestamp: Date.now(),
                regime: 'neutral',
                volatility: 'medium',
                sentiment: 'neutral',
                calendar_events: [],
                error: error.message
            };
        }
    }

    /**
     * Get snapshot of current strategy roster
     */
    getRosterSnapshot() {
        const allStrategies = this.strategyManager.getAllStrategies();
        const roster = {
            total: allStrategies.length,
            byStage: { R1: 0, R2: 0, R3: 0, LIVE: 0 },
            byStatus: { paper: 0, live: 0 },
            performance: {
                avgSharpe: 0,
                avgPF: 0,
                totalTrades: 0,
                underperformers: 0
            }
        };

        let totalSharpe = 0;
        let totalPF = 0;
        let validStrategies = 0;

        allStrategies.forEach(strategy => {
            // Count by stage (inferred from capital allocation)
            const stage = this.getStrategyStage(strategy);
            roster.byStage[stage]++;

            // Count by status
            roster.byStatus[strategy.status] = (roster.byStatus[strategy.status] || 0) + 1;

            // Aggregate performance
            const perf = strategy.performance || {};
            roster.performance.totalTrades += perf.total_trades || 0;

            if (perf.sharpe_ratio !== undefined) {
                totalSharpe += perf.sharpe_ratio;
                validStrategies++;
            }

            if (perf.profit_factor !== undefined) {
                totalPF += perf.profit_factor;
            }

            // Check for underperformers
            if (this.isUnderperforming(strategy)) {
                roster.performance.underperformers++;
            }
        });

        if (validStrategies > 0) {
            roster.performance.avgSharpe = totalSharpe / validStrategies;
            roster.performance.avgPF = totalPF / validStrategies;
        }

        return roster;
    }

    /**
     * Get capacity snapshot (budget, slots available)
     */
    getCapacitySnapshot() {
        const paperStrategies = this.strategyManager.getStrategiesByStatus('paper');
        const liveStrategies = this.strategyManager.getStrategiesByStatus('live');

        let paperBudgetUsed = 0;
        paperStrategies.forEach(strategy => {
            paperBudgetUsed += strategy.allocated_capital || 100;
        });

        const capacity = {
            paperBudget: {
                used: paperBudgetUsed,
                max: this.policy.ai_policy?.paper_cap_max || 20000,
                available: Math.max(0, (this.policy.ai_policy?.paper_cap_max || 20000) - paperBudgetUsed)
            },
            slots: {
                R1: {
                    used: this.getSlotsUsed('R1'),
                    max: this.policy.ai_policy?.rounds?.R1?.max_slots || 50,
                    available: Math.max(0, (this.policy.ai_policy?.rounds?.R1?.max_slots || 50) - this.getSlotsUsed('R1'))
                },
                R2: {
                    used: this.getSlotsUsed('R2'),
                    max: this.policy.ai_policy?.rounds?.R2?.max_slots || 20,
                    available: Math.max(0, (this.policy.ai_policy?.rounds?.R2?.max_slots || 20) - this.getSlotsUsed('R2'))
                },
                R3: {
                    used: this.getSlotsUsed('R3'),
                    max: this.policy.ai_policy?.rounds?.R3?.max_slots || 8,
                    available: Math.max(0, (this.policy.ai_policy?.rounds?.R3?.max_slots || 8) - this.getSlotsUsed('R3'))
                }
            }
        };

        return capacity;
    }

    /**
     * Get slots used for a specific round
     */
    getSlotsUsed(stage) {
        const allStrategies = this.strategyManager.getAllStrategies();
        return allStrategies.filter(strategy => this.getStrategyStage(strategy) === stage).length;
    }

    /**
     * Determine strategy's tournament stage
     */
    getStrategyStage(strategy) {
        if (strategy.status === 'live') return 'LIVE';
        if (strategy.status === 'paper') {
            const capital = strategy.allocated_capital || 0;
            if (capital <= 200) return 'R1';
            if (capital <= 800) return 'R2';
            if (capital <= 2000) return 'R3';
        }
        return 'R1'; // Default
    }

    /**
     * Check if strategy is underperforming
     */
    isUnderperforming(strategy) {
        const perf = strategy.performance || {};
        const thresholds = this.policy.ai_policy?.live_probation?.demote_if || {};

        return (
            perf.sharpe_ratio < (thresholds.sharpe_10d_max || 0.8) ||
            perf.max_drawdown > (thresholds.dd_max || 0.06) ||
            (perf.gate_breaches || 0) > (thresholds.breaches_24h_max || 3)
        );
    }

    /**
     * Compute orchestration needs based on triggers
     */
    async computeOrchestrationNeeds(context, roster, capacity) {
        const needs = {
            spawnR1: 0,
            promoteR1ToR2: 0,
            promoteR2ToR3: 0,
            promoteR3ToLive: 0,
            demoteLive: 0,
            demoteR3: 0,
            reseedFamilies: false
        };

        // Use policy engine to evaluate all triggers
        const evaluatedNeeds = this.policyEngine.evaluateTriggers(context, roster, capacity);

        // Merge evaluated needs with our base needs object
        Object.assign(needs, evaluatedNeeds);

        // Log trigger activity
        if (needs.triggers && needs.triggers.length > 0) {
            console.log(`[AI_ORCHESTRATOR] Active triggers: ${needs.triggers.join(', ')}`);
        }

        return needs;
    }

    /**
     * Execute orchestration decisions
     */
    async executeOrchestrationDecisions(needs, context, roster, capacity) {
        const decisions = [];

        // 1. Spawn new R1 strategies if needed
        if (needs.spawnR1 > 0) {
            const spawnDecision = await this.spawnR1Strategies(needs.spawnR1, context);
            decisions.push(...spawnDecision);
        }

        // 2. Handle promotions and demotions
        const tournamentDecisions = await this.executeTournamentDecisions(roster);
        decisions.push(...tournamentDecisions);

        // 3. Reseed families if regime changed
        if (needs.reseedFamilies) {
            const reseedDecision = await this.reseedFamiliesForRegime(context);
            decisions.push(reseedDecision);
        }

        return decisions;
    }

    /**
     * Spawn new R1 strategies
     */
    async spawnR1Strategies(count, context) {
        const decisions = [];
        const families = this.selectFamiliesForContext(context);

        console.log(`[AI_ORCHESTRATOR] Spawning ${count} R1 strategies with families:`, Object.keys(families));

        // Generate seeds for EvoTester
        const seedRequest = {
            families: Object.entries(families).map(([name, weight]) => ({
                name,
                weight
            })),
            count: count,
            bounds: this.getGeneBoundsForRegime(context.regime),
            objective: "after_cost_sharpe_pf_dd"
        };

        // Use seed generator to create phenotypes
        const phenotypes = this.seedGenerator.generatePhenotypes(seedRequest);

        // Register phenotypes with tournament
        phenotypes.forEach(phenotype => {
            const strategy = this.tournamentController.registerPhenotype({
                id: phenotype.id,
                label: phenotype.label,
                params: phenotype.params,
                constraints: phenotype.constraints
            });

            decisions.push({
                type: 'spawn',
                strategyId: phenotype.id,
                stage: 'R1',
                family: phenotype.family,
                reason: `AI: ${context.regime} regime response`,
                timestamp: new Date().toISOString()
            });
        });

        return decisions;
    }

    /**
     * Execute tournament decisions (promotions/demotions)
     */
    async executeTournamentDecisions(roster) {
        // The tournament controller already handles this automatically
        // We just need to trigger it and collect the results
        await this.tournamentController.runTournamentCycle();

        const decisions = [];

        // Get recent decisions from tournament controller
        // This would need to be implemented in the tournament controller
        const recentDecisions = this.tournamentController.getRecentDecisions?.() || [];

        recentDecisions.forEach(decision => {
            decisions.push({
                type: decision.decision,
                strategyId: decision.strategyId,
                fromStage: decision.fromStage,
                toStage: decision.toStage,
                reason: decision.reason,
                timestamp: new Date().toISOString()
            });
        });

        return decisions;
    }

    /**
     * Reseed families for new regime
     */
    async reseedFamiliesForRegime(context) {
        console.log(`[AI_ORCHESTRATOR] Reseeding families for ${context.regime} regime`);

        return {
            type: 'reseed',
            regime: context.regime,
            families: this.selectFamiliesForContext(context),
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Select families based on market context
     */
    selectFamiliesForContext(context) {
        const roster = this.getRosterSnapshot();
        return this.policyEngine.selectFamiliesForContext(context, roster);
    }

    /**
     * Get gene bounds for current regime
     */
    getGeneBoundsForRegime(regime) {
        return this.policyEngine.getGeneBoundsForRegime(regime);
    }

    /**
     * Detect regime change
     */
    detectRegimeChange(context) {
        const currentRegime = context.regime;
        const previousRegime = this.triggerStates.lastRegime;

        if (currentRegime !== previousRegime) {
            this.triggerStates.lastRegime = currentRegime;
            return true;
        }

        return false;
    }

    /**
     * Check for upcoming calendar events
     */
    checkCalendarEvents() {
        const events = [];
        const now = new Date();

        // This would integrate with a calendar service
        // For now, return empty array
        return events;
    }

    /**
     * Simulate EvoTester seeding (placeholder)
     */
    simulateEvoSeeding(request) {
        const phenotypes = [];
        const families = request.families;

        for (let i = 0; i < request.count; i++) {
            // Select family based on weights
            const family = this.selectWeightedFamily(families);

            phenotypes.push({
                id: `evo_${Date.now()}_${i}`,
                label: `${family}_phenotype_${i}`,
                family: family,
                params: this.generateRandomParams(family),
                constraints: {}
            });
        }

        return phenotypes;
    }

    /**
     * Select family based on weights
     */
    selectWeightedFamily(families) {
        const totalWeight = families.reduce((sum, f) => sum + f.weight, 0);
        let random = Math.random() * totalWeight;

        for (const family of families) {
            random -= family.weight;
            if (random <= 0) {
                return family.name;
            }
        }

        return families[0].name; // fallback
    }

    /**
     * Generate random parameters for a family
     */
    generateRandomParams(family) {
        const bounds = this.policy.families?.[family]?.genes || {};
        const params = {};

        Object.entries(bounds).forEach(([param, range]) => {
            if (Array.isArray(range) && range.length === 2) {
                const [min, max] = range;
                if (typeof min === 'number' && typeof max === 'number') {
                    params[param] = min + Math.random() * (max - min);
                } else {
                    params[param] = range[Math.floor(Math.random() * range.length)];
                }
            }
        });

        return params;
    }

    /**
     * Record decision cycle for monitoring
     */
    recordDecisionCycle(cycleStart, context, needs, decisions) {
        const cycle = {
            timestamp: cycleStart.toISOString(),
            duration: Date.now() - cycleStart.getTime(),
            context: context,
            needs: needs,
            decisions: decisions,
            roster_summary: this.getRosterSnapshot(),
            capacity_summary: this.getCapacitySnapshot()
        };

        this.decisionHistory.push(cycle);

        // Keep only last 100 cycles
        if (this.decisionHistory.length > 100) {
            this.decisionHistory.shift();
        }

        this.lastRun = new Date();
    }

    /**
     * Get orchestrator status for monitoring
     */
    getStatus() {
        return {
            isActive: this.isActive,
            lastRun: this.lastRun?.toISOString(),
            totalCycles: this.decisionHistory.length,
            currentContext: this.marketContext,
            recentDecisions: this.decisionHistory.slice(-5),
            policyVersion: this.policy.version || 'latest',
            triggers: this.triggerStates
        };
    }

    /**
     * Check if current time is within market hours (9:30 AM - 4:00 PM ET)
     */
    isMarketHours() {
        const now = new Date();
        const etOffset = now.getTimezoneOffset() === 300 ? 0 : 300 - now.getTimezoneOffset(); // Handle EDT/EST
        const etTime = new Date(now.getTime() + (etOffset * 60000));

        const hour = etTime.getHours();
        const minute = etTime.getMinutes();
        const currentMinutes = hour * 60 + minute;

        const marketOpen = 9 * 60 + 30;  // 9:30 AM
        const marketClose = 16 * 60;     // 4:00 PM

        // Check if it's a weekday (Monday = 1, Friday = 5)
        const dayOfWeek = etTime.getDay();
        const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;

        return isWeekday && currentMinutes >= marketOpen && currentMinutes <= marketClose;
    }

    /**
     * Manual trigger for testing
     */
    async triggerManualCycle() {
        console.log('[AI_ORCHESTRATOR] Manual cycle triggered');
        await this.runOrchestratorCycle();
    }
}

module.exports = AIOrchestrator;
