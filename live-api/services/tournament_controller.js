/**
 * Tournament Controller
 *
 * Manages the evolutionary trading tournament across rounds.
 * Orchestrates strategy progression from R1 (paper micro) to Live trading.
 * Integrates with EvoTester for genetic optimization.
 */

class TournamentController {
    constructor(strategyManager, eventBus) {
        this.strategyManager = strategyManager;
        this.eventBus = eventBus;

        // Tournament configuration
        this.rounds = {
            R1: {
                name: 'Micro Capital',
                minCapital: 100,
                maxCapital: 200,
                durationDays: 7,
                minTrades: 40,
                criteria: {
                    minSharpe: 0.9,
                    minPf: 1.15,
                    maxDd: 0.08,
                    maxBreaches: 0
                }
            },
            R2: {
                name: 'Growth Capital',
                minCapital: 500,
                maxCapital: 800,
                durationDays: 10,
                minTrades: 60,
                criteria: {
                    minSharpe: 1.1,
                    minPf: 1.25,
                    maxDd: 0.07,
                    maxSlippage: 0.0005 // 5 bps over model
                }
            },
            R3: {
                name: 'Pre-Live Soak',
                minCapital: 1500,
                maxCapital: 2000,
                durationDays: 15,
                minTrades: 80,
                criteria: {
                    minSharpe: 1.2,
                    minPf: 1.3,
                    maxDd: 0.06,
                    oosGatesPass: true,
                    goNoGo: 'GREEN'
                }
            },
            LIVE: {
                name: 'Live Probation',
                minCapital: 2500,
                maxCapital: 5000,
                durationDays: null, // Ongoing
                criteria: {
                    probationPeriod: 30, // days
                    maxSharpeDrop: 0.8,
                    maxDd: 0.06,
                    maxBreachesPerDay: 3,
                    maxSlippage: 0.0008 // 8 bps over model
                }
            }
        };

        // Tournament state
        this.currentGeneration = 1;
        this.activeStrategies = new Map();
        this.tournamentStats = {
            totalPromotions: 0,
            totalDemotions: 0,
            roundPassRates: {},
            generationStats: {}
        };

        // Auto-run tournament decisions every 15 minutes during market hours
        setInterval(() => this.runTournamentCycle(), 15 * 60 * 1000);
        this.runTournamentCycle(); // Initial run
    }

    /**
     * Main tournament cycle - run every 15 minutes
     */
    async runTournamentCycle() {
        try {
            console.log('[TOURNAMENT] Running tournament cycle...');

            const allStrategies = this.strategyManager.getAllStrategies();
            const decisions = this.evaluateStrategies(allStrategies);

            if (decisions.length > 0) {
                await this.applyDecisions(decisions);
                this.emitTournamentUpdate(decisions);
            }

            console.log(`[TOURNAMENT] Cycle complete. ${decisions.length} decisions made.`);
        } catch (error) {
            console.error('[TOURNAMENT] Error in tournament cycle:', error);
        }
    }

    /**
     * Evaluate all strategies and determine tournament decisions
     */
    evaluateStrategies(strategies) {
        const decisions = [];
        const now = new Date();

        strategies.forEach(strategy => {
            const decision = this.evaluateStrategy(strategy, now);
            if (decision) {
                decisions.push(decision);
            }
        });

        return decisions;
    }

    /**
     * Evaluate a single strategy for tournament progression
     */
    evaluateStrategy(strategy, now) {
        const perf = strategy.performance || {};
        const stage = this.getStrategyStage(strategy);
        const roundConfig = this.rounds[stage];

        if (!roundConfig) return null;

        // Check minimum time/trade requirements
        const ageDays = (now - new Date(strategy.created_at)) / (1000 * 60 * 60 * 24);

        if (stage !== 'LIVE' && ageDays < roundConfig.durationDays) {
            return null; // Not enough time yet
        }

        if (perf.total_trades < roundConfig.minTrades) {
            return null; // Not enough trades yet
        }

        // Evaluate promotion criteria
        if (this.meetsPromotionCriteria(strategy, roundConfig.criteria)) {
            const nextStage = this.getNextStage(stage);

            if (nextStage) {
                return {
                    strategyId: strategy.id,
                    decision: 'promote',
                    fromStage: stage,
                    toStage: nextStage,
                    reason: `Passed ${stage} criteria`,
                    metrics: {
                        sharpe: perf.sharpe_ratio,
                        pf: perf.profit_factor,
                        dd: perf.max_drawdown,
                        trades: perf.total_trades
                    }
                };
            }
        }

        // Check for demotion (Live strategies only)
        if (stage === 'LIVE' && this.shouldDemoteLiveStrategy(strategy)) {
            return {
                strategyId: strategy.id,
                decision: 'demote',
                fromStage: stage,
                toStage: 'R1',
                reason: 'Failed live probation criteria',
                metrics: {
                    sharpe: perf.sharpe_ratio,
                    dd: perf.max_drawdown,
                    breaches: perf.gate_breaches || 0
                }
            };
        }

        return null; // Hold current position
    }

    /**
     * Check if strategy meets promotion criteria for its current round
     */
    meetsPromotionCriteria(strategy, criteria) {
        const perf = strategy.performance || {};

        // Basic metrics check
        if (perf.sharpe_ratio < criteria.minSharpe) return false;
        if (perf.profit_factor < criteria.minPf) return false;
        if (perf.max_drawdown > criteria.maxDd) return false;

        // Breach check
        if (criteria.maxBreaches !== undefined) {
            if ((perf.gate_breaches || 0) > criteria.maxBreaches) return false;
        }

        // Slippage check
        if (criteria.maxSlippage !== undefined) {
            if ((perf.avg_slippage || 0) > criteria.maxSlippage) return false;
        }

        // OOS gates check
        if (criteria.oosGatesPass) {
            if (!this.passesOOSGates(strategy)) return false;
        }

        // Go/No-Go check
        if (criteria.goNoGo) {
            if (strategy.go_no_go !== criteria.goNoGo) return false;
        }

        return true;
    }

    /**
     * Check if live strategy should be demoted
     */
    shouldDemoteLiveStrategy(strategy) {
        const perf = strategy.performance || {};
        const criteria = this.rounds.LIVE.criteria;

        // Multiple failure conditions
        const failures = [];

        if (perf.sharpe_ratio < criteria.maxSharpeDrop) failures.push('sharpe');
        if (perf.max_drawdown > criteria.maxDd) failures.push('drawdown');
        if ((perf.gate_breaches || 0) > criteria.maxBreachesPerDay) failures.push('breaches');
        if ((perf.avg_slippage || 0) > criteria.maxSlippage) failures.push('slippage');

        return failures.length >= 2; // 2+ failures = demote
    }

    /**
     * Check if strategy passes OOS gates
     */
    passesOOSGates(strategy) {
        // This would integrate with your existing backtest gates
        // For now, check if strategy has been backtested
        return strategy.oos_results && strategy.oos_results.passed_gates;
    }

    /**
     * Get strategy's current tournament stage
     */
    getStrategyStage(strategy) {
        if (strategy.status === 'live') return 'LIVE';
        if (strategy.status === 'paper') {
            // Check capital allocation to determine round
            const capital = strategy.allocated_capital || 0;
            if (capital <= 200) return 'R1';
            if (capital <= 800) return 'R2';
            if (capital <= 2000) return 'R3';
        }
        return 'R1'; // Default
    }

    /**
     * Get next stage in tournament progression
     */
    getNextStage(currentStage) {
        const progression = { R1: 'R2', R2: 'R3', R3: 'LIVE' };
        return progression[currentStage];
    }

    /**
     * Apply tournament decisions
     */
    async applyDecisions(decisions) {
        for (const decision of decisions) {
            try {
                if (decision.decision === 'promote') {
                    await this.promoteStrategy(decision);
                } else if (decision.decision === 'demote') {
                    await this.demoteStrategy(decision);
                }

                // Update tournament stats
                this.updateTournamentStats(decision);

            } catch (error) {
                console.error(`[TOURNAMENT] Error applying decision for ${decision.strategyId}:`, error);
            }
        }
    }

    /**
     * Promote strategy to next round
     */
    async promoteStrategy(decision) {
        const nextRound = this.rounds[decision.toStage];
        const newCapital = this.calculateCapitalAllocation(decision.toStage);

        // Update strategy via your existing API
        const updateResult = await this.strategyManager.manualPromotion(
            decision.strategyId,
            `Tournament: ${decision.fromStage} → ${decision.toStage} (${decision.reason})`
        );

        if (updateResult.success) {
            // Update capital allocation
            const strategy = this.strategyManager.getStrategy(decision.strategyId);
            if (strategy) {
                strategy.allocated_capital = newCapital;
                strategy.tournament_stage = decision.toStage;
                this.strategyManager.saveStrategies();
            }
        }

        console.log(`[TOURNAMENT] Promoted ${decision.strategyId}: ${decision.fromStage} → ${decision.toStage} ($${newCapital})`);
    }

    /**
     * Demote strategy back to R1
     */
    async demoteStrategy(decision) {
        const demoteResult = await this.strategyManager.manualPromotion(
            decision.strategyId,
            `Tournament: ${decision.fromStage} → ${decision.toStage} (${decision.reason})`
        );

        if (demoteResult.success) {
            // Reset to R1 capital
            const strategy = this.strategyManager.getStrategy(decision.strategyId);
            if (strategy) {
                strategy.allocated_capital = 100; // R1 minimum
                strategy.tournament_stage = 'R1';
                this.strategyManager.saveStrategies();
            }
        }

        console.log(`[TOURNAMENT] Demoted ${decision.strategyId}: ${decision.fromStage} → ${decision.toStage}`);
    }

    /**
     * Calculate capital allocation for a stage
     */
    calculateCapitalAllocation(stage) {
        const round = this.rounds[stage];
        if (!round) return 100;

        // Simple allocation based on performance ranking
        // In production, this could be more sophisticated
        return Math.floor((round.minCapital + round.maxCapital) / 2);
    }

    /**
     * Update tournament statistics
     */
    updateTournamentStats(decision) {
        if (decision.decision === 'promote') {
            this.tournamentStats.totalPromotions++;
        } else if (decision.decision === 'demote') {
            this.tournamentStats.totalDemotions++;
        }

        // Track pass rates by round
        const fromRound = decision.fromStage;
        if (!this.tournamentStats.roundPassRates[fromRound]) {
            this.tournamentStats.roundPassRates[fromRound] = { promoted: 0, demoted: 0 };
        }

        if (decision.decision === 'promote') {
            this.tournamentStats.roundPassRates[fromRound].promoted++;
        } else if (decision.decision === 'demote') {
            this.tournamentStats.roundPassRates[fromRound].demoted++;
        }
    }

    /**
     * Emit tournament update event
     */
    emitTournamentUpdate(decisions) {
        const update = {
            timestamp: new Date().toISOString(),
            generation: this.currentGeneration,
            decisions: decisions,
            stats: this.tournamentStats,
            activeRounds: this.getActiveRoundsSummary()
        };

        // Emit via event bus (could also emit SSE)
        this.eventBus.publish({
            type: 'tournament_update',
            data: update
        });

        console.log(`[TOURNAMENT] Emitted update: ${decisions.length} decisions`);
    }

    /**
     * Get summary of active strategies by round
     */
    getActiveRoundsSummary() {
        const allStrategies = this.strategyManager.getAllStrategies();
        const summary = { R1: 0, R2: 0, R3: 0, LIVE: 0 };

        allStrategies.forEach(strategy => {
            const stage = this.getStrategyStage(strategy);
            if (summary[stage] !== undefined) {
                summary[stage]++;
            }
        });

        return summary;
    }

    /**
     * Register new phenotype from EvoTester
     */
    registerPhenotype(phenotypeData) {
        // This would be called by the AI brain when EvoTester produces new strategies
        const strategyConfig = {
            name: phenotypeData.label,
            parameters: phenotypeData.params,
            constraints: phenotypeData.constraints,
            promotion_criteria: this.rounds.R1.criteria,
            tournament_stage: 'R1',
            allocated_capital: 100, // R1 minimum
            source: 'evotester',
            generation: this.currentGeneration
        };

        const strategy = this.strategyManager.registerStrategy(
            phenotypeData.id,
            strategyConfig
        );

        console.log(`[TOURNAMENT] Registered EvoTester phenotype: ${phenotypeData.id} (Gen ${this.currentGeneration})`);

        return strategy;
    }

    /**
     * Send feedback to EvoTester
     */
    async sendEvoTesterFeedback(decisions) {
        // This would POST to EvoTester's feedback endpoint
        const feedback = {
            generation: this.currentGeneration,
            results: decisions.map(d => ({
                strategy_id: d.strategyId,
                decision: d.decision,
                metrics: d.metrics,
                context: {
                    stage: d.fromStage,
                    next_stage: d.toStage,
                    reason: d.reason
                }
            }))
        };

        // In production, this would make HTTP request to EvoTester
        console.log(`[TOURNAMENT] Sending feedback to EvoTester:`, JSON.stringify(feedback, null, 2));

        // Increment generation for next round
        this.currentGeneration++;
    }

    /**
     * Get recent decisions (for AI orchestrator)
     */
    getRecentDecisions(limit = 10) {
        // In production, this would be populated from a persistent decisions log
        // For now, return empty array - the AI orchestrator handles its own decision tracking
        return [];
    }

    /**
     * Get tournament dashboard data
     */
    getDashboardData() {
        return {
            current_generation: this.currentGeneration,
            rounds: Object.entries(this.rounds).map(([key, config]) => ({
                stage: key,
                name: config.name,
                active_strategies: this.getActiveRoundsSummary()[key] || 0,
                criteria: config.criteria
            })),
            stats: this.tournamentStats,
            recent_decisions: this.getRecentDecisions(5)
        };
    }
}

module.exports = TournamentController;
