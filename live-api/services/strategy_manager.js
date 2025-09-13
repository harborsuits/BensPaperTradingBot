const fs = require('fs');
const path = require('path');

/**
 * Strategy Manager
 *
 * Manages strategy lifecycle from paper trading to live deployment.
 * Handles performance tracking, promotion criteria, and broker switching.
 */
class StrategyManager {
    constructor(dataDir = './data') {
        this.dataDir = dataDir;
        this.strategiesFile = path.join(dataDir, 'strategies.json');
        this.promotionFile = path.join(dataDir, 'promotions.json');

        // Ensure data directory exists
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
        }

        this.strategies = this.loadStrategies();
        this.promotions = this.loadPromotions();
    }

    /**
     * Load strategies from disk
     */
    loadStrategies() {
        try {
            if (fs.existsSync(this.strategiesFile)) {
                const data = fs.readFileSync(this.strategiesFile, 'utf8');
                return JSON.parse(data);
            }
        } catch (error) {
            console.error('Error loading strategies:', error);
        }
        return {};
    }

    /**
     * Load promotion history from disk
     */
    loadPromotions() {
        try {
            if (fs.existsSync(this.promotionFile)) {
                const data = fs.readFileSync(this.promotionFile, 'utf8');
                return JSON.parse(data);
            }
        } catch (error) {
            console.error('Error loading promotions:', error);
        }
        return [];
    }

    /**
     * Save strategies to disk
     */
    saveStrategies() {
        try {
            fs.writeFileSync(this.strategiesFile, JSON.stringify(this.strategies, null, 2));
        } catch (error) {
            console.error('Error saving strategies:', error);
        }
    }

    /**
     * Save promotion history to disk
     */
    savePromotions() {
        try {
            fs.writeFileSync(this.promotionFile, JSON.stringify(this.promotions, null, 2));
        } catch (error) {
            console.error('Error saving promotions:', error);
        }
    }

    /**
     * Register a new strategy
     */
    registerStrategy(strategyId, config) {
        const strategy = {
            id: strategyId,
            name: config.name || strategyId,
            status: 'paper', // Start in paper trading
            broker: 'paper',
            created_at: new Date().toISOString(),
            config: config,
            performance: {
                total_trades: 0,
                win_rate: 0,
                sharpe_ratio: 0,
                max_drawdown: 0,
                total_return: 0,
                avg_trade_pnl: 0
            },
            promotion_criteria: config.promotion_criteria || this.getDefaultPromotionCriteria(),
            last_updated: new Date().toISOString()
        };

        this.strategies[strategyId] = strategy;
        this.saveStrategies();

        console.log(`[STRATEGY] Registered new strategy: ${strategyId} (${strategy.status})`);
        return strategy;
    }

    /**
     * Update strategy performance metrics
     */
    updatePerformance(strategyId, metrics) {
        if (!this.strategies[strategyId]) {
            console.warn(`Strategy ${strategyId} not found`);
            return false;
        }

        const strategy = this.strategies[strategyId];
        strategy.performance = { ...strategy.performance, ...metrics };
        strategy.last_updated = new Date().toISOString();

        // Check if strategy meets promotion criteria
        if (strategy.status === 'paper' && this.checkPromotionCriteria(strategy)) {
            this.promoteStrategy(strategyId);
        }

        this.saveStrategies();
        return true;
    }

    /**
     * Get default promotion criteria
     */
    getDefaultPromotionCriteria() {
        return {
            min_trades: 50,
            min_win_rate: 0.55,
            min_sharpe_ratio: 1.0,
            max_drawdown: 0.08,
            min_total_return: 0.05,
            consecutive_wins: 5
        };
    }

    /**
     * Check if strategy meets promotion criteria
     */
    checkPromotionCriteria(strategy) {
        const perf = strategy.performance;
        const criteria = strategy.promotion_criteria;

        return (
            perf.total_trades >= criteria.min_trades &&
            perf.win_rate >= criteria.min_win_rate &&
            perf.sharpe_ratio >= criteria.min_sharpe_ratio &&
            perf.max_drawdown <= criteria.max_drawdown &&
            perf.total_return >= criteria.min_total_return
        );
    }

    /**
     * Promote strategy from paper to live
     */
    promoteStrategy(strategyId) {
        if (!this.strategies[strategyId]) {
            return { success: false, error: 'Strategy not found' };
        }

        const strategy = this.strategies[strategyId];

        if (strategy.status !== 'paper') {
            return { success: false, error: 'Strategy not in paper trading' };
        }

        // Update strategy status
        strategy.status = 'live';
        strategy.broker = 'tradier'; // Switch to live broker
        strategy.promoted_at = new Date().toISOString();

        // Record promotion
        const promotion = {
            strategy_id: strategyId,
            promoted_at: strategy.promoted_at,
            paper_performance: { ...strategy.performance },
            promotion_criteria: { ...strategy.promotion_criteria },
            reason: 'auto_promotion'
        };

        this.promotions.push(promotion);
        this.saveStrategies();
        this.savePromotions();

        console.log(`[STRATEGY] Promoted ${strategyId} from paper to live trading`);
        console.log(`[STRATEGY] Performance: ${strategy.performance.win_rate * 100}% win rate, Sharpe: ${strategy.performance.sharpe_ratio}`);

        return {
            success: true,
            strategy: strategy,
            promotion: promotion
        };
    }

    /**
     * Manually promote strategy (admin override)
     */
    manualPromotion(strategyId, reason = 'manual_override') {
        const result = this.promoteStrategy(strategyId);
        if (result.success) {
            result.promotion.reason = reason;
            this.savePromotions();
        }
        return result;
    }

    /**
     * Demote strategy back to paper (if live performance is poor)
     */
    demoteStrategy(strategyId, reason = 'performance_decline') {
        if (!this.strategies[strategyId]) {
            return { success: false, error: 'Strategy not found' };
        }

        const strategy = this.strategies[strategyId];

        if (strategy.status !== 'live') {
            return { success: false, error: 'Strategy not in live trading' };
        }

        // Update strategy status
        strategy.status = 'paper';
        strategy.broker = 'paper';
        strategy.demoted_at = new Date().toISOString();
        strategy.demotion_reason = reason;

        // Record demotion
        const demotion = {
            strategy_id: strategyId,
            demoted_at: strategy.demoted_at,
            live_performance: { ...strategy.performance },
            reason: reason
        };

        // Add to promotions array as demotion record
        this.promotions.push(demotion);
        this.saveStrategies();
        this.savePromotions();

        console.log(`[STRATEGY] Demoted ${strategyId} from live to paper trading (${reason})`);

        return {
            success: true,
            strategy: strategy,
            demotion: demotion
        };
    }

    /**
     * Get strategy details
     */
    getStrategy(strategyId) {
        return this.strategies[strategyId] || null;
    }

    /**
     * Get all strategies
     */
    getAllStrategies() {
        return Object.values(this.strategies);
    }

    /**
     * Get strategies by status
     */
    getStrategiesByStatus(status) {
        return Object.values(this.strategies).filter(s => s.status === status);
    }

    /**
     * Get promotion history
     */
    getPromotionHistory(strategyId = null) {
        if (strategyId) {
            return this.promotions.filter(p => p.strategy_id === strategyId);
        }
        return this.promotions;
    }

    /**
     * Get promotion candidates (strategies ready for promotion)
     */
    getPromotionCandidates() {
        return Object.values(this.strategies).filter(strategy =>
            strategy.status === 'paper' && this.checkPromotionCriteria(strategy)
        );
    }

    /**
     * Get dashboard data for UI
     */
    getDashboardData() {
        const paperStrategies = this.getStrategiesByStatus('paper');
        const liveStrategies = this.getStrategiesByStatus('live');
        const promotionCandidates = this.getPromotionCandidates();

        return {
            summary: {
                total_strategies: Object.keys(this.strategies).length,
                paper_strategies: paperStrategies.length,
                live_strategies: liveStrategies.length,
                promotion_candidates: promotionCandidates.length
            },
            strategies: this.getAllStrategies(),
            promotion_candidates: promotionCandidates,
            recent_promotions: this.promotions.slice(-10).reverse()
        };
    }

    /**
     * Auto-manage strategies (called by AI brain)
     */
    autoManageStrategies() {
        const candidates = this.getPromotionCandidates();

        candidates.forEach(candidate => {
            console.log(`[AI] Auto-promoting ${candidate.id} from paper to live`);
            this.manualPromotion(candidate.id, 'ai_auto_promotion');
        });

        return {
            promoted_count: candidates.length,
            candidates: candidates
        };
    }

    /**
     * AI-driven strategy evaluation and management
     */
    evaluateAndManageStrategies(currentMetrics = {}) {
        // This would be called by the AI brain system
        // It handles all the automated strategy management

        const results = {
            registered: 0,
            updated: 0,
            promoted: 0,
            demoted: 0
        };

        // Auto-register new strategies detected by AI
        if (currentMetrics.newStrategies) {
            currentMetrics.newStrategies.forEach(strategy => {
                this.registerStrategy(strategy.id, strategy.config);
                results.registered++;
            });
        }

        // Update performance for existing strategies
        if (currentMetrics.performanceUpdates) {
            Object.entries(currentMetrics.performanceUpdates).forEach(([strategyId, metrics]) => {
                this.updatePerformance(strategyId, metrics);
                results.updated++;
            });
        }

        // Auto-manage promotions
        const promotionResults = this.autoManageStrategies();
        results.promoted = promotionResults.promoted_count;

        // Check for demotion candidates (poor live performance)
        const liveStrategies = this.getStrategiesByStatus('live');
        liveStrategies.forEach(strategy => {
            if (this.shouldDemoteStrategy(strategy)) {
                this.demoteStrategy(strategy.id, 'ai_performance_decline');
                results.demoted++;
            }
        });

        return results;
    }

    /**
     * Determine if a live strategy should be demoted
     */
    shouldDemoteStrategy(strategy) {
        const perf = strategy.performance;

        // Demote if recent performance is poor
        return (
            perf.sharpe_ratio < 0.5 ||
            perf.max_drawdown > 0.15 ||
            (perf.total_trades > 100 && perf.win_rate < 0.45)
        );
    }
}

module.exports = StrategyManager;
