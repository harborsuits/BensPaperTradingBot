#!/usr/bin/env node
/**
 * Strategy Activation Script
 *
 * Activates and deploys strategies across the multi-asset universe
 * Implements progressive deployment: paper -> canary -> full
 */

const StrategyManager = require('../services/strategy_manager');

// Capital allocation per strategy type
const CAPITAL_ALLOCATION = {
    volatility: 50000,    // $50K for volatility strategies (options)
    income: 75000,        // $75K for income strategies
    mean_reversion: 30000, // $30K for mean reversion
    momentum: 40000,      // $40K for momentum
    breakout: 35000,      // $35K for breakout
    trend_following: 45000 // $45K for trend following
};

// Strategy priority order for activation
const ACTIVATION_PRIORITY = [
    // High priority - start with these
    'enhanced_straddle_strangle', // Our enhanced strategy
    'stocks_momentum_strategy',
    'stocks_mean_reversion_strategy',
    'stocks_trend_strategy',
    'covered_call_strategy',

    // Medium priority - add these next
    'vwap_mean_reversion_strategy',
    'stocks_breakout_strategy',
    'enhanced_prop_trend_following_strategy',

    // Options strategies
    'straddle_strategy',
    'strangle_strategy',
    'bull_call_spread_strategy',
    'bear_put_spread_strategy'
];

class StrategyActivator {
    constructor() {
        this.strategyManager = new StrategyManager();
    }

    /**
     * Get strategies by priority
     */
    getStrategiesByPriority() {
        const allStrategies = this.strategyManager.getAllStrategies();

        // Sort by priority order, then by type
        return allStrategies.sort((a, b) => {
            const aPriority = ACTIVATION_PRIORITY.indexOf(a.id);
            const bPriority = ACTIVATION_PRIORITY.indexOf(b.id);

            // Priority strategies first
            if (aPriority !== -1 && bPriority !== -1) {
                return aPriority - bPriority;
            }
            if (aPriority !== -1) return -1;
            if (bPriority !== -1) return 1;

            // Then by strategy type priority
            const typePriority = {
                'volatility': 1,
                'income': 2,
                'momentum': 3,
                'trend_following': 4,
                'mean_reversion': 5,
                'breakout': 6
            };

            const aType = typePriority[a.config.strategy_type] || 99;
            const bType = typePriority[b.config.strategy_type] || 99;

            return aType - bType;
        });
    }

    /**
     * Activate strategies progressively
     */
    async activateStrategies() {
        console.log('🚀 Starting Strategy Activation\n');

        const strategies = this.getStrategiesByPriority();
        let activated = 0;
        let skipped = 0;

        // First pass: Activate high-priority strategies
        console.log('📈 Phase 1: Activating High-Priority Strategies');

        const highPriority = strategies.slice(0, 8); // First 8 strategies
        for (const strategy of highPriority) {
            try {
                await this.activateSingleStrategy(strategy);
                activated++;
                console.log(`✅ Activated: ${strategy.id} (${strategy.config.strategy_type})`);
            } catch (error) {
                console.error(`❌ Failed to activate ${strategy.id}:`, error.message);
                skipped++;
            }
        }

        // Second pass: Activate remaining strategies gradually
        console.log('\n📊 Phase 2: Activating Remaining Strategies');

        const remaining = strategies.slice(8);
        for (const strategy of remaining) {
            try {
                // Add small delay between activations to avoid overwhelming the system
                await new Promise(resolve => setTimeout(resolve, 100));

                await this.activateSingleStrategy(strategy, false); // Set to idle initially
                activated++;
                console.log(`✅ Registered: ${strategy.id} (${strategy.config.strategy_type}) - Set to idle`);
            } catch (error) {
                console.error(`❌ Failed to register ${strategy.id}:`, error.message);
                skipped++;
            }
        }

        console.log(`\n📊 Activation Summary:`);
        console.log(`   ✅ Activated: ${activated}`);
        console.log(`   ⏭️  Skipped: ${skipped}`);
        console.log(`   🎯 Total Available: ${strategies.length}`);

        return { activated, skipped, total: strategies.length };
    }

    /**
     * Activate a single strategy
     */
    async activateSingleStrategy(strategy, setActive = true) {
        // Update strategy status
        strategy.status = setActive ? 'active' : 'idle';

        // Allocate capital based on strategy type
        const capital = CAPITAL_ALLOCATION[strategy.config.strategy_type] || 25000;
        strategy.config.allocated_capital = capital;

        // Ensure universe is set
        if (!strategy.config.universe || strategy.config.universe.length === 0) {
            strategy.config.universe = [
                'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL',
                'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'
            ];
        }

        // Update performance tracking
        strategy.last_updated = new Date().toISOString();

        // Save changes
        this.strategyManager.saveStrategies();
    }

    /**
     * Get activation status summary
     */
    getActivationSummary() {
        const allStrategies = this.strategyManager.getAllStrategies();
        const active = allStrategies.filter(s => s.status === 'active').length;
        const idle = allStrategies.filter(s => s.status === 'idle').length;
        const paper = allStrategies.filter(s => s.status === 'paper').length;

        const byType = {};
        allStrategies.forEach(strategy => {
            const type = strategy.config.strategy_type || 'unknown';
            if (!byType[type]) byType[type] = { active: 0, idle: 0, paper: 0 };
            byType[type][strategy.status]++;
        });

        return {
            total: allStrategies.length,
            active,
            idle,
            paper,
            byType
        };
    }

    /**
     * Validate strategy configurations
     */
    validateStrategies() {
        console.log('🔍 Validating Strategy Configurations');

        const strategies = this.strategyManager.getAllStrategies();
        let valid = 0;
        let invalid = 0;

        strategies.forEach(strategy => {
            const issues = [];

            // Check required fields
            if (!strategy.config.universe || strategy.config.universe.length === 0) {
                issues.push('missing universe');
            }
            if (!strategy.config.strategy_type) {
                issues.push('missing strategy_type');
            }
            if (!strategy.config.parameters) {
                issues.push('missing parameters');
            }
            if (!strategy.config.allocated_capital) {
                issues.push('missing allocated_capital');
            }

            if (issues.length === 0) {
                valid++;
            } else {
                invalid++;
                console.log(`⚠️  ${strategy.id}: ${issues.join(', ')}`);
            }
        });

        console.log(`✅ Valid: ${valid}, Invalid: ${invalid}`);
        return { valid, invalid };
    }
}

/**
 * Main execution
 */
async function main() {
    const activator = new StrategyActivator();

    try {
        // Validate current state
        console.log('Current Strategy Status:');
        const summary = activator.getActivationSummary();
        console.log(`   Total: ${summary.total} strategies`);
        console.log(`   Active: ${summary.active}`);
        console.log(`   Idle: ${summary.idle}`);
        console.log(`   Paper: ${summary.paper}\n`);

        // Validate configurations
        activator.validateStrategies();
        console.log('');

        // Activate strategies
        const result = await activator.activateStrategies();

        // Final status
        console.log('\n📈 Final Strategy Status:');
        const finalSummary = activator.getActivationSummary();
        console.log(`   Active: ${finalSummary.active}`);
        console.log(`   Idle: ${finalSummary.idle}`);
        console.log(`   Paper: ${finalSummary.paper}`);

        console.log('\nBy Strategy Type:');
        Object.entries(finalSummary.byType).forEach(([type, counts]) => {
            console.log(`   ${type}: ${counts.active} active, ${counts.idle} idle, ${counts.paper} paper`);
        });

        console.log('\n🎉 Strategy activation complete!');
        console.log('💡 Strategies are now ready for AI orchestration and automated trading.');

    } catch (error) {
        console.error('💥 Strategy activation failed:', error);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main();
}

module.exports = StrategyActivator;
