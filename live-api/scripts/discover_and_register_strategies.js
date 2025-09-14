#!/usr/bin/env node
/**
 * Strategy Discovery and Registration Script
 *
 * Discovers all trading strategies from the trading_bot/strategies directory
 * and registers them with the live-api system for multi-asset deployment.
 */

const fs = require('fs');
const path = require('path');
const StrategyManager = require('../services/strategy_manager');

// Strategy universe - major assets to deploy across
const STRATEGY_UNIVERSE = [
    'SPY',   // S&P 500 ETF
    'QQQ',   // Nasdaq 100 ETF
    'AAPL',  // Apple
    'MSFT',  // Microsoft
    'GOOGL', // Google
    'AMZN',  // Amazon
    'TSLA',  // Tesla
    'NVDA',  // Nvidia
    'META',  // Meta
    'NFLX'   // Netflix
];

// Strategy configurations by type
const STRATEGY_CONFIGS = {
    // Volatility strategies (our enhanced straddle/strangle)
    volatility: {
        name: 'Enhanced Volatility Spread',
        description: 'Advanced options volatility strategies with dynamic positioning',
        asset_class: 'options',
        strategy_type: 'volatility',
        universe: STRATEGY_UNIVERSE,
        parameters: {
            max_position_size: 0.05, // 5% of capital per position
            volatility_threshold: 0.15,
            min_iv_rank: 0.2,
            max_iv_rank: 0.8,
            max_holding_period_days: 45,
            profit_target_pct: 0.25,
            stop_loss_pct: 0.15
        },
        promotion_criteria: {
            min_trades: 20,
            min_win_rate: 0.60,
            min_sharpe_ratio: 1.2,
            max_drawdown: 0.10,
            min_total_return: 0.15
        }
    },

    // Mean reversion strategies
    mean_reversion: {
        name: 'Enhanced Mean Reversion',
        description: 'Statistical mean reversion across multiple timeframes',
        asset_class: 'stocks',
        strategy_type: 'mean_reversion',
        universe: STRATEGY_UNIVERSE,
        parameters: {
            lookback_periods: [20, 50, 200],
            entry_threshold: 2.0, // Standard deviations
            exit_threshold: 0.5,
            max_holding_period: 20,
            position_size_pct: 0.03
        },
        promotion_criteria: {
            min_trades: 30,
            min_win_rate: 0.55,
            min_sharpe_ratio: 1.0,
            max_drawdown: 0.08
        }
    },

    // Momentum strategies
    momentum: {
        name: 'Multi-Timeframe Momentum',
        description: 'Momentum-based trading with regime adaptation',
        asset_class: 'stocks',
        strategy_type: 'momentum',
        universe: STRATEGY_UNIVERSE,
        parameters: {
            fast_ma: 20,
            slow_ma: 50,
            rsi_period: 14,
            momentum_threshold: 0.02,
            volume_multiplier: 1.5
        },
        promotion_criteria: {
            min_trades: 25,
            min_win_rate: 0.58,
            min_sharpe_ratio: 1.1,
            max_drawdown: 0.09
        }
    },

    // Breakout strategies
    breakout: {
        name: 'Volume-Weighted Breakout',
        description: 'Breakout trading with volume confirmation',
        asset_class: 'stocks',
        strategy_type: 'breakout',
        universe: STRATEGY_UNIVERSE,
        parameters: {
            consolidation_period: 20,
            breakout_threshold: 0.03,
            volume_threshold: 1.2,
            stop_distance: 0.05,
            target_distance: 0.10
        },
        promotion_criteria: {
            min_trades: 15,
            min_win_rate: 0.62,
            min_sharpe_ratio: 1.3,
            max_drawdown: 0.07
        }
    },

    // Trend following strategies
    trend_following: {
        name: 'Adaptive Trend Following',
        description: 'Trend following with dynamic position sizing',
        asset_class: 'stocks',
        strategy_type: 'trend_following',
        universe: STRATEGY_UNIVERSE,
        parameters: {
            trend_period: 50,
            confirmation_period: 20,
            trend_strength_threshold: 0.015,
            position_scaling: true,
            max_pyramiding: 3
        },
        promotion_criteria: {
            min_trades: 20,
            min_win_rate: 0.65,
            min_sharpe_ratio: 1.4,
            max_drawdown: 0.06
        }
    },

    // Options income strategies
    income: {
        name: 'Options Income Engine',
        description: 'Systematic options selling for income generation',
        asset_class: 'options',
        strategy_type: 'income',
        universe: STRATEGY_UNIVERSE,
        parameters: {
            target_premium_pct: 0.02, // 2% premium target
            max_dte: 45,
            min_dte: 7,
            delta_target: 0.20,
            position_limit_pct: 0.10
        },
        promotion_criteria: {
            min_trades: 50,
            min_win_rate: 0.75,
            min_sharpe_ratio: 1.8,
            max_drawdown: 0.05
        }
    }
};

class StrategyDiscoveryService {
    constructor() {
        this.strategyManager = new StrategyManager();
        this.discoveredStrategies = new Map();
    }

    /**
     * Discover strategies from the trading_bot directory
     */
    async discoverStrategies() {
        console.log('🔍 Discovering strategies from trading_bot/strategies...');

        const strategiesDir = path.join(__dirname, '../../trading_bot/strategies');

        if (!fs.existsSync(strategiesDir)) {
            console.error('❌ Trading bot strategies directory not found:', strategiesDir);
            return;
        }

        // Discover by category
        const categories = [
            'options/volatility_spreads',
            'options/income_strategies',
            'options/vertical_spreads',
            'stocks/mean_reversion',
            'stocks/momentum',
            'stocks/breakout',
            'stocks/trend',
            'forex/trend',
            'forex/momentum'
        ];

        for (const category of categories) {
            await this.discoverCategory(strategiesDir, category);
        }

        console.log(`✅ Discovered ${this.discoveredStrategies.size} strategy files`);
    }

    /**
     * Discover strategies in a specific category
     */
    async discoverCategory(baseDir, category) {
        const categoryPath = path.join(baseDir, category);

        if (!fs.existsSync(categoryPath)) {
            console.log(`⚠️  Category not found: ${category}`);
            return;
        }

        const files = fs.readdirSync(categoryPath, { recursive: true })
            .filter(file => file.endsWith('.py') && !file.includes('__pycache__'));

        for (const file of files) {
            const filePath = path.join(categoryPath, file);
            const strategyName = path.basename(file, '.py');

            // Skip base classes and test files
            if (strategyName.includes('base') ||
                strategyName.includes('test') ||
                strategyName.includes('example')) {
                continue;
            }

            // Extract strategy type from path
            const strategyType = this.extractStrategyType(category);
            const config = STRATEGY_CONFIGS[strategyType] || STRATEGY_CONFIGS.mean_reversion;

            this.discoveredStrategies.set(strategyName, {
                name: strategyName,
                file: filePath,
                category: category,
                type: strategyType,
                config: {
                    ...config,
                    id: strategyName.toLowerCase().replace(/[^a-z0-9]/g, '_'),
                    source: 'trading_bot',
                    category: category
                }
            });
        }
    }

    /**
     * Extract strategy type from category path
     */
    extractStrategyType(category) {
        if (category.includes('volatility_spreads')) return 'volatility';
        if (category.includes('income')) return 'income';
        if (category.includes('mean_reversion')) return 'mean_reversion';
        if (category.includes('momentum')) return 'momentum';
        if (category.includes('breakout')) return 'breakout';
        if (category.includes('trend')) return 'trend_following';
        return 'mean_reversion'; // default
    }

    /**
     * Register discovered strategies with the strategy manager
     */
    async registerStrategies() {
        console.log('📝 Registering strategies with live-api...');

        let registered = 0;
        let skipped = 0;

        for (const [strategyName, strategyInfo] of this.discoveredStrategies) {
            try {
                // Check if already registered
                const existing = this.strategyManager.getStrategy(strategyInfo.config.id);
                if (existing) {
                    console.log(`⏭️  Skipping already registered: ${strategyName}`);
                    skipped++;
                    continue;
                }

                // Register with strategy manager
                const strategy = this.strategyManager.registerStrategy(
                    strategyInfo.config.id,
                    strategyInfo.config
                );

                console.log(`✅ Registered: ${strategyName} (${strategyInfo.type})`);
                registered++;

            } catch (error) {
                console.error(`❌ Failed to register ${strategyName}:`, error.message);
            }
        }

        console.log(`📊 Registration complete: ${registered} registered, ${skipped} skipped`);
        return { registered, skipped };
    }

    /**
     * Special handling for our enhanced straddle/strangle strategy
     */
    async registerEnhancedVolatilityStrategy() {
        console.log('🚀 Registering Enhanced Straddle/Strangle Strategy...');

        const strategyId = 'enhanced_straddle_strangle';
        const config = {
            ...STRATEGY_CONFIGS.volatility,
            id: strategyId,
            name: 'Enhanced Straddle/Strangle Strategy',
            description: 'Advanced volatility spreads with machine learning and risk management',
            source: 'trading_bot',
            category: 'options/volatility_spreads',
            parameters: {
                ...STRATEGY_CONFIGS.volatility.parameters,
                // Enhanced parameters
                ml_enabled: true,
                dynamic_positioning: true,
                volatility_regime_adaptation: true,
                greeks_monitoring: true,
                adaptive_expiration: true
            }
        };

        try {
            const strategy = this.strategyManager.registerStrategy(strategyId, config);
            console.log('✅ Enhanced volatility strategy registered successfully');
            return strategy;
        } catch (error) {
            console.error('❌ Failed to register enhanced strategy:', error.message);
            return null;
        }
    }

    /**
     * Verify strategy universe configuration
     */
    verifyUniverse() {
        console.log('🌍 Verifying strategy universe configuration...');

        const strategies = this.strategyManager.getAllStrategies();
        let totalStrategies = 0;
        let strategiesWithUniverse = 0;

        strategies.forEach(strategy => {
            totalStrategies++;
            if (strategy.config && strategy.config.universe && strategy.config.universe.length > 0) {
                strategiesWithUniverse++;
                console.log(`✅ ${strategy.id}: ${strategy.config.universe.length} assets`);
            } else {
                console.log(`⚠️  ${strategy.id}: No universe configured`);
            }
        });

        console.log(`📊 Universe verification: ${strategiesWithUniverse}/${totalStrategies} strategies configured`);
        return { totalStrategies, strategiesWithUniverse };
    }
}

/**
 * Main execution
 */
async function main() {
    console.log('🚀 Starting Strategy Discovery and Registration\n');

    const discovery = new StrategyDiscoveryService();

    try {
        // Discover strategies
        await discovery.discoverStrategies();

        // Register discovered strategies
        await discovery.registerStrategies();

        // Special handling for enhanced strategy
        await discovery.registerEnhancedVolatilityStrategy();

        // Verify configuration
        discovery.verifyUniverse();

        console.log('\n🎉 Strategy discovery and registration complete!');
        console.log('💡 Next: Start AI orchestrator to begin automated strategy deployment');

    } catch (error) {
        console.error('💥 Strategy discovery failed:', error);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main();
}

module.exports = StrategyDiscoveryService;
