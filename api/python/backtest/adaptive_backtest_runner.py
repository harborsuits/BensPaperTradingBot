#!/usr/bin/env python
"""
Adaptive Strategy Backtest Runner

Main script for running backtests, parameter sweeps, and optimizations
for the adaptive trading strategy system.
"""

import logging
import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from trading_bot.backtest.market_data_generator import MarketDataGenerator, MarketRegimeType
from trading_bot.backtest.adaptive_backtest_engine import AdaptiveBacktestEngine
from trading_bot.backtest.parameter_sweep import ParameterSweep
from trading_bot.backtest.optimization_engine import OptimizationEngine
from trading_bot.backtest.walk_forward_testing import WalkForwardTester
from trading_bot.backtest.walk_forward_integration import WalkForwardRunner
from trading_bot.backtest.real_data_integration import RealMarketDataProvider, StrategyIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('adaptive_backtest.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_test_strategies() -> Dict[str, Dict[str, Any]]:
    """Set up test strategies for backtesting"""
    strategies = {
        'trend_following': {
            'name': 'Trend Following Strategy',
            'description': 'Follows market trends using moving averages',
            'category': 'trend_following',
            'symbols': ['SPY', 'QQQ', 'IWM', 'EEM'],
            'timeframes': ['1d'],
            'parameters': {
                'ma_fast': 20,
                'ma_slow': 50,
                'trailing_stop_pct': 2.0,
                'profit_target_pct': 3.0
            }
        },
        'mean_reversion': {
            'name': 'Mean Reversion Strategy',
            'description': 'Trades mean reversion using Bollinger Bands',
            'category': 'mean_reversion',
            'symbols': ['SPY', 'IWM', 'EEM', 'GLD'],
            'timeframes': ['1d'],
            'parameters': {
                'entry_threshold': 2.0,
                'profit_target_pct': 1.5,
                'stop_loss_pct': 1.0
            }
        },
        'breakout_strategy': {
            'name': 'Breakout Strategy',
            'description': 'Trades range breakouts',
            'category': 'breakout',
            'symbols': ['QQQ', 'IWM', 'EFA', 'USO'],
            'timeframes': ['1d'],
            'parameters': {
                'breakout_threshold': 2.0,
                'confirmation_period': 3,
                'trailing_stop_pct': 2.0
            }
        },
        'volatility_strategy': {
            'name': 'Volatility Strategy',
            'description': 'Trades volatility expansion and contraction',
            'category': 'volatility',
            'symbols': ['VXX', 'SPY', 'TLT', 'GLD'],
            'timeframes': ['1d'],
            'parameters': {
                'vix_threshold': 20,
                'position_size_scale': 0.8,
                'profit_target_mult': 1.3
            }
        }
    }
    
    return strategies

def setup_base_config() -> Dict[str, Any]:
    """Set up base configuration for adaptive strategy controller"""
    config = {
        'initial_equity': 10000.0,
        'allocation_frequency': 'daily',
        'parameter_update_frequency': 'daily',
        'performance_tracker': {
            'sharpe_window': 30,
            'performance_windows': [7, 30, 90],
            'min_trades_required': 5,
            'risk_free_rate': 0.0,
            'auto_save': True
        },
        'market_regime_detector': {
            'atr_period': 14,
            'trend_period': 20,
            'ma_fast_period': 20,
            'ma_slow_period': 50,
            'volatility_threshold': 1.5,
            'trend_strength_threshold': 0.6,
            'range_threshold': 0.3,
            'auto_save': True
        },
        'snowball_allocator': {
            'rebalance_frequency': 'daily',
            'snowball_reinvestment_ratio': 0.5,
            'min_weight': 0.05,
            'max_weight': 0.9,
            'normalization_method': 'simple',
            'smoothing_factor': 0.2
        },
        'position_sizer': {
            'base_risk_pct': 0.01,  # 1% base risk per trade
            'stop_loss_pct': 0.02,  # 2% default stop loss
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'perf_sharpe_window': 30,
            'perf_adj_min': 0.5,
            'perf_adj_max': 2.0
        },
        'emergency_brake': {
            'max_consecutive_losses': 3,
            'strategy_drawdown_limit_pct': 0.1,  # 10% strategy drawdown limit
            'global_drawdown_limit_pct': 0.05,   # 5% global drawdown limit
            'max_slippage_pct': 0.01            # 1% max slippage
        },
        'adaptive_risk_manager': {
            'equity_breakpoints': [10000, 25000, 50000, 100000],
            'initial_max_allocation': 0.9,  # 90% max allocation at lowest equity
            'target_max_allocation': 0.3,   # 30% max allocation at highest equity
            'initial_max_risk_pct': 0.02,   # 2% max risk at lowest equity
            'target_max_risk_pct': 0.005    # 0.5% max risk at highest equity
        }
    }
    
    return config

def run_basic_backtest():
    """Run a basic backtest with default configuration"""
    logger.info("Running basic backtest with default configuration")
    
    # Initialize components
    market_data_gen = MarketDataGenerator()
    backtest_engine = AdaptiveBacktestEngine(config={'output_dir': './results/basic_backtest'})
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Generate market data for different regimes
    market_data = {}
    
    # Test with a bull market
    for symbol in ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'VXX', 'TLT', 'USO', 'EFA']:
        market_data[symbol] = market_data_gen.generate_regime_data(
            symbol=symbol,
            regime=MarketRegimeType.BULL,
            days=252
        )
    
    # Run backtest
    results = backtest_engine.run_backtest(
        controller_config=base_config,
        market_data=market_data,
        strategies=strategies,
        simulation_days=252,
        name="basic_bull_market"
    )
    
    print("\nBasic Bull Market Backtest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Initial Equity: ${results['initial_equity']:.2f}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    
    return results

def run_multi_regime_backtest():
    """Run backtests across multiple market regimes"""
    logger.info("Running backtests across multiple market regimes")
    
    # Initialize components
    market_data_gen = MarketDataGenerator()
    backtest_engine = AdaptiveBacktestEngine(config={'output_dir': './results/multi_regime'})
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Define regimes to test
    regimes = [
        MarketRegimeType.BULL,
        MarketRegimeType.BEAR,
        MarketRegimeType.SIDEWAYS,
        MarketRegimeType.VOLATILE,
        MarketRegimeType.CRASH,
        MarketRegimeType.RECOVERY
    ]
    
    # Symbols to test
    symbols = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'VXX', 'TLT', 'USO', 'EFA']
    
    results = {}
    
    # Run backtest for each regime
    for regime in regimes:
        regime_name = regime.value
        logger.info(f"Running backtest for {regime_name} market")
        
        # Generate market data for this regime
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = market_data_gen.generate_regime_data(
                symbol=symbol,
                regime=regime,
                days=252
            )
        
        # Run backtest
        regime_results = backtest_engine.run_backtest(
            controller_config=base_config,
            market_data=market_data,
            strategies=strategies,
            simulation_days=252,
            name=f"multi_regime_{regime_name}"
        )
        
        results[regime_name] = {
            'total_return': regime_results['total_return'],
            'sharpe_ratio': regime_results['sharpe_ratio'],
            'max_drawdown': regime_results['max_drawdown'],
        }
    
    # Print summary results
    print("\nMulti-Regime Backtest Results:")
    print("-" * 50)
    print(f"{'Regime':<12} {'Return':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 50)
    
    for regime, metrics in results.items():
        print(f"{regime:<12} {metrics['total_return']:.2%}      {metrics['sharpe_ratio']:.2f}       {metrics['max_drawdown']:.2%}")
    
    # Generate comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot returns
    plt.subplot(3, 1, 1)
    returns = [metrics['total_return'] for metrics in results.values()]
    bars = plt.bar(results.keys(), returns)
    for i, bar in enumerate(bars):
        bar.set_color('green' if returns[i] >= 0 else 'red')
    plt.title('Total Return by Market Regime')
    plt.ylabel('Return')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Plot Sharpe ratios
    plt.subplot(3, 1, 2)
    sharpes = [metrics['sharpe_ratio'] for metrics in results.values()]
    bars = plt.bar(results.keys(), sharpes)
    for i, bar in enumerate(bars):
        bar.set_color('green' if sharpes[i] >= 0 else 'red')
    plt.title('Sharpe Ratio by Market Regime')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    # Plot max drawdowns
    plt.subplot(3, 1, 3)
    drawdowns = [metrics['max_drawdown'] for metrics in results.values()]
    bars = plt.bar(results.keys(), drawdowns)
    for i, bar in enumerate(bars):
        intensity = min(1.0, drawdowns[i] / 0.5)
        bar.set_color((1.0, 0.4 * (1 - intensity), 0.4 * (1 - intensity)))
    plt.title('Maximum Drawdown by Market Regime')
    plt.ylabel('Drawdown')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    plt.savefig('./results/multi_regime/regime_comparison.png')
    
    return results

def run_parameter_sweep():
    """Run parameter sweep to analyze sensitivity"""
    logger.info("Running parameter sweep for sensitivity analysis")
    
    # Initialize components
    market_data_gen = MarketDataGenerator()
    parameter_sweep = ParameterSweep(config={'output_dir': './results/parameter_sweep'})
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Define parameter grid to test
    parameter_grid = {
        'snowball_allocator.snowball_reinvestment_ratio': [0.2, 0.4, 0.6, 0.8],
        'snowball_allocator.min_weight': [0.03, 0.05, 0.1],
        'snowball_allocator.max_weight': [0.7, 0.8, 0.9],
        'position_sizer.atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'position_sizer.perf_adj_min': [0.3, 0.5, 0.7],
        'position_sizer.perf_adj_max': [1.5, 2.0, 2.5],
        'emergency_brake.max_consecutive_losses': [2, 3, 4, 5],
        'emergency_brake.strategy_drawdown_limit_pct': [0.05, 0.1, 0.15, 0.2]
    }
    
    # Generate market data - use a mixed market for more robust results
    market_data = {}
    symbols = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'VXX', 'TLT', 'USO', 'EFA']
    
    for symbol in symbols:
        market_data[symbol] = market_data_gen.generate_regime_data(
            symbol=symbol,
            regime=MarketRegimeType.MIXED,
            days=252
        )
    
    # Run parameter sweep
    results = parameter_sweep.run_sweep(
        parameter_grid=parameter_grid,
        base_config=base_config,
        market_data=market_data,
        strategies=strategies,
        simulation_days=252,
        name="sensitivity_analysis"
    )
    
    # Print most sensitive parameters
    sensitivity = results['sensitivity']
    
    print("\nParameter Sensitivity Analysis:")
    print("-" * 50)
    print(f"{'Parameter':<40} {'Impact on Sharpe':<15} {'Impact on Return':<15}")
    print("-" * 50)
    
    # Sort by impact on Sharpe ratio
    sorted_params = sorted(
        sensitivity.items(), 
        key=lambda x: x[1].get('sharpe_ratio', 0), 
        reverse=True
    )
    
    for param, impacts in sorted_params:
        sharpe_impact = impacts.get('sharpe_ratio', 0)
        return_impact = impacts.get('total_return', 0)
        print(f"{param:<40} {sharpe_impact:.3f}           {return_impact:.3f}")
    
    print("\nResults saved to ./results/parameter_sweep")
    
    return results

def run_regime_robustness_test():
    """Test configuration robustness across different market regimes"""
    logger.info("Running regime robustness test")
    
    # Initialize components
    parameter_sweep = ParameterSweep(config={'output_dir': './results/robustness_test'})
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Define regimes to test
    regimes = [
        MarketRegimeType.BULL,
        MarketRegimeType.BEAR,
        MarketRegimeType.SIDEWAYS,
        MarketRegimeType.VOLATILE,
        MarketRegimeType.CRASH,
        MarketRegimeType.RECOVERY
    ]
    
    # Symbols to test
    symbols = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'VXX', 'TLT', 'USO', 'EFA']
    
    # Run robustness test
    results = parameter_sweep.run_regime_robustness_test(
        config_to_test=base_config,
        strategies=strategies,
        symbols=symbols,
        regimes=regimes,
        days_per_regime=252,
        name="config_robustness"
    )
    
    # Print results
    if 'robustness_scores' in results:
        scores = results['robustness_scores']
        
        print("\nRegime Robustness Test Results:")
        print(f"Robustness Score: {scores.get('robustness_score', 0):.3f}")
        print(f"Average Sharpe: {scores.get('avg_sharpe', 0):.2f}")
        print(f"Minimum Sharpe: {scores.get('min_sharpe', 0):.2f}")
        print(f"Sharpe Std Dev: {scores.get('sharpe_std', 0):.2f}")
        print("\nResults saved to ./results/robustness_test")
    
    return results

def run_hyperparameter_optimization():
    """Run hyperparameter optimization using Optuna"""
    logger.info("Running hyperparameter optimization")
    
    # Initialize components
    market_data_gen = MarketDataGenerator()
    optimization_engine = OptimizationEngine(config={'output_dir': './results/optimization'})
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Define parameter ranges to optimize
    parameter_ranges = {
        'snowball_allocator.snowball_reinvestment_ratio': (0.1, 0.9),
        'snowball_allocator.min_weight': (0.01, 0.2),
        'snowball_allocator.max_weight': (0.6, 0.95),
        'snowball_allocator.smoothing_factor': (0.1, 0.5),
        'position_sizer.base_risk_pct': (0.005, 0.03),
        'position_sizer.atr_multiplier': (1.0, 4.0),
        'position_sizer.perf_adj_min': (0.2, 0.8),
        'position_sizer.perf_adj_max': (1.2, 3.0),
        'emergency_brake.max_consecutive_losses': (2, 6),
        'emergency_brake.strategy_drawdown_limit_pct': (0.05, 0.25),
        'emergency_brake.global_drawdown_limit_pct': (0.03, 0.1),
        'adaptive_risk_manager.initial_max_allocation': (0.5, 0.95),
        'adaptive_risk_manager.target_max_allocation': (0.2, 0.5)
    }
    
    # Generate mixed market data for optimization
    market_data = {}
    symbols = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'VXX', 'TLT', 'USO', 'EFA']
    
    for symbol in symbols:
        # Use a mixed market that contains different regimes
        # for more robust parameter optimization
        market_data[symbol] = market_data_gen.generate_regime_data(
            symbol=symbol,
            regime=MarketRegimeType.MIXED,
            days=252
        )
    
    # Run optimization with a smaller number of trials for demonstration
    results = optimization_engine.optimize(
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        market_data=market_data,
        strategies=strategies,
        simulation_days=252,
        n_trials=50,  # Reduced for demo purposes, use 100+ for real optimization
        timeout=1800,  # 30 minute timeout
        name="adaptive_optimization"
    )
    
    # Print best parameters
    print("\nHyperparameter Optimization Results:")
    print("-" * 50)
    print("Best Parameters:")
    
    for param, value in results['best_params'].items():
        print(f"{param}: {value}")
    
    print("\nBest Performance Metrics:")
    metrics = results['best_metrics']
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    print("\nResults saved to ./results/optimization")
    
    return results

def run_cross_validation_optimization():
    """Run cross-validation optimization across different market regimes"""
    logger.info("Running cross-validation optimization")
    
    # Initialize components
    optimization_engine = OptimizationEngine(config={'output_dir': './results/cross_validation'})
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Define parameter ranges to optimize
    parameter_ranges = {
        'snowball_allocator.snowball_reinvestment_ratio': (0.1, 0.9),
        'snowball_allocator.min_weight': (0.01, 0.2),
        'snowball_allocator.max_weight': (0.6, 0.95),
        'position_sizer.base_risk_pct': (0.005, 0.03),
        'position_sizer.atr_multiplier': (1.0, 4.0),
        'position_sizer.perf_adj_min': (0.2, 0.8),
        'position_sizer.perf_adj_max': (1.2, 3.0),
        'emergency_brake.max_consecutive_losses': (2, 6),
        'emergency_brake.strategy_drawdown_limit_pct': (0.05, 0.25),
        'emergency_brake.global_drawdown_limit_pct': (0.03, 0.1)
    }
    
    # Define regimes to test
    regimes = [
        MarketRegimeType.BULL,
        MarketRegimeType.BEAR,
        MarketRegimeType.SIDEWAYS,
        MarketRegimeType.VOLATILE
    ]
    
    # Symbols to test
    symbols = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'VXX', 'TLT']
    
    # Run cross-validation optimization with a small number of trials for demonstration
    results = optimization_engine.cross_validate_optimization(
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        strategies=strategies,
        regimes=regimes,
        symbols=symbols,
        days_per_regime=252,
        n_trials=40,  # Reduced for demo purposes
        name="cross_regime_optimization"
    )
    
    # Print robust parameters
    if 'robust_params' in results:
        print("\nCross-Validation Optimization Results:")
        print("-" * 50)
        print("Robust Parameters:")
        
        for param, value in results['robust_params'].items():
            print(f"{param}: {value}")
        
        if 'robustness_metrics' in results:
            print("\nRobustness Metrics:")
            metrics = results['robustness_metrics']
            
            print(f"Robustness Score: {metrics.get('robustness_score', 0):.3f}")
            print(f"Average Sharpe: {metrics.get('avg_sharpe', 0):.2f}")
            print(f"Minimum Sharpe: {metrics.get('min_sharpe', 0):.2f}")
            print(f"Average Return: {metrics.get('avg_return', 0):.2%}")
            print(f"Minimum Return: {metrics.get('min_return', 0):.2%}")
        
        print("\nTest Results by Regime:")
        if 'test_results' in results:
            for regime, regime_results in results['test_results'].items():
                if 'sharpe_ratio' in regime_results:
                    print(f"{regime}: Sharpe={regime_results['sharpe_ratio']:.2f}, Return={regime_results['total_return']:.2%}, DD={regime_results['max_drawdown']:.2%}")
        
        print("\nResults saved to ./results/cross_validation")
    
    return results

def run_walk_forward_test():
    """Run walk-forward testing to prevent overfitting by testing on unseen data"""
    logger.info("Running walk-forward testing...")
    
    # Create output directory
    os.makedirs('./results/walk_forward', exist_ok=True)
    
    # Setup walk-forward runner
    wf_runner = WalkForwardRunner(output_dir='./results/walk_forward')
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Define parameter ranges to test
    parameter_ranges = {
        'snowball_allocator.snowball_reinvestment_ratio': [0.3, 0.5, 0.7],
        'position_sizer.atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'position_sizer.perf_adj_max': [1.5, 2.0, 2.5]
    }
    
    # Test symbols
    symbols = ['SPY', 'QQQ', 'IWM']
    
    # Define date range (2 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Run basic walk-forward test
    results = wf_runner.run_basic_walk_forward(
        symbols=symbols,
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        start_date=start_date,
        end_date=end_date,
        train_window_size=252,  # 1 year training
        test_window_size=63,    # ~3 months testing
        step_size=21,           # ~1 month step
        name="basic_walk_forward"
    )
    
    # Print walk-forward testing results
    if 'metrics' in results:
        print("\nWalk-Forward Testing Results:")
        print("-" * 50)
        
        metrics = results['metrics']
        for key, value in metrics.items():
            if not isinstance(value, list) and not key.endswith('_std'):
                print(f"{key}: {value:.4f}")
        
        # Print in-sample vs out-of-sample performance
        if 'in_vs_out_sample' in results:
            print("\nIn-Sample vs Out-of-Sample Performance:")
            for metric, values in results['in_vs_out_sample'].items():
                print(f"{metric}:")
                print(f"  In-Sample Mean: {values['in_sample_mean']:.4f}")
                print(f"  Out-of-Sample Mean: {values['out_of_sample_mean']:.4f}")
                print(f"  Performance Drop: {values['percent_drop']:.2f}%")
        
        print("\nResults saved to ./results/walk_forward")
    
    return results

def run_multi_regime_walk_forward():
    """Run walk-forward testing across multiple market regimes"""
    logger.info("Running multi-regime walk-forward testing...")
    
    # Create output directory
    os.makedirs('./results/walk_forward/cross_regime', exist_ok=True)
    
    # Setup walk-forward runner
    wf_runner = WalkForwardRunner(output_dir='./results/walk_forward')
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Define parameter ranges to test
    parameter_ranges = {
        'snowball_allocator.snowball_reinvestment_ratio': [0.3, 0.5, 0.7],
        'position_sizer.atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'position_sizer.perf_adj_max': [1.5, 2.0, 2.5]
    }
    
    # Test symbols
    symbols = ['SPY', 'QQQ', 'IWM']
    
    # Define market regimes (example periods)
    today = datetime.now()
    regime_periods = {
        'bull_market': (
            datetime(2019, 1, 1),
            datetime(2019, 12, 31)
        ),
        'covid_crash': (
            datetime(2020, 2, 1),
            datetime(2020, 4, 30)
        ),
        'covid_recovery': (
            datetime(2020, 5, 1),
            datetime(2020, 12, 31)
        ),
        'inflation_period': (
            datetime(2021, 6, 1),
            datetime(2022, 6, 30)
        ),
        'recent_market': (
            datetime(2023, 1, 1),
            min(today, datetime(2023, 12, 31))
        )
    }
    
    # Run multi-regime walk-forward test
    results = wf_runner.run_multi_regime_walk_forward(
        symbols=symbols,
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        regime_periods=regime_periods,
        name="multi_regime_walk_forward"
    )
    
    # Print results summary
    print("\nMulti-Regime Walk-Forward Testing Results:")
    print("-" * 50)
    
    for regime, regime_results in results.items():
        if 'metrics' in regime_results:
            metrics = regime_results['metrics']
            print(f"\nRegime: {regime}")
            
            for key in ['sharpe_ratio_mean', 'total_return_mean', 'max_drawdown_mean']:
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nResults saved to ./results/walk_forward/cross_regime")
    
    return results

def run_nested_walk_forward():
    """Run nested walk-forward testing with inner and outer optimization loops"""
    logger.info("Running nested walk-forward testing...")
    
    # Create output directory
    os.makedirs('./results/walk_forward/nested', exist_ok=True)
    
    # Setup walk-forward runner
    wf_runner = WalkForwardRunner(output_dir='./results/walk_forward')
    
    # Setup strategies and base config
    strategies = setup_test_strategies()
    base_config = setup_base_config()
    
    # Define parameter ranges for outer optimization (less frequent updates)
    outer_parameter_ranges = {
        'snowball_allocator.max_weight': [0.7, 0.8, 0.9],
        'snowball_allocator.min_weight': [0.03, 0.05, 0.1],
        'emergency_brake.global_drawdown_limit_pct': [0.05, 0.1, 0.15]
    }
    
    # Define parameter ranges for inner optimization (more frequent updates)
    inner_parameter_ranges = {
        'snowball_allocator.snowball_reinvestment_ratio': [0.3, 0.5, 0.7],
        'position_sizer.atr_multiplier': [1.5, 2.0, 2.5, 3.0]
    }
    
    # Test symbols
    symbols = ['SPY', 'QQQ', 'IWM']
    
    # Define date range (5 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Run nested walk-forward test
    results = wf_runner.run_nested_walk_forward(
        symbols=symbols,
        outer_parameter_ranges=outer_parameter_ranges,
        inner_parameter_ranges=inner_parameter_ranges,
        base_config=base_config,
        start_date=start_date,
        end_date=end_date,
        name="nested_walk_forward"
    )
    
    # Print nested walk-forward testing results
    print("\nNested Walk-Forward Testing Results:")
    print("-" * 50)
    
    for i, window in enumerate(results['outer_windows']):
        print(f"\nOuter Window {i+1}:")
        print(f"  Train: {window['train_start']} to {window['train_end']}")
        print(f"  Test: {window['test_start']} to {window['test_end']}")
        
        if i < len(results['outer_best_parameters']):
            print("  Best outer parameters:")
            for param, value in results['outer_best_parameters'][i].items():
                print(f"    {param}: {value:.4f}")
    
    if 'test_performance' in results and results['test_performance']:
        avg_sharpe = np.mean([perf.get('sharpe_ratio', 0) for perf in results['test_performance']])
        avg_return = np.mean([perf.get('total_return', 0) for perf in results['test_performance']])
        
        print(f"\nAverage test performance:")
        print(f"  Sharpe Ratio: {avg_sharpe:.4f}")
        print(f"  Total Return: {avg_return:.4f}")
    
    print("\nResults saved to ./results/walk_forward/nested")
    
    return results

def main():
    """Main function to run the backtesting and optimization"""
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Adaptive Strategy Backtest Runner')
    parser.add_argument('--test', choices=['basic', 'regimes', 'sweep', 'robustness', 'optimize', 'cross_validate', 
                                         'walk_forward', 'multi_regime_walk_forward', 'nested_walk_forward', 'all'], 
                        default='basic', help='Test to run')
    
    args = parser.parse_args()
    
    # Run selected test
    if args.test == 'basic' or args.test == 'all':
        run_basic_backtest()
    
    if args.test == 'regimes' or args.test == 'all':
        run_multi_regime_backtest()
    
    if args.test == 'sweep' or args.test == 'all':
        run_parameter_sweep()
    
    if args.test == 'robustness' or args.test == 'all':
        run_regime_robustness_test()
    
    if args.test == 'optimize' or args.test == 'all':
        run_hyperparameter_optimization()
    
    if args.test == 'cross_validate' or args.test == 'all':
        run_cross_validation_optimization()
    
    if args.test == 'walk_forward' or args.test == 'all':
        run_walk_forward_test()
    
    if args.test == 'multi_regime_walk_forward' or args.test == 'all':
        run_multi_regime_walk_forward()
    
    if args.test == 'nested_walk_forward' or args.test == 'all':
        run_nested_walk_forward()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main()
