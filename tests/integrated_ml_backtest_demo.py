#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated ML & Backtest Demo

This script demonstrates the complete workflow of:
1. LSTM regime detection model training
2. Backtest results processing and integration
3. Performance analysis and visualization 
4. Strategy selection improvement through ML feedback

This serves as both a test and an example of how the components
work together to create an adaptive forex trading system.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import argparse
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integrated_demo')

# Import components
from trading_bot.strategies.strategy_template import MarketRegime
from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector
from trading_bot.backtesting.backtest_results import BacktestResultsManager
from trading_bot.backtesting.performance_integration import PerformanceIntegration
from trading_bot.ml.model_retraining import ModelRetrainer
from trading_bot.visualization.performance_dashboard import PerformanceDashboard


def generate_synthetic_backtest_data(output_dir: Optional[str] = None) -> str:
    """
    Generate synthetic backtest data for demonstration.
    
    Args:
        output_dir: Directory to save the data
        
    Returns:
        Path to the backtest data file
    """
    logger.info("Generating synthetic backtest data")
    
    if output_dir is None:
        output_dir = os.path.join(project_root, 'backtest_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get strategy selector for strategy list
    strategy_selector = ForexStrategySelector()
    strategies = list(strategy_selector.strategy_compatibility.keys())
    
    # Get regimes
    regimes = [r.name for r in MarketRegime if r != MarketRegime.UNKNOWN]
    
    # Create backtest result manager
    backtest_manager = BacktestResultsManager(output_dir)
    
    # Generate backtest results
    results_count = 0
    
    # Baseline performance by strategy type (before ML optimization)
    baseline_performance = {
        'trend_following': {'return': 12.5, 'drawdown': 8.0, 'win_rate': 58, 'profit_factor': 1.8},
        'breakout': {'return': 10.0, 'drawdown': 12.0, 'win_rate': 45, 'profit_factor': 1.6},
        'range': {'return': 8.0, 'drawdown': 6.0, 'win_rate': 62, 'profit_factor': 1.5},
        'momentum': {'return': 14.0, 'drawdown': 11.0, 'win_rate': 52, 'profit_factor': 1.7},
        'swing': {'return': 9.5, 'drawdown': 9.0, 'win_rate': 55, 'profit_factor': 1.4},
        'scalping': {'return': 7.5, 'drawdown': 5.0, 'win_rate': 65, 'profit_factor': 1.3}
    }
    
    # Strategy types
    strategy_types = {
        'forex_trend_following': 'trend_following',
        'forex_breakout': 'breakout',
        'forex_range': 'range',
        'forex_momentum': 'momentum',
        'forex_swing': 'swing',
        'forex_scalping': 'scalping'
    }
    
    # Optimal regime for each strategy
    optimal_regimes = {
        'forex_trend_following': ['TRENDING_UP', 'TRENDING_DOWN'],
        'forex_breakout': ['VOLATILE_BREAKOUT'],
        'forex_range': ['RANGING'],
        'forex_momentum': ['TRENDING_UP', 'TRENDING_DOWN'],
        'forex_swing': ['VOLATILE_REVERSAL'],
        'forex_scalping': ['RANGING', 'CHOPPY']
    }
    
    # Base parameters for each strategy
    base_parameters = {
        'forex_trend_following': {'ma_fast': 9, 'ma_slow': 21, 'atr_period': 14},
        'forex_breakout': {'breakout_period': 20, 'atr_multiple': 2.0},
        'forex_range': {'rsi_period': 14, 'overbought': 70, 'oversold': 30},
        'forex_momentum': {'roc_period': 10, 'signal_period': 9},
        'forex_swing': {'ema_period': 20, 'reversal_strength': 3},
        'forex_scalping': {'ma_period': 5, 'rsi_period': 5, 'take_profit_pips': 10}
    }
    
    # Generate results for different time periods
    end_date = datetime.now()
    
    # Generate three phases of results:
    # 1. Before ML (3-6 months ago)
    # 2. Basic ML (1-3 months ago)
    # 3. Enhanced ML with feedback (last month)
    periods = [
        {
            'name': 'Before ML',
            'start': end_date - timedelta(days=180),
            'end': end_date - timedelta(days=90),
            'performance_boost': 0.0  # No boost
        },
        {
            'name': 'Basic ML',
            'start': end_date - timedelta(days=90),
            'end': end_date - timedelta(days=30),
            'performance_boost': 0.15  # 15% boost with basic ML
        },
        {
            'name': 'Enhanced ML + Feedback',
            'start': end_date - timedelta(days=30),
            'end': end_date,
            'performance_boost': 0.30  # 30% boost with enhanced ML and feedback
        }
    ]
    
    np.random.seed(42)  # For reproducibility
    
    # Common symbols for testing
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    
    # Generate backtest results for each period
    for period in periods:
        logger.info(f"Generating backtest data for period: {period['name']}")
        
        for strategy_name in strategies:
            strategy_type = strategy_types.get(strategy_name, 'trend_following')
            base_perf = baseline_performance[strategy_type]
            
            # Get base parameters for this strategy
            params = base_parameters.get(strategy_name, {}).copy()
            
            # For each regime
            for regime in regimes:
                # More results for optimal regimes
                num_runs = 5 if regime in optimal_regimes.get(strategy_name, []) else 2
                
                # Regime compatibility factor (how well strategy performs in this regime)
                if regime in optimal_regimes.get(strategy_name, []):
                    compatibility = 0.9  # High compatibility 
                else:
                    compatibility = 0.5  # Medium compatibility
                
                # Period dates
                period_length = (period['end'] - period['start']).days
                segment_length = period_length // num_runs
                
                for run in range(num_runs):
                    # Generate dates for this run
                    run_start = period['start'] + timedelta(days=run * segment_length)
                    run_end = run_start + timedelta(days=segment_length - 1)
                    
                    # Apply performance modifiers
                    performance_modifier = (
                        compatibility * 
                        (1.0 + period['performance_boost']) * 
                        (1.0 + np.random.normal(0, 0.1))  # Random noise
                    )
                    
                    # Calculate metrics with modifiers
                    total_return = base_perf['return'] * performance_modifier
                    drawdown = base_perf['drawdown'] / (0.8 + 0.4 * performance_modifier)  # Lower is better
                    win_rate = min(95, base_perf['win_rate'] * performance_modifier)
                    profit_factor = base_perf['profit_factor'] * performance_modifier
                    
                    # Optimize parameters slightly for each run
                    run_params = params.copy()
                    for key in run_params:
                        # Add some noise to parameters
                        if isinstance(run_params[key], int):
                            run_params[key] += np.random.randint(-1, 2)  # -1, 0, or 1
                        elif isinstance(run_params[key], float):
                            run_params[key] *= (1.0 + np.random.normal(0, 0.05))  # ±5% noise
                    
                    # Create backtest result
                    backtest_id = backtest_manager.save_backtest_result(
                        strategy_name=strategy_name,
                        market_regime=regime,
                        start_date=run_start.strftime('%Y-%m-%d'),
                        end_date=run_end.strftime('%Y-%m-%d'),
                        symbols=symbols,
                        metrics={
                            'total_return': total_return,
                            'max_drawdown': drawdown,
                            'win_rate': win_rate,
                            'profit_factor': profit_factor,
                            'sharpe_ratio': (total_return / drawdown) * performance_modifier,
                            'trades': int(50 * performance_modifier)
                        },
                        parameters=run_params
                    )
                    
                    results_count += 1
    
    logger.info(f"Generated {results_count} synthetic backtest results")
    
    return backtest_manager.summary_file


def run_performance_integration(backtest_data_path: str) -> PerformanceIntegration:
    """
    Run performance integration to update strategy selector with backtest results.
    
    Args:
        backtest_data_path: Path to backtest data
        
    Returns:
        PerformanceIntegration instance after processing
    """
    logger.info("Running performance integration")
    
    # Create backtest manager pointing to the generated data
    backtest_dir = os.path.dirname(backtest_data_path)
    backtest_manager = BacktestResultsManager(backtest_dir)
    
    # Create performance integration
    perf_integration = PerformanceIntegration(backtest_manager)
    
    # Update performance records
    update_counts = perf_integration.update_performance_records()
    
    logger.info(f"Updated {sum(update_counts.values())} performance records")
    
    # Export performance matrix
    matrix_path = perf_integration.export_performance_matrix()
    logger.info(f"Exported performance matrix to {matrix_path}")
    
    return perf_integration


def run_model_retraining():
    """
    Run model retraining process.
    """
    logger.info("Simulating model retraining")
    
    # Initialize model retrainer
    retrainer = ModelRetrainer()
    
    # Check model health
    health = retrainer.check_model_health()
    logger.info("Model Health Check:")
    for model_type, metrics in health.items():
        status = metrics['status']
        status_marker = '✓' if status == 'healthy' else 'X' if status == 'missing' else '!'
        
        if metrics['exists']:
            logger.info(f"{status_marker} {model_type}: {status.upper()} (Age: {metrics['age_days']} days)")
        else:
            logger.info(f"{status_marker} {model_type}: MISSING")
    
    # Force retraining
    logger.info("Forcing model retraining")
    result = retrainer.retrain_models(force=True)
    
    if result['retrained']:
        logger.info("Retraining completed successfully!")
        for model_type, model_result in result['results'].items():
            if model_result.get('success', False):
                logger.info(f"- {model_type}: Succeeded (Accuracy: {model_result.get('accuracy', 'N/A')})")
            else:
                logger.info(f"- {model_type}: Failed ({model_result.get('error', 'Unknown error')})")
    else:
        logger.info(f"Retraining skipped: {result['reason']}")


def generate_performance_dashboard(perf_integration: PerformanceIntegration) -> str:
    """
    Generate performance dashboard.
    
    Args:
        perf_integration: PerformanceIntegration instance
        
    Returns:
        Path to the generated dashboard
    """
    logger.info("Generating performance dashboard")
    
    # Create dashboard with custom output directory
    visualizations_dir = os.path.join(project_root, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    dashboard = PerformanceDashboard(
        output_dir=visualizations_dir,
        perf_integration=perf_integration
    )
    
    # Generate dashboard
    dashboard_path = dashboard.generate_dashboard_summary()
    
    logger.info(f"Generated dashboard: {dashboard_path}")
    
    return dashboard_path


def demo_strategy_selector_improvements():
    """
    Demonstrate strategy selector improvements with ML and backtest integration.
    """
    logger.info("Demonstrating strategy selector improvements")
    
    # Create strategy selector
    selector = ForexStrategySelector()
    
    # Create synthetic market data for different regimes
    regimes_to_test = {
        MarketRegime.TRENDING_UP: {
            'last_candles': [
                {'open': 1.1, 'high': 1.15, 'low': 1.095, 'close': 1.13, 'volume': 1000},
                {'open': 1.13, 'high': 1.18, 'low': 1.125, 'close': 1.17, 'volume': 1200},
                {'open': 1.17, 'high': 1.22, 'low': 1.165, 'close': 1.21, 'volume': 1500},
                {'open': 1.21, 'high': 1.25, 'low': 1.205, 'close': 1.24, 'volume': 1800},
                {'open': 1.24, 'high': 1.28, 'low': 1.235, 'close': 1.27, 'volume': 2000},
            ]
        },
        MarketRegime.RANGING: {
            'last_candles': [
                {'open': 1.15, 'high': 1.18, 'low': 1.14, 'close': 1.17, 'volume': 1000},
                {'open': 1.17, 'high': 1.19, 'low': 1.15, 'close': 1.16, 'volume': 900},
                {'open': 1.16, 'high': 1.18, 'low': 1.15, 'close': 1.17, 'volume': 950},
                {'open': 1.17, 'high': 1.19, 'low': 1.16, 'close': 1.18, 'volume': 980},
                {'open': 1.18, 'high': 1.19, 'low': 1.15, 'close': 1.16, 'volume': 930},
            ]
        },
        MarketRegime.VOLATILE_BREAKOUT: {
            'last_candles': [
                {'open': 1.15, 'high': 1.16, 'low': 1.14, 'close': 1.15, 'volume': 900},
                {'open': 1.15, 'high': 1.16, 'low': 1.14, 'close': 1.16, 'volume': 950},
                {'open': 1.16, 'high': 1.17, 'low': 1.15, 'close': 1.17, 'volume': 1000},
                {'open': 1.17, 'high': 1.18, 'low': 1.16, 'close': 1.17, 'volume': 1050},
                {'open': 1.17, 'high': 1.25, 'low': 1.17, 'close': 1.24, 'volume': 3000},
            ]
        }
    }
    
    # Test strategy selection before and after performance updates
    results = []
    
    for regime_type, data in regimes_to_test.items():
        # Convert to DataFrame
        df = pd.DataFrame(data['last_candles'])
        df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='1H')
        df.set_index('datetime', inplace=True)
        
        # Create market data
        market_data = {'EURUSD': df.copy()}
        
        # Run selector before updating performance
        logger.info(f"Testing regime: {regime_type.name}")
        
        # Reset performance records for demonstration
        selector.performance_records = {}
        
        # Select strategy without performance feedback
        selected_before = selector.select_optimal_strategy(
            market_data=market_data, 
            current_time=datetime.now(),
            account_size=10000.0,
            use_performance_history=False
        )
        
        # Simulate some performance history
        for strategy_name in selector.strategy_compatibility.keys():
            # Add better performance for optimal strategies in each regime
            if regime_type == MarketRegime.TRENDING_UP and 'trend' in strategy_name:
                score = 0.85
            elif regime_type == MarketRegime.RANGING and 'range' in strategy_name:
                score = 0.88
            elif regime_type == MarketRegime.VOLATILE_BREAKOUT and 'breakout' in strategy_name:
                score = 0.9
            else:
                score = 0.6  # Average performance
            
            # Record multiple performance data points
            for _ in range(5):
                selector.record_strategy_performance(
                    strategy_name=strategy_name,
                    regime=regime_type,
                    performance_score=score * (1 + np.random.normal(0, 0.05))  # Add noise
                )
        
        # Select strategy with performance feedback
        selected_after = selector.select_optimal_strategy(
            market_data=market_data, 
            current_time=datetime.now(),
            account_size=10000.0,
            use_performance_history=True
        )
        
        results.append({
            'regime': regime_type.name,
            'before': selected_before,
            'after': selected_after
        })
    
    # Print comparison
    logger.info("\nStrategy Selection Before/After Performance Integration:")
    logger.info("=" * 60)
    logger.info(f"{'Regime':<20} {'Before':<20} {'After':<20}")
    logger.info("-" * 60)
    
    for result in results:
        logger.info(f"{result['regime']:<20} {result['before']:<20} {result['after']:<20}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run integrated ML & Backtest demo')
    parser.add_argument('--skip-gen', action='store_true', help='Skip synthetic data generation')
    parser.add_argument('--backtest-data', help='Path to existing backtest data')
    
    args = parser.parse_args()
    
    logger.info("Starting integrated ML & backtest demo")
    
    # Step 1: Generate synthetic backtest data
    if args.skip_gen and args.backtest_data:
        backtest_data_path = args.backtest_data
        logger.info(f"Using existing backtest data: {backtest_data_path}")
    else:
        backtest_data_path = generate_synthetic_backtest_data()
        logger.info(f"Generated synthetic backtest data: {backtest_data_path}")
    
    # Step 2: Run performance integration
    perf_integration = run_performance_integration(backtest_data_path)
    
    # Step 3: Demonstrate strategy selector improvements
    demo_strategy_selector_improvements()
    
    # Step 4: Generate performance dashboard
    dashboard_path = generate_performance_dashboard(perf_integration)
    
    # Step 5: Simulate model retraining
    run_model_retraining()
    
    logger.info("\nIntegrated demo completed successfully!")
    logger.info(f"Dashboard available at: {dashboard_path}")
    logger.info(f"Visualizations saved to: {os.path.join(project_root, 'visualizations')}")


if __name__ == "__main__":
    main()
