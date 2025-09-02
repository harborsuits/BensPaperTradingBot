"""
Walk Forward Integration Module

This module integrates walk-forward testing with the main backtesting framework.
It provides utilities to run walk-forward analysis from the main runner script.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from trading_bot.backtest.walk_forward_testing import WalkForwardTester
from trading_bot.backtest.adaptive_backtest_engine import AdaptiveBacktestEngine
from trading_bot.backtest.real_data_integration import RealMarketDataProvider, StrategyIntegration

# Set up logging
logger = logging.getLogger(__name__)

class WalkForwardRunner:
    """
    Runner for walk-forward testing that integrates with the main backtesting framework.
    
    This class provides methods to run different types of walk-forward analysis:
    1. Basic walk-forward: Train on historical data, test on future data, roll forward
    2. Multi-regime walk-forward: Test across different market regimes
    3. Nested walk-forward: Optimize parameters within each walk-forward window
    4. Cross-asset walk-forward: Test parameter robustness across different assets
    """
    
    def __init__(self, output_dir: str = './results/walk_forward'):
        """
        Initialize the walk-forward runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.backtest_engine = AdaptiveBacktestEngine()
        self.walk_forward_tester = WalkForwardTester(self.backtest_engine, output_dir)
        self.market_data_provider = RealMarketDataProvider()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def run_basic_walk_forward(self,
                            symbols: List[str],
                            parameter_ranges: Dict[str, List[Any]],
                            base_config: Dict[str, Any],
                            strategies: Optional[Dict[str, Dict[str, Any]]] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            train_window_size: int = 252,  # Default 1 year
                            test_window_size: int = 63,    # Default ~3 months
                            step_size: int = 21,           # Default ~1 month
                            optimization_metric: str = 'sharpe_ratio',
                            use_cached_data: bool = True,
                            name: str = 'basic_walk_forward') -> Dict[str, Any]:
        """
        Run basic walk-forward analysis using real market data.
        
        Args:
            symbols: List of trading symbols
            parameter_ranges: Dict of parameter names to lists of values to test
            base_config: Base configuration for the strategy controller
            strategies: Dict of strategies to use (if None, will use default strategies)
            start_date: Start date for data (if None, will use 3 years ago)
            end_date: End date for data (if None, will use today)
            train_window_size: Number of days in training window
            test_window_size: Number of days in testing window
            step_size: Number of days to move forward between tests
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            use_cached_data: Whether to use cached market data if available
            name: Name for this test run
            
        Returns:
            Dict of results
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=3*365)  # Default to 3 years
        
        logger.info(f"Running basic walk-forward analysis from {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {symbols}")
        
        # Get market data
        market_data = self.market_data_provider.get_multi_symbol_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cached_data
        )
        
        # Check if we have data for all symbols
        if not all(symbol in market_data for symbol in symbols):
            missing = [symbol for symbol in symbols if symbol not in market_data]
            logger.error(f"Missing market data for symbols: {missing}")
            return {'error': f"Missing market data for symbols: {missing}"}
        
        # Get strategies if not provided
        if strategies is None:
            # Try to load strategies from the trading system
            strategies = StrategyIntegration.load_all_strategies()
            
            # If no strategies found, use default test strategies
            if not strategies:
                strategies = {
                    'trend_following': {
                        'name': 'Trend Following Strategy',
                        'description': 'Follows market trends using moving averages',
                        'category': 'trend_following',
                        'parameters': {
                            'ma_fast': 20,
                            'ma_slow': 50,
                            'trailing_stop_pct': 2.0
                        }
                    },
                    'mean_reversion': {
                        'name': 'Mean Reversion Strategy',
                        'description': 'Trades mean reversion using Bollinger Bands',
                        'category': 'mean_reversion',
                        'parameters': {
                            'lookback': 20,
                            'entry_threshold': 2.0,
                            'exit_threshold': 0.0
                        }
                    }
                }
        
        # Run walk-forward analysis
        results = self.walk_forward_tester.run_walk_forward_analysis(
            market_data=market_data,
            parameter_ranges=parameter_ranges,
            base_config=base_config,
            strategies=strategies,
            train_window_size=train_window_size,
            test_window_size=test_window_size,
            step_size=step_size,
            optimization_metric=optimization_metric,
            name=name
        )
        
        return results

    def run_multi_regime_walk_forward(self,
                                    symbols: List[str],
                                    parameter_ranges: Dict[str, List[Any]],
                                    base_config: Dict[str, Any],
                                    strategies: Optional[Dict[str, Dict[str, Any]]] = None,
                                    regime_periods: Optional[Dict[str, Tuple[datetime, datetime]]] = None,
                                    optimization_metric: str = 'sharpe_ratio',
                                    use_cached_data: bool = True,
                                    name: str = 'multi_regime_walk_forward') -> Dict[str, Dict[str, Any]]:
        """
        Run walk-forward analysis across multiple market regimes.
        
        Args:
            symbols: List of trading symbols
            parameter_ranges: Dict of parameter names to lists of values to test
            base_config: Base configuration for the strategy controller
            strategies: Dict of strategies to use (if None, will use default strategies)
            regime_periods: Dict mapping regime names to (start_date, end_date) tuples
                           If None, will use predefined periods for bull, bear, and sideways markets
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            use_cached_data: Whether to use cached market data if available
            name: Name for this test run
            
        Returns:
            Dict mapping regime names to results
        """
        # Define default market regimes if not provided
        if regime_periods is None:
            # Define sample market regimes
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
        
        logger.info(f"Running multi-regime walk-forward analysis for {len(regime_periods)} regimes")
        
        # Run walk-forward analysis for each regime
        results = {}
        for regime_name, (start_date, end_date) in regime_periods.items():
            logger.info(f"Processing regime: {regime_name} ({start_date.date()} to {end_date.date()})")
            
            # Run walk-forward analysis for this regime
            regime_results = self.run_basic_walk_forward(
                symbols=symbols,
                parameter_ranges=parameter_ranges,
                base_config=base_config,
                strategies=strategies,
                start_date=start_date,
                end_date=end_date,
                train_window_size=min(252, (end_date - start_date).days // 2),  # Use half the regime period for training
                test_window_size=min(126, (end_date - start_date).days // 4),   # Use quarter the regime period for testing
                step_size=min(21, (end_date - start_date).days // 10),          # Smaller step size for shorter regimes
                optimization_metric=optimization_metric,
                use_cached_data=use_cached_data,
                name=f"{name}_{regime_name}"
            )
            
            results[regime_name] = regime_results
        
        # Calculate meta-statistics across regimes
        self._calculate_cross_regime_statistics(results, name)
        
        return results
    
    def run_nested_walk_forward(self,
                             symbols: List[str],
                             outer_parameter_ranges: Dict[str, List[Any]],
                             inner_parameter_ranges: Dict[str, List[Any]],
                             base_config: Dict[str, Any],
                             strategies: Optional[Dict[str, Dict[str, Any]]] = None,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             outer_train_window: int = 756,  # 3 years
                             outer_test_window: int = 252,   # 1 year
                             inner_train_window: int = 252,  # 1 year
                             inner_test_window: int = 63,    # ~3 months
                             outer_step_size: int = 252,     # 1 year step
                             inner_step_size: int = 21,      # ~1 month step
                             optimization_metric: str = 'sharpe_ratio',
                             use_cached_data: bool = True,
                             name: str = 'nested_walk_forward') -> Dict[str, Any]:
        """
        Run nested walk-forward analysis with inner and outer optimization loops.
        
        The outer loop optimizes stable, long-term parameters.
        The inner loop optimizes more dynamic, shorter-term parameters.
        
        Args:
            symbols: List of trading symbols
            outer_parameter_ranges: Dict of parameter names to lists of values for outer optimization
            inner_parameter_ranges: Dict of parameter names to lists of values for inner optimization
            base_config: Base configuration for the strategy controller
            strategies: Dict of strategies to use (if None, will use default strategies)
            start_date: Start date for data (if None, will use 5 years ago)
            end_date: End date for data (if None, will use today)
            outer_train_window: Number of days in outer training window
            outer_test_window: Number of days in outer testing window
            inner_train_window: Number of days in inner training window
            inner_test_window: Number of days in inner testing window
            outer_step_size: Number of days to move forward between outer tests
            inner_step_size: Number of days to move forward between inner tests
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            use_cached_data: Whether to use cached market data if available
            name: Name for this test run
            
        Returns:
            Dict of results
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=5*365)  # Default to 5 years
        
        logger.info(f"Running nested walk-forward analysis from {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Outer parameters: {list(outer_parameter_ranges.keys())}")
        logger.info(f"Inner parameters: {list(inner_parameter_ranges.keys())}")
        
        # Get market data
        market_data = self.market_data_provider.get_multi_symbol_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cached_data
        )
        
        # Check if we have data for all symbols
        if not all(symbol in market_data for symbol in symbols):
            missing = [symbol for symbol in symbols if symbol not in market_data]
            logger.error(f"Missing market data for symbols: {missing}")
            return {'error': f"Missing market data for symbols: {missing}"}
        
        # Get strategies if not provided
        if strategies is None:
            strategies = StrategyIntegration.load_all_strategies()
        
        # Calculate number of windows for outer loop
        total_days = (end_date - start_date).days
        num_outer_windows = max(1, (total_days - outer_train_window - outer_test_window) // outer_step_size + 1)
        
        logger.info(f"Running {num_outer_windows} outer windows")
        
        # Initialize results
        results = {
            'outer_windows': [],
            'inner_results': [],
            'outer_best_parameters': [],
            'test_performance': []
        }
        
        # Run the nested walk-forward analysis
        for i in range(num_outer_windows):
            outer_start_idx = i * outer_step_size
            outer_start_date = start_date + timedelta(days=outer_start_idx)
            outer_train_end_date = outer_start_date + timedelta(days=outer_train_window)
            outer_test_end_date = min(outer_train_end_date + timedelta(days=outer_test_window), end_date)
            
            logger.info(f"Outer Window {i+1}/{num_outer_windows}:")
            logger.info(f"  Outer Training: {outer_start_date.date()} to {outer_train_end_date.date()}")
            logger.info(f"  Outer Testing:  {outer_train_end_date.date()} to {outer_test_end_date.date()}")
            
            # Create training data subset for outer window
            outer_train_data = {}
            for symbol, df in market_data.items():
                train_df = df[(df['date'] >= outer_start_date) & (df['date'] <= outer_train_end_date)].copy()
                outer_train_data[symbol] = train_df
            
            # Create testing data subset for outer window
            outer_test_data = {}
            for symbol, df in market_data.items():
                test_df = df[(df['date'] > outer_train_end_date) & (df['date'] <= outer_test_end_date)].copy()
                outer_test_data[symbol] = test_df
            
            # Optimize outer parameters
            from trading_bot.backtest.optimization_engine import OptimizationEngine
            
            # Convert parameter ranges to tuples for optimizer
            outer_range_tuples = {}
            for param, values in outer_parameter_ranges.items():
                if isinstance(values, tuple) and len(values) == 2:
                    outer_range_tuples[param] = values
                elif isinstance(values, list) and len(values) > 0:
                    outer_range_tuples[param] = (min(values), max(values))
            
            # Run optimization for outer parameters
            outer_optimizer = OptimizationEngine(metric=optimization_metric)
            outer_result = outer_optimizer.optimize(
                parameter_ranges=outer_range_tuples,
                base_config=base_config,
                market_data=outer_train_data,
                strategies=strategies,
                n_trials=20  # Limit trials for faster walk-forward testing
            )
            
            # Get best outer parameters
            best_outer_params = outer_result['best_params']
            logger.info(f"  Best outer parameters: {best_outer_params}")
            
            # Create base config with optimized outer parameters
            inner_base_config = base_config.copy()
            for param, value in best_outer_params.items():
                inner_base_config[param] = value
            
            # Run inner walk-forward with fixed outer parameters
            inner_results = self.walk_forward_tester.run_walk_forward_analysis(
                market_data=outer_test_data,
                parameter_ranges=inner_parameter_ranges,
                base_config=inner_base_config,
                strategies=strategies,
                train_window_size=inner_train_window,
                test_window_size=inner_test_window,
                step_size=inner_step_size,
                optimization_metric=optimization_metric,
                name=f"{name}_outer_{i+1}_inner"
            )
            
            # Run test with best outer parameters and best inner parameters from last window
            if inner_results['parameters']:
                best_inner_params = inner_results['parameters'][-1]
                logger.info(f"  Best inner parameters from last window: {best_inner_params}")
                
                # Combine outer and inner parameters
                test_config = inner_base_config.copy()
                for param, value in best_inner_params.items():
                    test_config[param] = value
                
                # Run backtest on test data with optimized parameters
                test_days = (outer_test_end_date - outer_train_end_date).days
                test_result = self.backtest_engine.run_backtest(
                    controller_config=test_config,
                    market_data=outer_test_data,
                    strategies=strategies,
                    simulation_days=test_days,
                    name=f"{name}_outer_{i+1}_test"
                )
                
                results['test_performance'].append(test_result['performance_metrics'])
            
            # Store results for this outer window
            outer_window_info = {
                'window': i + 1,
                'train_start': outer_start_date.strftime('%Y-%m-%d'),
                'train_end': outer_train_end_date.strftime('%Y-%m-%d'),
                'test_start': outer_train_end_date.strftime('%Y-%m-%d'),
                'test_end': outer_test_end_date.strftime('%Y-%m-%d')
            }
            
            results['outer_windows'].append(outer_window_info)
            results['inner_results'].append(inner_results)
            results['outer_best_parameters'].append(best_outer_params)
        
        # Save nested results
        self._save_nested_results(results, name)
        
        return results
    
    def _calculate_cross_regime_statistics(self, regime_results, name):
        """Calculate statistics across different market regimes"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create directory for cross-regime analysis
        cross_regime_dir = os.path.join(self.output_dir, 'cross_regime')
        os.makedirs(cross_regime_dir, exist_ok=True)
        
        # Extract key performance metrics for each regime
        regimes = list(regime_results.keys())
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        
        # Create dataframes for performance comparison
        metric_data = {metric: [] for metric in metrics}
        regime_names = []
        
        for regime, results in regime_results.items():
            if 'metrics' in results:
                for metric in metrics:
                    if f'{metric}_mean' in results['metrics']:
                        metric_data[metric].append(results['metrics'][f'{metric}_mean'])
                    else:
                        metric_data[metric].append(np.nan)
                regime_names.append(regime)
        
        # Create performance comparison chart
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            if metric_data[metric]:
                bars = plt.bar(range(len(regime_names)), metric_data[metric])
                plt.xticks(range(len(regime_names)), regime_names, rotation=45, ha='right')
                plt.title(f'Average {metric.replace("_", " ").title()} by Market Regime')
                plt.ylabel(metric.replace("_", " ").title())
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(cross_regime_dir, f"{name}_cross_regime_performance.png"))
        plt.close()
        
        # Extract parameter stability across regimes
        # For each parameter, see how much it varies across regimes
        all_params = set()
        for regime, results in regime_results.items():
            if 'parameters' in results and results['parameters']:
                for param_set in results['parameters']:
                    all_params.update(param_set.keys())
        
        param_stability = {param: {'values': [], 'regimes': []} for param in all_params}
        
        for regime, results in regime_results.items():
            if 'parameters' in results and results['parameters']:
                # For each parameter, take the median value across windows
                for param in all_params:
                    param_values = [params.get(param, np.nan) for params in results['parameters'] if param in params]
                    if param_values:
                        median_value = np.nanmedian(param_values)
                        param_stability[param]['values'].append(median_value)
                        param_stability[param]['regimes'].append(regime)
        
        # Create parameter stability chart
        plt.figure(figsize=(14, 10))
        
        for i, param in enumerate(list(all_params)[:min(4, len(all_params))]):
            plt.subplot(2, 2, i+1)
            if param_stability[param]['values']:
                bars = plt.bar(range(len(param_stability[param]['regimes'])), param_stability[param]['values'])
                plt.xticks(range(len(param_stability[param]['regimes'])), param_stability[param]['regimes'], rotation=45, ha='right')
                plt.title(f'{param.replace("_", " ").title()} by Market Regime')
                plt.ylabel(param.replace("_", " ").title())
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(cross_regime_dir, f"{name}_cross_regime_parameters.png"))
        plt.close()
    
    def _save_nested_results(self, results, name):
        """Save nested walk-forward results"""
        import json
        
        # Create a JSON-serializable version of the results
        serializable_results = {
            'outer_windows': results['outer_windows'],
            'outer_best_parameters': results['outer_best_parameters']
        }
        
        # Save performance metrics for each outer window
        if results['test_performance']:
            serializable_results['test_performance'] = []
            for perf in results['test_performance']:
                # Convert numpy values to regular Python types
                perf_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                           for k, v in perf.items()}
                serializable_results['test_performance'].append(perf_dict)
        
        # Save nested results as JSON
        nested_dir = os.path.join(self.output_dir, 'nested')
        os.makedirs(nested_dir, exist_ok=True)
        
        results_file = os.path.join(nested_dir, f"{name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved nested walk-forward results to {results_file}")

# Test function (when run as script)
def test_walk_forward_runner():
    """Test the walk-forward runner with sample data"""
    # Create walk-forward runner
    runner = WalkForwardRunner()
    
    # Define parameter ranges
    parameter_ranges = {
        'snowball_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'atr_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0]
    }
    
    # Define base config
    base_config = {
        'max_allocation': 0.95,
        'initial_allocation': 0.7,
        'snowball_ratio': 0.5,
        'atr_multiplier': 2.0,
        'use_adaptive_sizing': True
    }
    
    # Run basic walk-forward test
    results = runner.run_basic_walk_forward(
        symbols=['SPY', 'QQQ', 'IWM'],
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    print("Basic walk-forward testing complete!")
    
    # Run multi-regime test
    regime_results = runner.run_multi_regime_walk_forward(
        symbols=['SPY', 'QQQ', 'IWM'],
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        regime_periods={
            'bull_market': (datetime(2023, 1, 1), datetime(2023, 7, 31)),
            'consolidation': (datetime(2023, 8, 1), datetime(2023, 10, 31)),
            'year_end_rally': (datetime(2023, 11, 1), datetime(2023, 12, 31))
        }
    )
    
    print("Multi-regime walk-forward testing complete!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_walk_forward_runner()
