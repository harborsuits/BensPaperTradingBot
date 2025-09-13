"""
Walk Forward Testing Module

This module implements walk-forward analysis for trading strategies to prevent overfitting.
It uses rolling windows of training and testing data to validate strategy parameters.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class WalkForwardTester:
    """
    Performs walk-forward testing to validate trading strategy parameters.
    
    Walk-forward testing uses rolling windows of data: train on a historical window,
    test on the subsequent out-of-sample window, then roll forward.
    This prevents overfitting by ensuring the strategy works on unseen data.
    """
    
    def __init__(self, backtest_engine, output_dir: str = './results/walk_forward'):
        """
        Initialize the walk-forward tester.
        
        Args:
            backtest_engine: Instance of the backtesting engine
            output_dir: Directory to save results
        """
        self.backtest_engine = backtest_engine
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Store test results
        self.results = {
            'windows': [],
            'training_performance': [],
            'test_performance': [],
            'parameters': [],
            'metrics': {}
        }
    
    def run_walk_forward_analysis(self,
                               market_data: Dict[str, pd.DataFrame],
                               parameter_ranges: Dict[str, List[Any]],
                               base_config: Dict[str, Any],
                               strategies: Dict[str, Dict[str, Any]],
                               train_window_size: int = 252,  # Default 1 year
                               test_window_size: int = 63,    # Default ~3 months
                               step_size: int = 21,           # Default ~1 month
                               optimization_metric: str = 'sharpe_ratio',
                               name: str = 'walk_forward') -> Dict[str, Any]:
        """
        Run walk-forward analysis.
        
        Args:
            market_data: Dict mapping symbols to price DataFrames
            parameter_ranges: Dict of parameter names to lists of values to test
            base_config: Base configuration for the strategy controller
            strategies: Dict of strategies to use
            train_window_size: Number of days in training window
            test_window_size: Number of days in testing window
            step_size: Number of days to move forward between tests
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            name: Name for this test run
            
        Returns:
            Dict of results
        """
        # Validate that we have enough data
        min_required_days = train_window_size + test_window_size
        
        # Check all symbols have enough data
        for symbol, df in market_data.items():
            if len(df) < min_required_days:
                logger.error(f"Not enough data for {symbol}. Need at least {min_required_days} days, but got {len(df)}.")
                return {'error': f"Not enough data for {symbol}."}
        
        # Get the date range from the first symbol
        symbol = list(market_data.keys())[0]
        dates = market_data[symbol]['date'].sort_values().to_list()
        
        # Calculate number of windows
        total_days = len(dates)
        num_windows = max(1, (total_days - min_required_days) // step_size + 1)
        
        logger.info(f"Running walk-forward analysis with {num_windows} windows")
        logger.info(f"Training window: {train_window_size} days, Testing window: {test_window_size} days")
        
        # Reset results
        self.results = {
            'windows': [],
            'training_performance': [],
            'test_performance': [],
            'parameters': [],
            'metrics': {}
        }
        
        # Run the walk-forward analysis
        for i in tqdm(range(num_windows), desc="Walk-Forward Windows"):
            start_idx = i * step_size
            train_end_idx = start_idx + train_window_size
            test_end_idx = min(train_end_idx + test_window_size, total_days)
            
            # Get dates for this window
            train_start_date = dates[start_idx]
            train_end_date = dates[train_end_idx - 1]
            test_start_date = dates[train_end_idx]
            test_end_date = dates[test_end_idx - 1] if test_end_idx - 1 < len(dates) else dates[-1]
            
            logger.info(f"Window {i+1}/{num_windows}:")
            logger.info(f"  Training: {train_start_date} to {train_end_date}")
            logger.info(f"  Testing:  {test_start_date} to {test_end_date}")
            
            # Create training data subset
            train_data = {}
            for symbol, df in market_data.items():
                train_df = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)].copy()
                train_data[symbol] = train_df
            
            # Create testing data subset
            test_data = {}
            for symbol, df in market_data.items():
                test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()
                test_data[symbol] = test_df
            
            # Get window duration in days
            train_days = (pd.Timestamp(train_end_date) - pd.Timestamp(train_start_date)).days + 1
            test_days = (pd.Timestamp(test_end_date) - pd.Timestamp(test_start_date)).days + 1
            
            # Find optimal parameters on training data
            optimization_result = self._optimize_on_window(
                train_data, 
                parameter_ranges, 
                base_config, 
                strategies, 
                simulation_days=train_days,
                optimization_metric=optimization_metric
            )
            
            # Use best parameters on test data
            best_params = optimization_result['best_parameters']
            test_config = base_config.copy()
            for param, value in best_params.items():
                test_config[param] = value
            
            # Run backtest on test data with optimized parameters
            test_result = self.backtest_engine.run_backtest(
                controller_config=test_config,
                market_data=test_data,
                strategies=strategies,
                simulation_days=test_days,
                name=f"{name}_window_{i+1}_test"
            )
            
            # Store results for this window
            window_info = {
                'window': i + 1,
                'train_start': train_start_date,
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date
            }
            
            self.results['windows'].append(window_info)
            self.results['training_performance'].append(optimization_result['best_performance'])
            self.results['test_performance'].append(test_result['performance_metrics'])
            self.results['parameters'].append(best_params)
        
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()
        
        # Save results
        self._save_results(name)
        
        # Generate visualizations
        self._generate_visualizations(name)
        
        return self.results
    
    def _optimize_on_window(self,
                          market_data: Dict[str, pd.DataFrame],
                          parameter_ranges: Dict[str, List[Any]],
                          base_config: Dict[str, Any],
                          strategies: Dict[str, Dict[str, Any]],
                          simulation_days: int = 252,
                          optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Find optimal parameters for a single training window.
        
        Args:
            market_data: Market data for this window
            parameter_ranges: Parameter ranges to test
            base_config: Base configuration
            strategies: Strategies to use
            simulation_days: Number of days to simulate
            optimization_metric: Metric to optimize
            
        Returns:
            Dict with best parameters and performance
        """
        # Get optimization engine
        from trading_bot.backtest.optimization_engine import OptimizationEngine
        
        # Convert discrete parameter ranges to min/max format for optimizer
        param_range_tuples = {}
        for param, values in parameter_ranges.items():
            # If already a tuple of (min, max), use it directly
            if isinstance(values, tuple) and len(values) == 2:
                param_range_tuples[param] = values
            elif isinstance(values, list) and len(values) > 0:
                # Otherwise convert list to min/max range
                param_range_tuples[param] = (min(values), max(values))
        
        # Run optimization
        optimizer = OptimizationEngine(metric=optimization_metric)
        result = optimizer.optimize(
            parameter_ranges=param_range_tuples,
            base_config=base_config,
            market_data=market_data,
            strategies=strategies,
            simulation_days=simulation_days,
            n_trials=20  # Limit trials for faster walk-forward testing
        )
        
        return {
            'best_parameters': result['best_params'],
            'best_performance': result['best_performance']
        }
    
    def _calculate_aggregate_metrics(self) -> None:
        """Calculate aggregate metrics across all windows"""
        if not self.results['test_performance']:
            return
        
        # Extract key metrics from test performance
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Initialize aggregated metrics
        for metric in metrics:
            self.results['metrics'][metric] = []
            
            # Extract metric values from all test windows
            for test_perf in self.results['test_performance']:
                if metric in test_perf:
                    self.results['metrics'][metric].append(test_perf[metric])
            
            # Calculate statistics
            metric_values = self.results['metrics'][metric]
            if metric_values:
                self.results['metrics'][f'{metric}_mean'] = np.mean(metric_values)
                self.results['metrics'][f'{metric}_median'] = np.median(metric_values)
                self.results['metrics'][f'{metric}_std'] = np.std(metric_values)
                self.results['metrics'][f'{metric}_min'] = min(metric_values)
                self.results['metrics'][f'{metric}_max'] = max(metric_values)
        
        # Calculate parameter stability
        param_stability = {}
        for param in self.results['parameters'][0].keys():
            param_values = [params[param] for params in self.results['parameters']]
            param_stability[param] = {
                'mean': np.mean(param_values),
                'median': np.median(param_values),
                'std': np.std(param_values),
                'min': min(param_values),
                'max': max(param_values),
                'variation_coeff': np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else float('inf')
            }
        
        self.results['parameter_stability'] = param_stability
        
        # Calculate in-sample vs out-of-sample performance difference
        in_vs_out = {}
        for metric in metrics:
            in_values = []
            out_values = []
            
            for i in range(len(self.results['training_performance'])):
                if metric in self.results['training_performance'][i] and metric in self.results['test_performance'][i]:
                    in_values.append(self.results['training_performance'][i][metric])
                    out_values.append(self.results['test_performance'][i][metric])
            
            if in_values and out_values:
                in_mean = np.mean(in_values)
                out_mean = np.mean(out_values)
                
                in_vs_out[metric] = {
                    'in_sample_mean': in_mean,
                    'out_of_sample_mean': out_mean,
                    'difference': out_mean - in_mean,
                    'percent_drop': ((in_mean - out_mean) / in_mean) * 100 if in_mean != 0 else float('inf')
                }
        
        self.results['in_vs_out_sample'] = in_vs_out
    
    def _save_results(self, name: str) -> None:
        """Save walk-forward results to disk"""
        # Create a JSON-serializable version of the results
        serializable_results = self.results.copy()
        
        # Convert any non-serializable objects
        for i, window in enumerate(serializable_results['windows']):
            for key, value in window.items():
                if isinstance(value, (datetime, pd.Timestamp)):
                    serializable_results['windows'][i][key] = value.isoformat()
        
        # Save results as JSON
        results_file = os.path.join(self.output_dir, f"{name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved walk-forward results to {results_file}")
    
    def _generate_visualizations(self, name: str) -> None:
        """Generate visualizations of walk-forward results"""
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Extract metrics
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        windows = [w['window'] for w in self.results['windows']]
        
        # 1. Create performance comparison across windows
        plt.figure(figsize=(12, 8))
        
        for metric in metrics:
            in_sample = [train_perf.get(metric, 0) for train_perf in self.results['training_performance']]
            out_sample = [test_perf.get(metric, 0) for test_perf in self.results['test_performance']]
            
            plt.subplot(2, 2, metrics.index(metric) + 1)
            plt.plot(windows, in_sample, 'b-', label=f'In-sample {metric}')
            plt.plot(windows, out_sample, 'r-', label=f'Out-of-sample {metric}')
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xlabel('Window Number')
            plt.ylabel(metric.replace("_", " ").title())
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_metrics_comparison.png"))
        plt.close()
        
        # 2. Create parameter evolution chart
        params = list(self.results['parameters'][0].keys())
        
        plt.figure(figsize=(12, 8))
        
        for i, param in enumerate(params[:min(4, len(params))]):
            values = [params_dict[param] for params_dict in self.results['parameters']]
            
            plt.subplot(2, 2, i + 1)
            plt.plot(windows, values, 'g-', marker='o')
            plt.title(f'{param.replace("_", " ").title()} Evolution')
            plt.xlabel('Window Number')
            plt.ylabel(param.replace("_", " ").title())
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_parameter_evolution.png"))
        plt.close()
        
        # 3. In-sample vs. Out-of-sample performance
        plt.figure(figsize=(10, 6))
        
        for metric in metrics:
            in_values = []
            out_values = []
            
            for i in range(len(self.results['training_performance'])):
                if metric in self.results['training_performance'][i] and metric in self.results['test_performance'][i]:
                    in_values.append(self.results['training_performance'][i][metric])
                    out_values.append(self.results['test_performance'][i][metric])
            
            if in_values and out_values:
                x_pos = metrics.index(metric)
                in_mean = np.mean(in_values)
                out_mean = np.mean(out_values)
                
                plt.bar(x_pos - 0.2, in_mean, width=0.4, label='In-sample' if x_pos == 0 else "")
                plt.bar(x_pos + 0.2, out_mean, width=0.4, label='Out-of-sample' if x_pos == 0 else "")
        
        plt.xticks(range(len(metrics)), [m.replace("_", " ").title() for m in metrics])
        plt.title('In-sample vs. Out-of-sample Performance')
        plt.ylabel('Average Value')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f"{name}_in_vs_out.png"))
        plt.close()

# Test function (when run as script)
def test_walk_forward_tester():
    """Test the walk-forward tester with sample data"""
    from trading_bot.backtest.adaptive_backtest_engine import AdaptiveBacktestEngine
    from trading_bot.backtest.market_data_generator import MarketDataGenerator
    
    # Create sample data
    data_gen = MarketDataGenerator()
    market_data = data_gen.generate_multi_regime_data(
        symbols=['SPY', 'QQQ', 'IWM'],
        days=756,  # 3 years
        seed=42
    )
    
    # Create strategies config
    strategies = {
        'trend_following': {
            'name': 'Trend Following Strategy',
            'parameters': {
                'ma_fast': 20,
                'ma_slow': 50
            }
        },
        'mean_reversion': {
            'name': 'Mean Reversion Strategy',
            'parameters': {
                'lookback': 20,
                'threshold': 2.0
            }
        }
    }
    
    # Create backtest engine
    backtest_engine = AdaptiveBacktestEngine()
    
    # Create walk-forward tester
    wf_tester = WalkForwardTester(backtest_engine)
    
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
    
    # Run walk-forward analysis
    results = wf_tester.run_walk_forward_analysis(
        market_data=market_data,
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        strategies=strategies,
        train_window_size=252,  # 1 year
        test_window_size=63,    # ~3 months
        step_size=21            # ~1 month step
    )
    
    print("Walk-forward testing complete!")
    print(f"Number of windows tested: {len(results['windows'])}")
    print("Performance metrics:")
    for metric, value in results['metrics'].items():
        if not isinstance(value, list):
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_walk_forward_tester()
