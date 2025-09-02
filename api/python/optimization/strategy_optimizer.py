#!/usr/bin/env python3
"""
Strategy Optimizer Module

This module provides a framework for optimizing trading strategies through various methods,
including grid search, random search, Bayesian optimization, and genetic algorithms.
It integrates with the backtesting system to evaluate strategy performance.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
import itertools
import time
from rich.logging import RichHandler
from tqdm import tqdm

# Import the parameter optimizer
from trading_bot.optimization.parameter_optimizer import ParameterOptimizer, OptimizationMethod
from trading_bot.backtesting.unified_backtester import UnifiedBacktester

# Configure logging
logger = logging.getLogger(__name__)

class StrategyMetric(str, Enum):
    """Performance metrics that can be used for optimization."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CUSTOM = "custom"

class StrategyOptimizer:
    """
    Optimizes trading strategy parameters using backtesting for evaluation.
    
    Features:
    - Integrates with existing backtesting framework
    - Multiple optimization methods (grid search, Bayesian, genetic)
    - Parallel backtest evaluation
    - Cross-validation across different market regimes
    - Parameter sensitivity analysis
    - Result visualization
    """
    
    def __init__(
        self,
        strategy_class: Any,
        param_grid: Dict[str, List[Any]],
        scoring_function: Union[str, Callable] = 'sharpe_ratio',
        test_period: Tuple[str, str] = None,
        validation_period: Tuple[str, str] = None,
        initial_capital: float = 10000.0,
        n_jobs: int = 1,
        verbose: bool = False
    ):
        """
        Initialize the strategy optimizer.
        
        Args:
            strategy_class: The strategy class to optimize
            param_grid: Dictionary of parameters to optimize with lists of values to try
            scoring_function: Metric to evaluate performance ('sharpe_ratio', 'sortino_ratio', 
                             'total_return', 'calmar_ratio', or custom function)
            test_period: (start_date, end_date) for the backtest
            validation_period: Optional (start_date, end_date) for validation
            initial_capital: Starting capital for backtests
            n_jobs: Number of parallel jobs to run
            verbose: Whether to print progress information
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.test_period = test_period
        self.validation_period = validation_period
        self.initial_capital = initial_capital
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Set up scoring function
        self.scoring_function_name = scoring_function if isinstance(scoring_function, str) else 'custom'
        self.scoring_function = self._get_scoring_function(scoring_function)
        
        # Results storage
        self.results = []
        self.best_params = None
        self.best_score = None
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging for the optimizer."""
        level = logging.DEBUG if self.verbose else logging.INFO
        
        self.logger = logging.getLogger("StrategyOptimizer")
        self.logger.setLevel(level)
        
        # Check if handler already exists to avoid duplicates
        if not self.logger.handlers:
            handler = RichHandler(rich_tracebacks=True)
            handler.setLevel(level)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _get_scoring_function(self, scoring_function: Union[str, Callable]) -> Callable:
        """Get the appropriate scoring function."""
        if callable(scoring_function):
            return scoring_function
        
        scoring_functions = {
            'sharpe_ratio': lambda results: results.get('sharpe_ratio', -999),
            'sortino_ratio': lambda results: results.get('sortino_ratio', -999),
            'total_return': lambda results: results.get('total_return', -999),
            'calmar_ratio': lambda results: results.get('calmar_ratio', -999),
            'profit_factor': lambda results: results.get('profit_factor', -999),
            'win_rate': lambda results: results.get('win_rate', -999),
            'max_drawdown': lambda results: -results.get('max_drawdown', 999),  # Negative for minimization
        }
        
        if scoring_function in scoring_functions:
            return scoring_functions[scoring_function]
        else:
            self.logger.warning(f"Unknown scoring function '{scoring_function}', using sharpe_ratio instead")
            return scoring_functions['sharpe_ratio']
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from the parameter grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single parameter set with backtesting."""
        try:
            # Create strategy instance with the parameters
            strategy_instance = self.strategy_class(**params)
            
            # Create backtester
            backtester = UnifiedBacktester(
                strategies=[strategy_instance],
                initial_capital=self.initial_capital,
                start_date=self.test_period[0],
                end_date=self.test_period[1],
                debug_mode=False,
            )
            
            # Run backtest
            results = backtester.run_backtest()
            
            # Extract performance metrics
            performance = results.get('performance', {})
            
            # Calculate score
            score = self.scoring_function(performance)
            
            # Return results
            return {
                'params': params,
                'score': score,
                'performance': performance
            }
        except Exception as e:
            self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
            return {
                'params': params,
                'score': float('-inf'),
                'performance': {},
                'error': str(e)
            }
    
    def _validate_best_params(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the best parameters on the validation period."""
        if not self.validation_period:
            return {}
        
        try:
            # Create strategy instance with best parameters
            strategy_instance = self.strategy_class(**best_params)
            
            # Create backtester for validation period
            backtester = UnifiedBacktester(
                strategies=[strategy_instance],
                initial_capital=self.initial_capital,
                start_date=self.validation_period[0],
                end_date=self.validation_period[1],
                debug_mode=False,
            )
            
            # Run backtest on validation data
            results = backtester.run_backtest()
            
            # Extract performance metrics
            performance = results.get('performance', {})
            
            # Calculate score on validation set
            score = self.scoring_function(performance)
            
            return {
                'validation_score': score,
                'validation_performance': performance
            }
        except Exception as e:
            self.logger.error(f"Error validating best parameters: {str(e)}")
            return {
                'validation_score': float('-inf'),
                'validation_performance': {},
                'validation_error': str(e)
            }
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary containing optimization results including best parameters and performance.
        """
        self.logger.info("Starting parameter optimization")
        self.logger.info(f"Strategy: {self.strategy_class.__name__}")
        self.logger.info(f"Scoring function: {self.scoring_function_name}")
        self.logger.info(f"Parameter grid: {self.param_grid}")
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        total_combinations = len(param_combinations)
        
        self.logger.info(f"Total parameter combinations to evaluate: {total_combinations}")
        
        # Run evaluations
        if self.n_jobs > 1:
            # Parallel execution
            self.results = self._run_parallel_optimization(param_combinations)
        else:
            # Sequential execution
            self.results = self._run_sequential_optimization(param_combinations)
        
        # Sort results by score
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        # Get best parameters
        if self.results:
            self.best_params = self.results[0]['params']
            self.best_score = self.results[0]['score']
            self.best_performance = self.results[0]['performance']
            
            self.logger.info(f"Best score: {self.best_score}")
            self.logger.info(f"Best parameters: {self.best_params}")
            
            # Validate on validation period if specified
            if self.validation_period:
                validation_results = self._validate_best_params(self.best_params)
                validation_score = validation_results.get('validation_score')
                
                self.logger.info(f"Validation score: {validation_score}")
                
                # Add validation results to output
                self.results[0].update(validation_results)
        else:
            self.logger.warning("No valid results found during optimization")
        
        return self._create_optimization_report()
    
    def _run_sequential_optimization(self, param_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run optimization sequentially."""
        results = []
        
        for params in tqdm(param_combinations, desc="Evaluating parameters", disable=not self.verbose):
            result = self._evaluate_params(params)
            results.append(result)
        
        return results
    
    def _run_parallel_optimization(self, param_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run optimization in parallel using ProcessPoolExecutor."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(self._evaluate_params, params): params for params in param_combinations}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating parameters", disable=not self.verbose):
                result = future.result()
                results.append(result)
        
        return results
    
    def _create_optimization_report(self) -> Dict[str, Any]:
        """Create a comprehensive report of the optimization results."""
        top_n_results = self.results[:min(10, len(self.results))]
        
        report = {
            'strategy_name': self.strategy_class.__name__,
            'scoring_function': self.scoring_function_name,
            'parameter_grid': self.param_grid,
            'test_period': self.test_period,
            'validation_period': self.validation_period,
            'total_combinations_tested': len(self.results),
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'best_performance': self.best_performance,
            'top_results': top_n_results,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        
        # Add validation results if available
        if self.validation_period and self.results:
            report['validation_score'] = self.results[0].get('validation_score')
            report['validation_performance'] = self.results[0].get('validation_performance', {})
        
        return report
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to a JSON file.
        
        Args:
            filepath: Path to save the results file
        """
        report = self._create_optimization_report()
        
        # Convert any non-serializable objects to strings
        def json_serializable(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            else:
                return str(obj)
        
        # Process the report to ensure it's JSON serializable
        def process_dict(d):
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result[key] = process_dict(value)
                elif isinstance(value, list):
                    result[key] = [process_dict(item) if isinstance(item, dict) else
                                   json_serializable(item) for item in value]
                else:
                    result[key] = json_serializable(value)
            return result
        
        serializable_report = process_dict(report)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=4)
        
        self.logger.info(f"Optimization results saved to {filepath}")
    
    def plot_optimization_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization results to visualize parameter impact.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results:
                self.logger.warning("No results to plot")
                return
            
            # Convert results to DataFrame for easier plotting
            data = []
            for result in self.results:
                row = result['params'].copy()
                row['score'] = result['score']
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # For each parameter, create a subplot showing its impact on score
            param_names = list(self.param_grid.keys())
            n_params = len(param_names)
            
            cols = min(3, n_params)
            rows = (n_params + cols - 1) // cols
            
            for i, param in enumerate(param_names):
                plt.subplot(rows, cols, i+1)
                
                if df[param].dtype in (np.float64, np.int64):
                    # For numeric parameters, use a scatter plot
                    plt.scatter(df[param], df['score'], alpha=0.6)
                    plt.xlabel(param)
                    plt.ylabel('Score')
                else:
                    # For categorical parameters, use a box plot
                    sns.boxplot(x=param, y='score', data=df)
                
                plt.title(f'Impact of {param}')
                plt.tight_layout()
            
            plt.suptitle(f'Parameter Optimization Results for {self.strategy_class.__name__}', 
                         fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Plot saved to {save_path}")
            
            plt.show()
        except ImportError:
            self.logger.error("Matplotlib and/or seaborn are required for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")

    def walk_forward_optimization(self, 
                                 time_windows: List[Tuple[str, str]],
                                 window_size: int = 252,
                                 step_size: int = 63) -> Dict[str, Any]:
        """
        Perform walk-forward optimization using rolling time windows.
        
        Args:
            time_windows: List of (start_date, end_date) tuples for each optimization window
            window_size: Size of each optimization window in trading days
            step_size: Step size between windows in trading days
            
        Returns:
            Dictionary containing time-varying optimal parameters
        """
        self.logger.info("Starting walk-forward optimization")
        self.logger.info(f"Number of time windows: {len(time_windows)}")
        
        window_results = []
        
        for i, (start, end) in enumerate(time_windows):
            self.logger.info(f"Optimizing window {i+1}/{len(time_windows)}: {start} to {end}")
            
            # Set the current window as the test period
            self.test_period = (start, end)
            
            # Run optimization for this window
            results = self.optimize()
            
            # Store results for this window
            window_result = {
                'window_index': i,
                'start_date': start,
                'end_date': end,
                'best_params': self.best_params,
                'best_score': self.best_score
            }
            
            window_results.append(window_result)
        
        # Analyze parameter stability
        param_stability = self._analyze_parameter_stability(window_results)
        
        # Create comprehensive report
        wfo_report = {
            'strategy_name': self.strategy_class.__name__,
            'scoring_function': self.scoring_function_name,
            'window_size': window_size,
            'step_size': step_size,
            'window_results': window_results,
            'parameter_stability': param_stability
        }
        
        return wfo_report
    
    def _analyze_parameter_stability(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the stability of optimal parameters across time windows."""
        if not window_results:
            return {}
        
        # Extract parameters for each window
        param_series = {}
        
        # First, identify all parameters that appear in any window
        all_params = set()
        for result in window_results:
            all_params.update(result['best_params'].keys())
        
        # Initialize series for each parameter
        for param in all_params:
            param_series[param] = []
        
        # Fill in values for each window
        for result in window_results:
            for param in all_params:
                value = result['best_params'].get(param, None)
                param_series[param].append(value)
        
        # Calculate stability metrics
        stability_metrics = {}
        for param, values in param_series.items():
            # Convert to numpy array, handling non-numeric types
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v) if v is not None else np.nan)
                except (ValueError, TypeError):
                    numeric_values.append(np.nan)
            
            values_array = np.array(numeric_values)
            non_nan_values = values_array[~np.isnan(values_array)]
            
            # Skip metrics for parameters with no numeric values
            if len(non_nan_values) == 0:
                stability_metrics[param] = {
                    'type': 'non-numeric',
                    'unique_values': list(set([v for v in values if v is not None]))
                }
                continue
            
            # Calculate stability metrics
            stability_metrics[param] = {
                'mean': float(np.mean(non_nan_values)) if len(non_nan_values) > 0 else None,
                'std': float(np.std(non_nan_values)) if len(non_nan_values) > 0 else None,
                'min': float(np.min(non_nan_values)) if len(non_nan_values) > 0 else None,
                'max': float(np.max(non_nan_values)) if len(non_nan_values) > 0 else None,
                'coefficient_of_variation': float(np.std(non_nan_values) / np.mean(non_nan_values)) 
                                           if len(non_nan_values) > 0 and np.mean(non_nan_values) != 0 else None,
            }
        
        return stability_metrics


if __name__ == "__main__":
    # Example usage
    from trading_bot.strategy.moving_average_strategy import MovingAverageStrategy
    
    # Define parameter grid
    param_grid = {
        'short_window': [5, 10, 20],
        'long_window': [50, 100, 200],
        'symbol': ['SPY']
    }
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        strategy_class=MovingAverageStrategy,
        param_grid=param_grid,
        scoring_function='sharpe_ratio',
        test_period=('2020-01-01', '2022-01-01'),
        validation_period=('2022-01-01', '2022-12-31'),
        initial_capital=10000.0,
        n_jobs=1,
        verbose=True
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save results
    optimizer.save_results('optimization_results.json')
    
    # Plot results
    optimizer.plot_optimization_results('optimization_plot.png') 