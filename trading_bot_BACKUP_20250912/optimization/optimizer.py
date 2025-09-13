import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional, Callable, Iterator
import logging
import time
import os
import json
from datetime import datetime
from enum import Enum
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .parameter_space import ParameterSpace
from .search_methods import SearchMethod, GridSearch, RandomSearch, BayesianOptimization, GeneticAlgorithm

# Import trading components if in same package
try:
    from ..backtesting.order_book_simulator import OrderBookSimulator, MarketRegime, MarketRegimeDetector
except ImportError:
    # Handle case where imports might be different
    try:
        from trading_bot.backtesting.order_book_simulator import OrderBookSimulator, MarketRegime, MarketRegimeDetector
    except ImportError:
        pass

logger = logging.getLogger(__name__)

class OptimizationMetric(Enum):
    """Metrics for parameter optimization"""
    TOTAL_RETURN = "total_return"           # Total return
    ANNUALIZED_RETURN = "annualized_return" # Annualized return
    SHARPE_RATIO = "sharpe_ratio"           # Sharpe ratio
    SORTINO_RATIO = "sortino_ratio"         # Sortino ratio
    CALMAR_RATIO = "calmar_ratio"           # Calmar ratio
    MAX_DRAWDOWN = "max_drawdown"           # Maximum drawdown
    VOLATILITY = "volatility"               # Volatility
    WIN_RATE = "win_rate"                   # Win rate
    PROFIT_FACTOR = "profit_factor"         # Profit factor
    REGIME_STABILITY = "regime_stability"   # Stability across regimes
    CUSTOM = "custom"                       # Custom metric

class RegimeWeight(Enum):
    """Weight strategies for different market regimes"""
    EQUAL = "equal"                        # Equal weight for all regimes
    DURATION = "duration"                   # Weight by regime duration
    CUSTOM = "custom"                       # Custom weights

class WalkForwardMethod(Enum):
    """Methods for walk-forward optimization"""
    ANCHORED = "anchored"                   # Fixed start date, expanding window
    ROLLING = "rolling"                     # Moving window
    EXPANDING = "expanding"                 # Expanding window with fixed test size

class ParameterOptimizer:
    """
    Framework for parameter optimization across market regimes
    
    Features:
    - Optimizes parameters across different market regimes
    - Supports multiple search methods (grid, random, Bayesian, genetic)
    - Walk-forward optimization to prevent overfitting
    - Parallel processing for faster optimization
    - Comprehensive reporting and visualization
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        search_method: SearchMethod,
        objective_metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO,
        regime_weights: Dict[MarketRegime, float] = None,
        weight_strategy: RegimeWeight = RegimeWeight.EQUAL,
        is_maximizing: bool = True,
        use_walk_forward: bool = False,
        walk_forward_method: WalkForwardMethod = WalkForwardMethod.ROLLING,
        train_size: int = 200,
        test_size: int = 50,
        n_workers: int = None,
        output_dir: str = "optimization_results",
        verbose: bool = True
    ):
        """
        Initialize parameter optimizer
        
        Args:
            parameter_space: Space of parameters to optimize
            search_method: Method for searching parameter space
            objective_metric: Metric to optimize
            regime_weights: Weights for different market regimes
            weight_strategy: Strategy for weighting different regimes
            is_maximizing: Whether to maximize (True) or minimize (False) the objective
            use_walk_forward: Whether to use walk-forward optimization
            walk_forward_method: Method for walk-forward optimization
            train_size: Size of training set in bars
            test_size: Size of test set in bars
            n_workers: Number of parallel workers (default: number of CPU cores - 1)
            output_dir: Directory for saving results
            verbose: Whether to print detailed progress
        """
        self.parameter_space = parameter_space
        self.search_method = search_method
        self.objective_metric = objective_metric
        self.weight_strategy = weight_strategy
        self.use_walk_forward = use_walk_forward
        self.walk_forward_method = walk_forward_method
        self.train_size = train_size
        self.test_size = test_size
        self.verbose = verbose
        self.output_dir = output_dir
        
        # Set objective maximization/minimization
        if is_maximizing:
            self.search_method.set_max_objective()
        else:
            self.search_method.set_min_objective()
        
        # Set number of workers
        if n_workers is None:
            self.n_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_workers = max(1, n_workers)
        
        # Set up regime weights
        self.regime_weights = regime_weights or {}
        
        # Initialize results storage
        self.results = []
        self.best_params = None
        self.best_objective = float('-inf') if is_maximizing else float('inf')
        self.is_maximizing = is_maximizing
        
        # Initialize regime tracking
        self.regime_detector = None
        self.regime_segments = {}
        self.regime_performance = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized ParameterOptimizer with {parameter_space}")
    
    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None:
        """
        Set market regime detector for regime-aware optimization
        
        Args:
            regime_detector: Market regime detector
        """
        self.regime_detector = regime_detector
    
    def segment_data_by_regime(self, prices: pd.Series) -> Dict[MarketRegime, List[Tuple[int, int]]]:
        """
        Segment historical data by market regime
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary of regime -> list of (start, end) indices
        """
        if self.regime_detector is None:
            logger.warning("No regime detector set, using single regime")
            return {MarketRegime.UNKNOWN: [(0, len(prices) - 1)]}
        
        # Reset regime detector
        self.regime_detector.reset()
        
        # Detect regimes for each bar
        regimes = []
        for i, price in enumerate(prices):
            if i > 0:
                returns = (price / prices[i-1]) - 1
            else:
                returns = 0
            
            regime = self.regime_detector.update(price, returns)
            regimes.append(regime)
        
        # Find contiguous segments of the same regime
        segments = {}
        current_regime = regimes[0]
        start_idx = 0
        
        for i, regime in enumerate(regimes):
            if regime != current_regime or i == len(regimes) - 1:
                # End of a segment (or last bar)
                end_idx = i - 1 if regime != current_regime else i
                
                if current_regime not in segments:
                    segments[current_regime] = []
                
                # Only add segment if it's long enough
                if end_idx - start_idx + 1 >= 20:  # Minimum 20 bars per segment
                    segments[current_regime].append((start_idx, end_idx))
                
                # Start new segment
                current_regime = regime
                start_idx = i
        
        # Compute weights if using DURATION strategy
        if self.weight_strategy == RegimeWeight.DURATION:
            total_bars = len(prices)
            for regime, segments_list in segments.items():
                regime_bars = sum(end - start + 1 for start, end in segments_list)
                self.regime_weights[regime] = regime_bars / total_bars
        
        # Log detected regimes
        for regime, segments_list in segments.items():
            total_bars = sum(end - start + 1 for start, end in segments_list)
            logger.info(f"Regime {regime.name}: {len(segments_list)} segments, {total_bars} bars total")
        
        self.regime_segments = segments
        return segments
    
    def create_walk_forward_windows(
        self, 
        data_length: int
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create walk-forward windows for optimization
        
        Args:
            data_length: Length of the data
            
        Returns:
            List of (train_indices, test_indices) for each window
        """
        windows = []
        
        if not self.use_walk_forward:
            # Use all data for training if not using walk-forward
            all_indices = list(range(data_length))
            windows.append((all_indices, []))
            return windows
        
        if self.walk_forward_method == WalkForwardMethod.ANCHORED:
            # Anchored walk-forward: fixed start date, expanding window
            start = 0
            while start + self.train_size + self.test_size <= data_length:
                train_end = start + self.train_size
                test_end = min(data_length, train_end + self.test_size)
                
                train_indices = list(range(start, train_end))
                test_indices = list(range(train_end, test_end))
                
                windows.append((train_indices, test_indices))
                
                # Move forward by test size
                start += self.test_size
        
        elif self.walk_forward_method == WalkForwardMethod.ROLLING:
            # Rolling walk-forward: moving window
            start = 0
            while start + self.train_size + self.test_size <= data_length:
                train_end = start + self.train_size
                test_end = min(data_length, train_end + self.test_size)
                
                train_indices = list(range(start, train_end))
                test_indices = list(range(train_end, test_end))
                
                windows.append((train_indices, test_indices))
                
                # Move forward by test size
                start += self.test_size
        
        elif self.walk_forward_method == WalkForwardMethod.EXPANDING:
            # Expanding walk-forward: expanding window with fixed test size
            start = 0
            min_train_size = self.train_size
            
            while start + min_train_size + self.test_size <= data_length:
                train_end = data_length - self.test_size
                test_end = data_length
                
                train_indices = list(range(start, train_end))
                test_indices = list(range(train_end, test_end))
                
                windows.append((train_indices, test_indices))
                
                # Move forward by test size
                start += self.test_size
        
        return windows
    
    def optimize(
        self, 
        strategy_evaluator: Callable[[Dict[str, Any], List[int]], Dict[str, float]],
        ohlcv_data: pd.DataFrame = None,
        prices: pd.Series = None,
        max_evaluations: int = 100,
        timeout_seconds: int = None
    ) -> Dict[str, Any]:
        """
        Run parameter optimization
        
        Args:
            strategy_evaluator: Function that evaluates a strategy with given parameters
                                Args: parameters, indices to use
                                Returns: Dict of performance metrics
            ohlcv_data: OHLCV data for optimization (optional)
            prices: Price series for optimization (optional)
            max_evaluations: Maximum number of evaluations
            timeout_seconds: Timeout in seconds
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Extract prices from OHLCV data if not provided
        if prices is None and ohlcv_data is not None:
            if 'close' in ohlcv_data.columns:
                prices = ohlcv_data['close']
            else:
                raise ValueError("OHLCV data must contain 'close' column or prices must be provided")
        
        if prices is None:
            raise ValueError("Either OHLCV data or prices must be provided")
        
        # Segment data by regime if regime detector is available
        if self.regime_detector is not None:
            self.segment_data_by_regime(prices)
        
        # Create walk-forward windows
        windows = self.create_walk_forward_windows(len(prices))
        
        logger.info(f"Created {len(windows)} walk-forward windows")
        
        # Set up parallel processing
        n_evaluations = 0
        eval_results = []
        
        # Main optimization loop
        while n_evaluations < max_evaluations:
            # Check timeout
            if timeout_seconds and time.time() - start_time > timeout_seconds:
                logger.info(f"Optimization timed out after {timeout_seconds} seconds")
                break
            
            # Get next parameter set to evaluate
            params = self.search_method.suggest()
            
            # If search method has no more suggestions, we're done
            if hasattr(self.search_method, 'has_next') and not self.search_method.has_next:
                logger.info("Search method has no more suggestions, optimization complete")
                break
            
            # Evaluate parameters on all windows
            window_results = []
            
            # Use parallel processing for window evaluation
            futures = []
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                for window_idx, (train_indices, test_indices) in enumerate(windows):
                    future = executor.submit(
                        self._evaluate_window,
                        strategy_evaluator=strategy_evaluator,
                        params=params,
                        train_indices=train_indices,
                        test_indices=test_indices,
                        window_idx=window_idx
                    )
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    window_result = future.result()
                    window_results.append(window_result)
            
            # Calculate aggregated performance across windows
            aggregated_result = self._aggregate_window_results(window_results)
            
            # Store the evaluation result
            eval_result = {
                'params': params,
                'windows': window_results,
                'aggregated': aggregated_result,
                'evaluation_number': n_evaluations,
                'timestamp': datetime.now().isoformat()
            }
            eval_results.append(eval_result)
            
            # Register result with search method
            objective_value = aggregated_result.get(self.objective_metric.value, 0)
            self.search_method.register_result(params, objective_value)
            
            # Update best result
            is_better = ((self.is_maximizing and objective_value > self.best_objective) or 
                         (not self.is_maximizing and objective_value < self.best_objective))
            
            if is_better:
                self.best_objective = objective_value
                self.best_params = params.copy()
                logger.info(f"New best parameters found: {self.best_params}")
                logger.info(f"New best objective: {self.best_objective}")
            
            # Log progress
            if self.verbose and n_evaluations % 10 == 0:
                logger.info(f"Completed {n_evaluations} evaluations out of {max_evaluations}")
                logger.info(f"Best objective so far: {self.best_objective}")
            
            n_evaluations += 1
        
        # Optimization complete, save results
        optimization_result = {
            'best_params': self.best_params,
            'best_objective': self.best_objective,
            'evaluations': eval_results,
            'n_evaluations': n_evaluations,
            'duration_seconds': time.time() - start_time,
            'parameter_space': str(self.parameter_space),
            'search_method': self.search_method.__class__.__name__,
            'objective_metric': self.objective_metric.value,
            'is_maximizing': self.is_maximizing,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(optimization_result)
        
        # Generate report
        self._generate_report(optimization_result)
        
        return optimization_result
    
    def _evaluate_window(
        self,
        strategy_evaluator: Callable,
        params: Dict[str, Any],
        train_indices: List[int],
        test_indices: List[int],
        window_idx: int
    ) -> Dict[str, Any]:
        """
        Evaluate parameters on a single window
        
        Args:
            strategy_evaluator: Strategy evaluation function
            params: Parameters to evaluate
            train_indices: Training indices
            test_indices: Testing indices
            window_idx: Window index
            
        Returns:
            Window evaluation result
        """
        # Evaluate on training set
        train_result = strategy_evaluator(params, train_indices)
        
        # Evaluate on test set if available
        test_result = {}
        if test_indices:
            test_result = strategy_evaluator(params, test_indices)
        
        # Evaluate on each regime if available
        regime_results = {}
        if self.regime_segments:
            for regime, segments in self.regime_segments.items():
                regime_indices = []
                for start, end in segments:
                    regime_indices.extend(range(start, end + 1))
                
                # Only evaluate if there are enough data points
                if len(regime_indices) >= 20:
                    regime_result = strategy_evaluator(params, regime_indices)
                    regime_results[regime.name] = regime_result
        
        # Create window result
        window_result = {
            'window_idx': window_idx,
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'train_result': train_result,
            'test_result': test_result,
            'regime_results': regime_results
        }
        
        return window_result
    
    def _aggregate_window_results(self, window_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate results across windows and regimes
        
        Args:
            window_results: List of window evaluation results
            
        Returns:
            Aggregated performance metrics
        """
        # Initialize aggregated metrics
        aggregated = {}
        
        # Aggregate in-sample (training) results
        train_metrics = {}
        for window_result in window_results:
            train_result = window_result['train_result']
            for metric, value in train_result.items():
                if metric not in train_metrics:
                    train_metrics[metric] = []
                train_metrics[metric].append(value)
        
        # Calculate mean and std for each in-sample metric
        for metric, values in train_metrics.items():
            aggregated[f'train_{metric}_mean'] = np.mean(values)
            aggregated[f'train_{metric}_std'] = np.std(values)
        
        # Aggregate out-of-sample (test) results if available
        test_metrics = {}
        for window_result in window_results:
            test_result = window_result.get('test_result', {})
            if not test_result:
                continue
                
            for metric, value in test_result.items():
                if metric not in test_metrics:
                    test_metrics[metric] = []
                test_metrics[metric].append(value)
        
        # Calculate mean and std for each out-of-sample metric
        for metric, values in test_metrics.items():
            if values:
                aggregated[f'test_{metric}_mean'] = np.mean(values)
                aggregated[f'test_{metric}_std'] = np.std(values)
                
                # Calculate stability (ratio of test to train)
                if f'train_{metric}_mean' in aggregated and aggregated[f'train_{metric}_mean'] != 0:
                    test_mean = aggregated[f'test_{metric}_mean']
                    train_mean = aggregated[f'train_{metric}_mean']
                    stability = test_mean / train_mean
                    aggregated[f'{metric}_stability'] = stability
        
        # Aggregate regime results if available
        regime_metrics = {}
        for window_result in window_results:
            regime_results = window_result.get('regime_results', {})
            for regime, result in regime_results.items():
                if regime not in regime_metrics:
                    regime_metrics[regime] = {}
                
                for metric, value in result.items():
                    if metric not in regime_metrics[regime]:
                        regime_metrics[regime][metric] = []
                    regime_metrics[regime][metric].append(value)
        
        # Calculate mean for each regime metric
        for regime, metrics in regime_metrics.items():
            for metric, values in metrics.items():
                aggregated[f'{regime}_{metric}_mean'] = np.mean(values)
        
        # Calculate regime stability if objective metric is available
        objective = self.objective_metric.value
        regime_stability = []
        
        if regime_metrics:
            # Get objective values for each regime
            regime_objectives = {}
            for regime, metrics in regime_metrics.items():
                if objective in metrics:
                    regime_objectives[regime] = np.mean(metrics[objective])
            
            # Calculate coefficient of variation across regimes
            if regime_objectives:
                regime_values = list(regime_objectives.values())
                mean_objective = np.mean(regime_values)
                std_objective = np.std(regime_values)
                
                # Lower CV = more stable across regimes
                if mean_objective != 0:
                    aggregated['regime_cv'] = std_objective / abs(mean_objective)
                    aggregated['regime_stability'] = 1.0 / (1.0 + aggregated['regime_cv'])
        
        # Set the primary objective based on the specified metric
        if self.objective_metric == OptimizationMetric.SHARPE_RATIO:
            if 'test_sharpe_ratio_mean' in aggregated:
                aggregated[objective] = aggregated['test_sharpe_ratio_mean']
            else:
                aggregated[objective] = aggregated.get('train_sharpe_ratio_mean', 0)
            
            # Consider regime stability in objective
            if 'regime_stability' in aggregated:
                # Weighted combination of performance and stability
                stability_weight = 0.3  # 30% weight on regime stability
                performance_weight = 1.0 - stability_weight
                
                raw_objective = aggregated[objective]
                stability = aggregated['regime_stability']
                
                aggregated[objective] = (performance_weight * raw_objective + 
                                       stability_weight * raw_objective * stability)
        
        # For other metrics, use mean of test results if available, otherwise train
        else:
            if f'test_{objective}_mean' in aggregated:
                aggregated[objective] = aggregated[f'test_{objective}_mean']
            else:
                aggregated[objective] = aggregated.get(f'train_{objective}_mean', 0)
        
        return aggregated
    
    def _save_results(self, optimization_result: Dict[str, Any]) -> None:
        """
        Save optimization results to file
        
        Args:
            optimization_result: Optimization results
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.output_dir, f"optimization_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save summary results
        summary = {
            'best_params': optimization_result['best_params'],
            'best_objective': optimization_result['best_objective'],
            'n_evaluations': optimization_result['n_evaluations'],
            'duration_seconds': optimization_result['duration_seconds'],
            'objective_metric': optimization_result['objective_metric'],
            'timestamp': optimization_result['timestamp']
        }
        
        with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save all evaluation results (can be large)
        with open(os.path.join(result_dir, 'evaluations.json'), 'w') as f:
            json.dump(optimization_result['evaluations'], f, indent=2, default=str)
        
        # Save parameters as CSV for easier viewing
        results_df = self.search_method.get_results_df()
        if not results_df.empty:
            results_df.to_csv(os.path.join(result_dir, 'parameter_results.csv'), index=False)
        
        logger.info(f"Optimization results saved to {result_dir}")
    
    def _generate_report(self, optimization_result: Dict[str, Any]) -> None:
        """
        Generate report with visualizations
        
        Args:
            optimization_result: Optimization results
        """
        # Create visualizations if matplotlib is available
        try:
            # Get results dataframe
            results_df = self.search_method.get_results_df()
            if results_df.empty:
                return
            
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(self.output_dir, f"optimization_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)
            
            # Plot objective values by evaluation number
            plt.figure(figsize=(10, 6))
            plt.plot(results_df.index, results_df['objective'])
            plt.title('Objective Value by Evaluation')
            plt.xlabel('Evaluation Number')
            plt.ylabel(self.objective_metric.value)
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'objective_by_evaluation.png'))
            plt.close()
            
            # Plot parameter distributions for top results
            top_n = min(20, len(results_df))
            top_results = results_df.head(top_n)
            
            parameter_columns = [col for col in top_results.columns if col != 'objective']
            
            # Create a separate plot for each parameter
            for param in parameter_columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(top_results[param], top_results['objective'])
                plt.title(f'{param} vs. {self.objective_metric.value}')
                plt.xlabel(param)
                plt.ylabel(self.objective_metric.value)
                plt.grid(True)
                plt.savefig(os.path.join(result_dir, f'{param}_vs_objective.png'))
                plt.close()
            
            # If there are only 2-3 parameters, create a pairs plot
            if 2 <= len(parameter_columns) <= 3:
                # Create scatter matrix
                from pandas.plotting import scatter_matrix
                scatter_matrix(top_results[parameter_columns + ['objective']], 
                              figsize=(12, 12), alpha=0.5, diagonal='kde')
                plt.savefig(os.path.join(result_dir, 'parameter_scatter_matrix.png'))
                plt.close()
            
            logger.info(f"Optimization report generated in {result_dir}")
        
        except Exception as e:
            logger.warning(f"Error generating report: {e}")
    
    def plot_walk_forward_results(self, optimization_result: Dict[str, Any]) -> None:
        """
        Plot walk-forward optimization results
        
        Args:
            optimization_result: Optimization results
        """
        try:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(self.output_dir, f"walkforward_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)
            
            # Extract best result
            best_params = optimization_result['best_params']
            best_result = None
            
            for eval_result in optimization_result['evaluations']:
                if eval_result['params'] == best_params:
                    best_result = eval_result
                    break
            
            if not best_result:
                return
            
            # Extract window results
            window_results = best_result['windows']
            
            # Extract metrics for each window
            window_indices = []
            train_metrics = []
            test_metrics = []
            
            objective = self.objective_metric.value
            
            for window in window_results:
                window_idx = window['window_idx']
                window_indices.append(window_idx)
                
                train_result = window['train_result']
                train_metric = train_result.get(objective, 0)
                train_metrics.append(train_metric)
                
                test_result = window.get('test_result', {})
                test_metric = test_result.get(objective, 0)
                test_metrics.append(test_metric)
            
            # Plot train vs. test metrics
            plt.figure(figsize=(12, 6))
            plt.plot(window_indices, train_metrics, 'b-', label='Train')
            plt.plot(window_indices, test_metrics, 'r-', label='Test')
            plt.title(f'Walk-Forward {objective} by Window')
            plt.xlabel('Window Index')
            plt.ylabel(objective)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'walk_forward_metrics.png'))
            plt.close()
            
            # Plot train/test ratio (stability)
            ratios = []
            for train_metric, test_metric in zip(train_metrics, test_metrics):
                if train_metric != 0:
                    ratios.append(test_metric / train_metric)
                else:
                    ratios.append(0)
            
            plt.figure(figsize=(12, 6))
            plt.plot(window_indices, ratios, 'g-')
            plt.axhline(y=1.0, color='r', linestyle='--')
            plt.title(f'Test/Train {objective} Ratio by Window')
            plt.xlabel('Window Index')
            plt.ylabel('Test/Train Ratio')
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'test_train_ratio.png'))
            plt.close()
            
            logger.info(f"Walk-forward visualization saved to {result_dir}")
            
        except Exception as e:
            logger.warning(f"Error plotting walk-forward results: {e}")
    
    def plot_regime_performance(self, optimization_result: Dict[str, Any]) -> None:
        """
        Plot performance across different market regimes
        
        Args:
            optimization_result: Optimization results
        """
        try:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(self.output_dir, f"regime_performance_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)
            
            # Extract best result
            best_params = optimization_result['best_params']
            best_result = None
            
            for eval_result in optimization_result['evaluations']:
                if eval_result['params'] == best_params:
                    best_result = eval_result
                    break
            
            if not best_result:
                return
            
            # Collect regime performance metrics
            regime_performance = {}
            objective = self.objective_metric.value
            
            for window in best_result['windows']:
                regime_results = window.get('regime_results', {})
                for regime, metrics in regime_results.items():
                    if regime not in regime_performance:
                        regime_performance[regime] = []
                    
                    metric_value = metrics.get(objective, 0)
                    regime_performance[regime].append(metric_value)
            
            # Calculate mean performance for each regime
            regime_means = {}
            for regime, values in regime_performance.items():
                if values:
                    regime_means[regime] = np.mean(values)
            
            # Plot regime performance comparison
            if regime_means:
                regimes = list(regime_means.keys())
                means = [regime_means[r] for r in regimes]
                
                plt.figure(figsize=(12, 6))
                plt.bar(regimes, means)
                plt.title(f'{objective} by Market Regime')
                plt.xlabel('Market Regime')
                plt.ylabel(objective)
                plt.grid(True, axis='y')
                plt.savefig(os.path.join(result_dir, 'regime_performance.png'))
                plt.close()
                
                # Calculate and plot regime stability
                overall_mean = np.mean(means)
                relative_performance = [m / overall_mean for m in means]
                
                plt.figure(figsize=(12, 6))
                plt.bar(regimes, relative_performance)
                plt.axhline(y=1.0, color='r', linestyle='--')
                plt.title('Relative Performance by Market Regime')
                plt.xlabel('Market Regime')
                plt.ylabel('Relative Performance (1.0 = Average)')
                plt.grid(True, axis='y')
                plt.savefig(os.path.join(result_dir, 'relative_regime_performance.png'))
                plt.close()
                
                logger.info(f"Regime performance visualization saved to {result_dir}")
        
        except Exception as e:
            logger.warning(f"Error plotting regime performance: {e}")

    def evaluate_parameters(
        self,
        strategy_evaluator: Callable,
        params: Dict[str, Any],
        prices: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate a specific set of parameters
        
        Args:
            strategy_evaluator: Strategy evaluation function
            params: Parameters to evaluate
            prices: Price series
            
        Returns:
            Evaluation results
        """
        # Segment data by regime if regime detector is available
        if self.regime_detector is not None:
            self.segment_data_by_regime(prices)
        
        # Create walk-forward windows
        windows = self.create_walk_forward_windows(len(prices))
        
        # Evaluate parameters on all windows
        window_results = []
        for window_idx, (train_indices, test_indices) in enumerate(windows):
            window_result = self._evaluate_window(
                strategy_evaluator=strategy_evaluator,
                params=params,
                train_indices=train_indices,
                test_indices=test_indices,
                window_idx=window_idx
            )
            window_results.append(window_result)
        
        # Calculate aggregated performance across windows
        aggregated_result = self._aggregate_window_results(window_results)
        
        # Create evaluation result
        eval_result = {
            'params': params,
            'windows': window_results,
            'aggregated': aggregated_result,
            'timestamp': datetime.now().isoformat()
        }
        
        return eval_result 