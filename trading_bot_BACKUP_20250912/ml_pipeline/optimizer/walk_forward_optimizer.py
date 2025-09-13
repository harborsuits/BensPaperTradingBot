"""
Walk Forward Optimizer Module

Implements walk-forward optimization to prevent overfitting by testing 
strategies on out-of-sample data with rolling time windows.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import os
import json

from trading_bot.ml_pipeline.optimizer.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

class WalkForwardOptimizer:
    """
    Walk-forward optimization for trading strategies
    
    Uses rolling time windows with:
    - In-sample period for optimization
    - Out-of-sample period for validation
    
    This approach better simulates real trading and reduces overfitting
    by ensuring strategies work on previously unseen data.
    """
    
    def __init__(self, base_optimizer: BaseOptimizer, config=None):
        """
        Initialize the walk-forward optimizer
        
        Args:
            base_optimizer: Base optimizer to use for each in-sample period
            config: Configuration dictionary with parameters
        """
        self.base_optimizer = base_optimizer
        self.config = config or {}
        
        # Storage for results
        self.optimization_results = []
        self.validation_results = []
        self.cumulative_results = {}
        
        # Walk-forward parameters
        self.in_sample_days = self.config.get('in_sample_days', 90)  # 3 months in-sample
        self.out_sample_days = self.config.get('out_sample_days', 30)  # 1 month out-of-sample
        self.step_days = self.config.get('step_days', 30)  # Step forward by 1 month
        self.min_history_days = self.config.get('min_history_days', 180)  # Minimum data required
        self.n_windows = self.config.get('n_windows', 6)  # Number of walk-forward windows
        
        # Results directory
        self.results_dir = self.config.get('results_dir', 'walk_forward_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Walk Forward Optimizer initialized with {self.n_windows} windows")
    
    def optimize(self, 
                strategy_class, 
                param_space: Dict[str, Union[List, Tuple]], 
                historical_data: Dict[str, pd.DataFrame],
                metric: str = 'sharpe_ratio',
                metric_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run walk-forward optimization
        
        Args:
            strategy_class: Class of the strategy to optimize
            param_space: Dictionary of parameter names and possible values
            historical_data: Dictionary of symbol -> DataFrame with historical data
            metric: Metric to optimize ('total_profit', 'sortino', 'sharpe', etc.)
            metric_function: Optional custom function to calculate metric
            
        Returns:
            Dictionary with walk-forward optimization results
        """
        logger.info(f"Starting walk-forward optimization for {strategy_class.__name__}")
        
        start_time = datetime.now()
        
        # Clear previous results
        self.optimization_results = []
        self.validation_results = []
        self.cumulative_results = {}
        
        # Check if we have enough data
        for symbol, df in historical_data.items():
            if len(df) < self.min_history_days:
                logger.warning(f"Not enough data for {symbol}: {len(df)} days < {self.min_history_days} days")
        
        # Calculate window boundaries
        windows = self._calculate_windows(historical_data)
        
        if not windows:
            error_msg = "Unable to calculate windows, not enough data"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Run walk-forward optimization
        for window_idx, window in enumerate(windows):
            logger.info(f"Processing window {window_idx+1}/{len(windows)}")
            
            # Get in-sample and out-of-sample data
            in_sample_data = self._slice_data(historical_data, window['in_sample_start'], window['in_sample_end'])
            out_sample_data = self._slice_data(historical_data, window['out_sample_start'], window['out_sample_end'])
            
            # Skip if not enough data
            if not in_sample_data or not out_sample_data:
                logger.warning(f"Skipping window {window_idx+1} due to insufficient data")
                continue
            
            logger.info(f"In-sample period: {window['in_sample_start']} to {window['in_sample_end']}")
            logger.info(f"Out-of-sample period: {window['out_sample_start']} to {window['out_sample_end']}")
            
            # Optimize using in-sample data
            try:
                in_sample_results = self.base_optimizer.optimize(
                    strategy_class,
                    param_space,
                    in_sample_data,
                    metric,
                    metric_function
                )
                
                self.optimization_results.append({
                    'window': window_idx + 1,
                    'in_sample_start': window['in_sample_start'],
                    'in_sample_end': window['in_sample_end'],
                    'best_params': in_sample_results['best_params'],
                    'best_metrics': in_sample_results['best_metrics']
                })
                
                # Validate on out-of-sample data
                validation_results = self._validate_params(
                    strategy_class,
                    in_sample_results['best_params'],
                    out_sample_data,
                    metric
                )
                
                self.validation_results.append({
                    'window': window_idx + 1,
                    'out_sample_start': window['out_sample_start'],
                    'out_sample_end': window['out_sample_end'],
                    'params': in_sample_results['best_params'],
                    'metrics': validation_results
                })
            except Exception as e:
                logger.error(f"Error processing window {window_idx+1}: {e}")
        
        # Calculate aggregate results
        robustness_metrics = self._calculate_robustness()
        
        # Calculate elapsed time
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        # Prepare final results
        final_results = {
            'strategy': strategy_class.__name__,
            'optimization_method': 'walk_forward',
            'n_windows': len(windows),
            'in_sample_days': self.in_sample_days,
            'out_sample_days': self.out_sample_days,
            'step_days': self.step_days,
            'optimization_results': self.optimization_results,
            'validation_results': self.validation_results,
            'robustness_metrics': robustness_metrics,
            'elapsed_time': elapsed_time,
            'timestamp': end_time.isoformat()
        }
        
        # Store cumulative results
        self.cumulative_results = final_results
        
        # Save results to disk
        self._save_results(final_results)
        
        logger.info(f"Walk-forward optimization completed in {elapsed_time:.1f} seconds")
        
        return final_results
    
    def _calculate_windows(self, historical_data: Dict[str, pd.DataFrame]) -> List[Dict[str, datetime]]:
        """
        Calculate time windows for walk-forward testing
        
        Args:
            historical_data: Dictionary of symbol -> DataFrame with historical data
            
        Returns:
            List of window dictionaries with start/end dates
        """
        # Find the common date range across all symbols
        min_start_date = None
        max_end_date = None
        
        for symbol, df in historical_data.items():
            if df.empty:
                continue
                
            start_date = df.index[0].to_pydatetime()
            end_date = df.index[-1].to_pydatetime()
            
            if min_start_date is None or start_date > min_start_date:
                min_start_date = start_date
                
            if max_end_date is None or end_date < max_end_date:
                max_end_date = end_date
        
        if min_start_date is None or max_end_date is None:
            logger.error("No valid data found")
            return []
        
        # Check if we have enough data
        total_days = (max_end_date - min_start_date).days
        if total_days < self.min_history_days:
            logger.error(f"Not enough data: {total_days} days < {self.min_history_days} days")
            return []
        
        # Calculate windows
        windows = []
        
        # Start from the earliest possible in-sample end date
        in_sample_end = max_end_date - timedelta(days=self.out_sample_days)
        
        for i in range(self.n_windows):
            # Calculate window boundaries
            in_sample_start = in_sample_end - timedelta(days=self.in_sample_days)
            out_sample_start = in_sample_end + timedelta(days=1)  # Day after in-sample ends
            out_sample_end = out_sample_start + timedelta(days=self.out_sample_days - 1)
            
            # Make sure we have enough data
            if in_sample_start < min_start_date:
                logger.info(f"Reached beginning of data, stopping at window {i+1}")
                break
                
            # Add window
            windows.append({
                'in_sample_start': in_sample_start,
                'in_sample_end': in_sample_end,
                'out_sample_start': out_sample_start,
                'out_sample_end': out_sample_end
            })
            
            # Move to next window
            in_sample_end = in_sample_end - timedelta(days=self.step_days)
        
        # Reverse the windows so they are in chronological order
        return list(reversed(windows))
    
    def _slice_data(self, 
                   historical_data: Dict[str, pd.DataFrame], 
                   start_date: datetime, 
                   end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Slice data for a specific time period
        
        Args:
            historical_data: Dictionary of symbol -> DataFrame with historical data
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dictionary of symbol -> DataFrame with sliced data
        """
        sliced_data = {}
        
        for symbol, df in historical_data.items():
            if df.empty:
                continue
                
            # Slice data
            mask = (df.index >= start_date) & (df.index <= end_date)
            sliced_df = df.loc[mask].copy()
            
            if not sliced_df.empty:
                sliced_data[symbol] = sliced_df
        
        return sliced_data
    
    def _validate_params(self, 
                        strategy_class, 
                        params: Dict[str, Any], 
                        out_sample_data: Dict[str, pd.DataFrame],
                        metric: str) -> Dict[str, float]:
        """
        Validate parameters on out-of-sample data
        
        Args:
            strategy_class: Strategy class to validate
            params: Parameters to validate
            out_sample_data: Out-of-sample data
            metric: Metric to track
            
        Returns:
            Dictionary with validation metrics
        """
        # Initialize strategy with parameters
        strategy = strategy_class(parameters=params)
        
        # Calculate metrics for each symbol
        symbol_results = []
        
        for symbol, data in out_sample_data.items():
            if data.empty:
                continue
                
            # Generate signals
            signals = []
            for i in range(len(data) - 1):  # Leave last row for position exit
                df_subset = data.iloc[:i+1].copy()
                try:
                    signal = strategy.generate_signals(df_subset)
                    if signal:
                        signal['timestamp'] = data.index[i]
                        signal['symbol'] = symbol
                        signals.append(signal)
                except Exception as e:
                    logger.debug(f"Error generating validation signal for {symbol} at index {i}: {e}")
            
            # If no signals generated, skip this symbol
            if not signals:
                continue
                
            # Convert signals to dataframe
            signals_df = pd.DataFrame(signals)
            
            # Calculate performance metrics
            symbol_metrics = self.base_optimizer._calculate_metrics(signals_df, data, symbol)
            symbol_results.append(symbol_metrics)
        
        # Combine results across symbols
        if not symbol_results:
            return {"error": "No signals generated for any symbol"}
            
        return self.base_optimizer._combine_symbol_metrics(symbol_results)
    
    def _calculate_robustness(self) -> Dict[str, float]:
        """
        Calculate robustness metrics from walk-forward results
        
        Returns:
            Dictionary with robustness metrics
        """
        # If we don't have results, return empty metrics
        if not self.validation_results:
            return {}
        
        # Extract metrics from validation results
        validation_metrics = {}
        
        for validation in self.validation_results:
            window = validation['window']
            metrics = validation['metrics']
            
            for metric_name, value in metrics.items():
                if metric_name not in validation_metrics:
                    validation_metrics[metric_name] = []
                    
                validation_metrics[metric_name].append(value)
        
        # Calculate robustness metrics
        robustness = {}
        
        for metric_name, values in validation_metrics.items():
            if not values:
                continue
                
            # Calculate statistics
            robustness[f"{metric_name}_mean"] = np.mean(values)
            robustness[f"{metric_name}_std"] = np.std(values)
            robustness[f"{metric_name}_min"] = np.min(values)
            robustness[f"{metric_name}_max"] = np.max(values)
            
            # Calculate consistency score (ratio of positive windows)
            if metric_name in ['total_profit', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']:
                positive_count = sum(1 for v in values if v > 0)
                robustness[f"{metric_name}_consistency"] = positive_count / len(values) if len(values) > 0 else 0
            elif metric_name in ['max_drawdown']:
                # For drawdown (lower is better), use negative threshold
                positive_count = sum(1 for v in values if v > -0.1)  # Drawdown better than -10%
                robustness[f"{metric_name}_consistency"] = positive_count / len(values) if len(values) > 0 else 0
        
        # Calculate parameter stability
        param_stability = self._calculate_parameter_stability()
        robustness.update(param_stability)
        
        return robustness
    
    def _calculate_parameter_stability(self) -> Dict[str, float]:
        """
        Calculate parameter stability across windows
        
        Returns:
            Dictionary with parameter stability metrics
        """
        # If we don't have results, return empty metrics
        if len(self.optimization_results) <= 1:
            return {}
        
        # Extract parameters from optimization results
        param_values = {}
        
        for opt_result in self.optimization_results:
            for param_name, value in opt_result['best_params'].items():
                if param_name not in param_values:
                    param_values[param_name] = []
                
                param_values[param_name].append(value)
        
        # Calculate stability metrics
        stability = {}
        
        for param_name, values in param_values.items():
            if not values:
                continue
                
            # Check if values are numeric
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                # Calculate coefficient of variation (lower is more stable)
                mean = np.mean(values)
                std = np.std(values)
                
                if mean != 0:
                    cv = std / abs(mean)
                    stability[f"{param_name}_cv"] = cv
                    stability[f"{param_name}_stability"] = 1 / (1 + cv)  # Transform to [0,1] where higher is better
                else:
                    stability[f"{param_name}_cv"] = float('inf')
                    stability[f"{param_name}_stability"] = 0
            else:
                # For non-numeric parameters, calculate entropy of distribution
                value_counts = {}
                for value in values:
                    str_value = str(value)
                    value_counts[str_value] = value_counts.get(str_value, 0) + 1
                
                # Normalize counts to probabilities
                n = len(values)
                entropy = -sum((count/n) * np.log(count/n) for count in value_counts.values())
                
                stability[f"{param_name}_entropy"] = entropy
                stability[f"{param_name}_stability"] = 1 / (1 + entropy)  # Transform to [0,1] where higher is better
        
        # Calculate overall parameter stability (average of individual stabilities)
        stability_scores = [v for k, v in stability.items() if k.endswith('_stability')]
        if stability_scores:
            stability["overall_parameter_stability"] = np.mean(stability_scores)
        
        return stability
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Save walk-forward results to disk
        
        Args:
            results: Walk-forward results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = results['strategy']
        filename = f"{strategy_name}_walkforward_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Filter non-serializable objects and convert numpy types
        serializable_results = {}
        
        for key, value in results.items():
            if key in ['optimization_results', 'validation_results']:
                # These contain lists of dictionaries
                serializable_list = []
                
                for item in value:
                    serializable_item = {}
                    
                    for k, v in item.items():
                        if isinstance(v, (np.integer, np.floating)):
                            serializable_item[k] = float(v)
                        elif isinstance(v, datetime):
                            serializable_item[k] = v.isoformat()
                        elif isinstance(v, dict):
                            # Handle nested dictionaries
                            serializable_item[k] = {
                                dk: float(dv) if isinstance(dv, (np.integer, np.floating)) else dv
                                for dk, dv in v.items()
                            }
                        else:
                            serializable_item[k] = v
                    
                    serializable_list.append(serializable_item)
                
                serializable_results[key] = serializable_list
            elif isinstance(value, dict):
                # Handle dictionaries
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
        logger.info(f"Walk-forward results saved to {filepath}")
