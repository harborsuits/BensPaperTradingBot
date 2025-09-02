"""
Multi-Timeframe Optimizer Module

Provides cross-timeframe testing and validation for trading strategies.
"""

import logging
import copy
import itertools
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime

import numpy as np
import pandas as pd

from trading_bot.ml_pipeline.optimizer.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

class MultiTimeframeOptimizer:
    """
    Multi-timeframe optimizer for trading strategies
    
    Tests strategies across multiple timeframes and evaluates consistency
    of performance to identify robust strategies.
    """
    
    def __init__(self, base_optimizer: BaseOptimizer, config=None):
        """
        Initialize the multi-timeframe optimizer
        
        Args:
            base_optimizer: Base optimizer to use for each timeframe
            config: Configuration dictionary with parameters
        """
        self.base_optimizer = base_optimizer
        self.config = config or {}
        
        # Storage for results across timeframes
        self.timeframe_results = {}
        self.combined_results = {}
        
        # Configure timeframes to test
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', 'D'])
        self.min_consistent_timeframes = self.config.get('min_consistent_timeframes', 3)
        
        # Weights for different timeframes
        self.timeframe_weights = self.config.get('timeframe_weights', {})
        # Default weights if not specified
        if not self.timeframe_weights:
            # By default, higher timeframes have higher weights
            self.timeframe_weights = {
                '1m': 0.5,
                '5m': 0.6,
                '15m': 0.7,
                '30m': 0.8,
                '1h': 0.9,
                '4h': 1.0,
                'D': 1.1,
                'W': 1.2
            }
    
    def run_multi_timeframe_test(self, 
                                strategy_class, 
                                param_space: Dict[str, Union[List, Tuple]], 
                                historical_data_sets: Dict[str, Dict[str, pd.DataFrame]],
                                metric: str = 'total_profit',
                                metric_function: Optional[Callable] = None,
                                regime_detector=None) -> Dict[str, Any]:
        """
        Run optimization across multiple timeframes
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Dictionary of parameter names and possible values
            historical_data_sets: Dictionary of timeframe -> symbol -> DataFrame
            metric: Metric to optimize ('total_profit', 'sortino', 'sharpe', etc.)
            metric_function: Optional custom function to calculate metric
            regime_detector: Optional regime detector to use for regime-specific testing
            
        Returns:
            Dictionary with multi-timeframe optimization results
        """
        logger.info(f"Starting multi-timeframe optimization for {strategy_class.__name__}")
        
        start_time = datetime.now()
        
        # Clear previous results
        self.timeframe_results = {}
        self.combined_results = {}
        
        # Run optimization for each timeframe
        for timeframe in self.timeframes:
            if timeframe in historical_data_sets:
                logger.info(f"Optimizing for timeframe: {timeframe}")
                
                # Get data for this timeframe
                timeframe_data = historical_data_sets[timeframe]
                
                # Run optimization
                timeframe_results = self.base_optimizer.optimize(
                    strategy_class,
                    param_space,
                    timeframe_data,
                    metric,
                    metric_function
                )
                
                # Store results
                self.timeframe_results[timeframe] = timeframe_results
            else:
                logger.warning(f"No data available for timeframe: {timeframe}")
        
        # Calculate cross-timeframe consistency
        consistency_results = self.calculate_timeframe_consistency(metric)
        
        # Find best parameters across timeframes
        best_consistent_params = self._find_best_consistent_params(metric)
        
        # If we have a regime detector, test the best parameters under different regimes
        regime_results = {}
        if regime_detector:
            logger.info("Testing best parameters under different market regimes")
            regime_results = self._test_parameters_by_regime(
                strategy_class,
                best_consistent_params,
                historical_data_sets,
                metric,
                regime_detector
            )
        
        # Calculate elapsed time
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        # Prepare final results
        final_results = {
            'strategy': strategy_class.__name__,
            'optimization_method': 'multi_timeframe',
            'best_consistent_params': best_consistent_params,
            'timeframe_results': {tf: results['best_params'] for tf, results in self.timeframe_results.items()},
            'consistency_metrics': consistency_results,
            'regime_results': regime_results,
            'elapsed_time': elapsed_time,
            'timestamp': end_time.isoformat()
        }
        
        # Store combined results
        self.combined_results = final_results
        
        logger.info(f"Multi-timeframe optimization completed in {elapsed_time:.1f} seconds")
        logger.info(f"Best consistent parameters: {best_consistent_params}")
        
        return final_results
    
    def calculate_timeframe_consistency(self, metric: str) -> Dict[str, Any]:
        """
        Calculate consistency across timeframes
        
        Args:
            metric: Metric to use for consistency calculation
            
        Returns:
            Dict with consistency metrics
        """
        if not self.timeframe_results:
            return {"error": "No timeframe results available"}
        
        # Extract best parameters for each timeframe
        timeframe_best_params = {}
        timeframe_metrics = {}
        
        for timeframe, results in self.timeframe_results.items():
            timeframe_best_params[timeframe] = results['best_params']
            timeframe_metrics[timeframe] = results['best_metrics'].get(metric, 0)
        
        # Calculate parameter stability - how similar are the best parameters across timeframes
        param_stability = self._calculate_parameter_stability(timeframe_best_params)
        
        # Calculate performance consistency - how consistent is the performance across timeframes
        performance_consistency = self._calculate_performance_consistency(timeframe_metrics)
        
        # Calculate cross-validation score - test each timeframe's best parameters on other timeframes
        cross_validation_scores = self._calculate_cross_validation_scores(metric)
        
        return {
            'param_stability': param_stability,
            'performance_consistency': performance_consistency,
            'cross_validation_scores': cross_validation_scores,
            'timeframe_metrics': timeframe_metrics
        }
    
    def _calculate_parameter_stability(self, timeframe_best_params: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate parameter stability across timeframes
        
        Args:
            timeframe_best_params: Dictionary of timeframe -> best parameters
            
        Returns:
            Dict with parameter stability metrics
        """
        if len(timeframe_best_params) <= 1:
            return {'stability_score': 0, 'parameter_variance': {}}
        
        # Calculate variance of each parameter across timeframes
        all_param_names = set()
        for params in timeframe_best_params.values():
            all_param_names.update(params.keys())
        
        param_variance = {}
        param_values = {}
        
        for param_name in all_param_names:
            param_values[param_name] = []
            
            for timeframe, params in timeframe_best_params.items():
                if param_name in params:
                    value = params[param_name]
                    
                    # Convert to numeric if possible for variance calculation
                    if isinstance(value, (int, float)):
                        param_values[param_name].append(value)
            
            # Calculate variance if we have numeric values
            if param_values[param_name] and all(isinstance(v, (int, float)) for v in param_values[param_name]):
                param_variance[param_name] = np.var(param_values[param_name])
            else:
                # For non-numeric parameters, calculate entropy of distribution
                if param_values[param_name]:
                    value_counts = {}
                    for value in param_values[param_name]:
                        value_counts[str(value)] = value_counts.get(str(value), 0) + 1
                    
                    # Normalize counts to probabilities
                    n = len(param_values[param_name])
                    entropy = -sum((count/n) * np.log(count/n) for count in value_counts.values())
                    param_variance[param_name] = entropy
                else:
                    param_variance[param_name] = 0
        
        # Calculate overall stability score (lower variance = higher stability)
        if param_variance:
            # Normalize variances to [0,1] range and invert
            normalized_variances = []
            for var in param_variance.values():
                if var != 0:
                    # Apply logarithmic scaling to handle wide range of variances
                    norm_var = 1 / (1 + np.log1p(var))
                else:
                    norm_var = 1.0
                normalized_variances.append(norm_var)
            
            stability_score = np.mean(normalized_variances)
        else:
            stability_score = 0
        
        return {
            'stability_score': stability_score,
            'parameter_variance': param_variance
        }
    
    def _calculate_performance_consistency(self, timeframe_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate performance consistency across timeframes
        
        Args:
            timeframe_metrics: Dictionary of timeframe -> metric value
            
        Returns:
            Dict with performance consistency metrics
        """
        if len(timeframe_metrics) <= 1:
            return {
                'consistency_score': 0,
                'coefficient_of_variation': 0,
                'min_max_ratio': 0
            }
        
        # Extract metric values
        metric_values = list(timeframe_metrics.values())
        
        # Calculate statistics
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        
        # Coefficient of variation (lower is more consistent)
        if mean != 0:
            cv = std / abs(mean)
            cv_score = 1 / (1 + cv)  # Transform to [0,1] range where higher is better
        else:
            cv_score = 0
        
        # Min/max ratio (higher is more consistent)
        min_value = min(metric_values)
        max_value = max(metric_values)
        
        if max_value != 0:
            min_max_ratio = abs(min_value / max_value)
        else:
            min_max_ratio = 0
        
        # Overall consistency score
        consistency_score = (cv_score + min_max_ratio) / 2
        
        return {
            'consistency_score': consistency_score,
            'coefficient_of_variation': cv,
            'min_max_ratio': min_max_ratio
        }
    
    def _calculate_cross_validation_scores(self, metric: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate cross-validation scores across timeframes
        
        Args:
            metric: Metric to use for cross-validation
            
        Returns:
            Dict with cross-validation scores
        """
        # This would need to re-run the strategy with each timeframe's best parameters
        # We're returning a placeholder here - this would need to be implemented with
        # actual strategy evaluation on each timeframe's data
        
        cross_validation = {}
        for source_tf, source_results in self.timeframe_results.items():
            cross_validation[source_tf] = {}
            for target_tf in self.timeframe_results.keys():
                # Placeholder - would need actual evaluation
                if source_tf == target_tf:
                    cross_validation[source_tf][target_tf] = 1.0
                else:
                    cross_validation[source_tf][target_tf] = 0.5
        
        return cross_validation
    
    def _find_best_consistent_params(self, metric: str) -> Dict[str, Any]:
        """
        Find parameters that perform well across timeframes
        
        Args:
            metric: Metric to use for finding best parameters
            
        Returns:
            Dict with best consistent parameters
        """
        if not self.timeframe_results:
            return {}
        
        # Extract all parameter combinations that were evaluated
        all_param_sets = []
        for timeframe, results in self.timeframe_results.items():
            if 'all_evaluations' in results:
                all_param_sets.extend([
                    {
                        'params': eval['params'],
                        'timeframe': timeframe,
                        'metric': eval['metrics'].get(metric, float('-inf')),
                        'generation': eval.get('generation', 0)
                    }
                    for eval in results['all_evaluations']
                ])
        
        # Group by parameter combination
        param_performance = {}
        for param_set in all_param_sets:
            # Convert params to hashable format
            param_key = tuple(sorted((k, str(v)) for k, v in param_set['params'].items()))
            
            if param_key not in param_performance:
                param_performance[param_key] = {
                    'params': param_set['params'],
                    'timeframes': {},
                    'total_score': 0,
                    'weighted_score': 0,
                    'count': 0
                }
            
            # Record performance for this timeframe
            timeframe = param_set['timeframe']
            metric_value = param_set['metric']
            
            param_performance[param_key]['timeframes'][timeframe] = metric_value
            param_performance[param_key]['count'] += 1
        
        # Calculate aggregate scores
        for param_key, data in param_performance.items():
            # Calculate scores only if we have results for multiple timeframes
            if len(data['timeframes']) >= self.min_consistent_timeframes:
                # Simple total
                data['total_score'] = sum(data['timeframes'].values())
                
                # Weighted score by timeframe importance
                weighted_sum = 0
                total_weight = 0
                
                for tf, value in data['timeframes'].items():
                    weight = self.timeframe_weights.get(tf, 1.0)
                    weighted_sum += value * weight
                    total_weight += weight
                
                data['weighted_score'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Find parameters with best weighted score that have results for enough timeframes
        best_params = {}
        best_score = float('-inf')
        
        for data in param_performance.values():
            if len(data['timeframes']) >= self.min_consistent_timeframes and data['weighted_score'] > best_score:
                best_score = data['weighted_score']
                best_params = data['params']
        
        return best_params
    
    def _test_parameters_by_regime(self, 
                                 strategy_class, 
                                 parameters: Dict[str, Any],
                                 historical_data_sets: Dict[str, Dict[str, pd.DataFrame]],
                                 metric: str,
                                 regime_detector) -> Dict[str, Dict[str, float]]:
        """
        Test parameters under different market regimes
        
        Args:
            strategy_class: Strategy class to test
            parameters: Parameters to test
            historical_data_sets: Dictionary of timeframe -> symbol -> DataFrame
            metric: Metric to evaluate
            regime_detector: Regime detector to identify market regimes
            
        Returns:
            Dict with performance under different regimes
        """
        regime_results = {}
        
        # For each timeframe
        for timeframe, data in historical_data_sets.items():
            regime_results[timeframe] = {}
            
            # For each symbol
            for symbol, df in data.items():
                # Skip if data is too small
                if len(df) < 100:
                    continue
                
                # Detect regimes
                try:
                    regimes = regime_detector.detect_regimes(df)
                    
                    # Group data by regime
                    regime_data = {}
                    for i, regime in enumerate(regimes):
                        if regime not in regime_data:
                            regime_data[regime] = []
                        regime_data[regime].append(i)
                    
                    # Test strategy on each regime
                    for regime, indices in regime_data.items():
                        if len(indices) < 50:  # Skip if too few data points
                            continue
                            
                        # Create subset of data for this regime
                        regime_df = df.iloc[indices].copy()
                        
                        # Initialize strategy
                        strategy = strategy_class(parameters=parameters)
                        
                        # Generate signals
                        signals = []
                        for i in range(len(regime_df) - 1):
                            df_subset = regime_df.iloc[:i+1].copy()
                            try:
                                signal = strategy.generate_signals(df_subset)
                                if signal:
                                    signal['timestamp'] = regime_df.index[i]
                                    signal['symbol'] = symbol
                                    signals.append(signal)
                            except Exception as e:
                                logger.debug(f"Error generating signal for regime {regime}: {e}")
                        
                        # Calculate metrics
                        if signals:
                            signals_df = pd.DataFrame(signals)
                            metrics = self.base_optimizer._calculate_metrics(signals_df, regime_df, symbol)
                            
                            # Store results
                            if regime not in regime_results[timeframe]:
                                regime_results[timeframe][regime] = {}
                            
                            regime_results[timeframe][regime][symbol] = metrics
                
                except Exception as e:
                    logger.error(f"Error detecting regimes for {symbol} on {timeframe}: {e}")
        
        return regime_results
