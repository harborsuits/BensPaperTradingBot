"""
Optimization Methods for Strategy Components

Provides specific implementations of optimization algorithms for trading strategy components.
"""

import numpy as np
import pandas as pd
import logging
import random
import itertools
from typing import Dict, List, Any, Tuple, Optional
import time
import concurrent.futures
from tqdm import tqdm
import copy

from trading_bot.strategies.optimizer.enhanced_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class GridSearchOptimizer(BaseOptimizer):
    """Exhaustive grid search optimizer for strategy components"""
    
    def __init__(self):
        super().__init__()
        self.name = "grid_search"
        self.description = "Exhaustive grid search optimizer"
    
    def optimize(self) -> OptimizationResult:
        """Perform grid search optimization"""
        # Call parent method to initialize
        result = super().optimize()
        
        try:
            # Generate all parameter combinations
            param_names = list(self.parameter_ranges.keys())
            param_values = list(self.parameter_ranges.values())
            
            total_combinations = 1
            for values in param_values:
                total_combinations *= len(values)
            
            # Check if we have too many combinations
            if total_combinations > self.max_evaluations:
                logger.warning(f"Grid search would require {total_combinations} evaluations, "
                              f"which exceeds max_evaluations ({self.max_evaluations}). "
                              f"Consider using random search instead.")
                
                # We'll proceed but with a truncated grid
                result.warning = "Max evaluations exceeded, grid search truncated"
            
            # Generate all combinations (up to max_evaluations)
            combinations = list(itertools.product(*param_values))
            if len(combinations) > self.max_evaluations:
                # Truncate and shuffle to get a representative sample
                random.shuffle(combinations)
                combinations = combinations[:self.max_evaluations]
            
            # Convert to parameter dictionaries
            parameter_sets = []
            for combo in combinations:
                params = {name: value for name, value in zip(param_names, combo)}
                parameter_sets.append(params)
            
            # Start timer
            start_time = time.time()
            
            # Evaluate all parameter sets
            performance_metrics = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
                # Submit all evaluation tasks
                future_to_params = {
                    executor.submit(self._evaluate_parameters, params): params 
                    for params in parameter_sets
                }
                
                # Process results as they complete
                for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_params), 
                                               total=len(future_to_params),
                                               desc="Grid Search Progress")):
                    params = future_to_params[future]
                    try:
                        metrics = future.result()
                        # Add to performance metrics
                        performance_metrics.append(metrics)
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {params}: {e}")
                        # Add dummy metrics with error
                        performance_metrics.append({'error': str(e)})
            
            # End timer
            end_time = time.time()
            optimization_time = end_time - start_time
            
            # Find best parameters based on evaluation metric
            best_idx, best_metrics = self._find_best_metrics(performance_metrics)
            
            if best_idx is not None:
                best_parameters = parameter_sets[best_idx]
            else:
                best_parameters = {}
            
            # Calculate parameter importance
            parameter_importance = self._calculate_parameter_importance(parameter_sets, performance_metrics)
            
            # Update result
            result.parameter_sets = [
                {'parameters': params, 'metrics': metrics}
                for params, metrics in zip(parameter_sets, performance_metrics)
            ]
            result.best_parameters = best_parameters
            result.best_performance = best_metrics
            result.optimization_time = optimization_time
            result.status = "completed"
            result.parameter_importance = parameter_importance
            
            # Save results if path is provided
            if self.save_path:
                self._save_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in grid search optimization: {e}")
            result.status = "failed"
            result.error_message = str(e)
            return result
    
    def _find_best_metrics(self, metrics_list: List[Dict[str, float]]) -> Tuple[Optional[int], Dict[str, float]]:
        """Find the best metrics based on the evaluation metric"""
        best_idx = None
        best_value = float('-inf') if self.evaluation_metric != 'max_drawdown' else float('inf')
        best_metrics = {}
        
        for i, metrics in enumerate(metrics_list):
            # Skip if there was an error
            if 'error' in metrics:
                continue
            
            # Get metric value based on evaluation metric
            if self.evaluation_metric in metrics:
                value = metrics[self.evaluation_metric]
            elif 'total_profit' in metrics:
                value = metrics['total_profit']
            elif 'profit_improvement' in metrics:
                value = metrics['profit_improvement']
            elif 'total_weighted_return' in metrics:
                value = metrics['total_weighted_return']
            else:
                # No valid metric found
                continue
            
            # Compare based on direction (minimize for drawdown, maximize for others)
            if self.evaluation_metric == 'max_drawdown':
                if value < best_value:
                    best_value = value
                    best_idx = i
                    best_metrics = metrics
            else:
                if value > best_value:
                    best_value = value
                    best_idx = i
                    best_metrics = metrics
        
        return best_idx, best_metrics
    
    def _calculate_parameter_importance(self, parameter_sets: List[Dict[str, Any]], 
                                       metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate parameter importance based on correlation with performance"""
        # This is a simplified approach - a real implementation would use more sophisticated methods
        
        importance = {}
        
        # Convert parameters to DataFrame
        param_df = pd.DataFrame(parameter_sets)
        
        # Extract metric values
        metric_values = []
        for metrics in metrics_list:
            # Skip if there was an error
            if 'error' in metrics:
                metric_values.append(np.nan)
                continue
            
            # Get metric value based on evaluation metric
            if self.evaluation_metric in metrics:
                value = metrics[self.evaluation_metric]
            elif 'total_profit' in metrics:
                value = metrics['total_profit']
            elif 'profit_improvement' in metrics:
                value = metrics['profit_improvement']
            elif 'total_weighted_return' in metrics:
                value = metrics['total_weighted_return']
            else:
                # No valid metric found
                value = np.nan
            
            metric_values.append(value)
        
        # Skip if we don't have enough valid values
        if sum(~np.isnan(metric_values)) < 5:
            return importance
        
        # Add metric values to DataFrame
        param_df['metric'] = metric_values
        
        # Drop rows with NaN values
        param_df = param_df.dropna()
        
        # Calculate correlation for numeric parameters
        for param in param_df.columns:
            if param == 'metric':
                continue
            
            # Skip non-numeric parameters
            if not pd.api.types.is_numeric_dtype(param_df[param]):
                continue
            
            # Calculate correlation
            corr = param_df[param].corr(param_df['metric'])
            
            # Use absolute correlation as importance
            importance[param] = abs(corr)
        
        # Normalize importance values
        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                for param, value in importance.items():
                    importance[param] = value / max_importance
        
        return importance


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimizer for strategy components"""
    
    def __init__(self):
        super().__init__()
        self.name = "random_search"
        self.description = "Random search optimizer"
    
    def optimize(self) -> OptimizationResult:
        """Perform random search optimization"""
        # Call parent method to initialize
        result = super().optimize()
        
        try:
            # Generate random parameter sets
            parameter_sets = []
            
            for _ in range(self.max_evaluations):
                params = {}
                for param_name, param_values in self.parameter_ranges.items():
                    # Randomly select a value
                    params[param_name] = random.choice(param_values)
                
                parameter_sets.append(params)
            
            # Start timer
            start_time = time.time()
            
            # Evaluate all parameter sets
            performance_metrics = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
                # Submit all evaluation tasks
                future_to_params = {
                    executor.submit(self._evaluate_parameters, params): params 
                    for params in parameter_sets
                }
                
                # Process results as they complete
                for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_params), 
                                               total=len(future_to_params),
                                               desc="Random Search Progress")):
                    params = future_to_params[future]
                    try:
                        metrics = future.result()
                        # Add to performance metrics
                        performance_metrics.append(metrics)
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {params}: {e}")
                        # Add dummy metrics with error
                        performance_metrics.append({'error': str(e)})
            
            # End timer
            end_time = time.time()
            optimization_time = end_time - start_time
            
            # Find best parameters based on evaluation metric
            best_idx, best_metrics = self._find_best_metrics(performance_metrics)
            
            if best_idx is not None:
                best_parameters = parameter_sets[best_idx]
            else:
                best_parameters = {}
            
            # Calculate parameter importance
            parameter_importance = self._calculate_parameter_importance(parameter_sets, performance_metrics)
            
            # Update result
            result.parameter_sets = [
                {'parameters': params, 'metrics': metrics}
                for params, metrics in zip(parameter_sets, performance_metrics)
            ]
            result.best_parameters = best_parameters
            result.best_performance = best_metrics
            result.optimization_time = optimization_time
            result.status = "completed"
            result.parameter_importance = parameter_importance
            
            # Save results if path is provided
            if self.save_path:
                self._save_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in random search optimization: {e}")
            result.status = "failed"
            result.error_message = str(e)
            return result
    
    def _find_best_metrics(self, metrics_list: List[Dict[str, float]]) -> Tuple[Optional[int], Dict[str, float]]:
        """Find the best metrics based on the evaluation metric"""
        # Same implementation as in GridSearchOptimizer
        best_idx = None
        best_value = float('-inf') if self.evaluation_metric != 'max_drawdown' else float('inf')
        best_metrics = {}
        
        for i, metrics in enumerate(metrics_list):
            # Skip if there was an error
            if 'error' in metrics:
                continue
            
            # Get metric value based on evaluation metric
            if self.evaluation_metric in metrics:
                value = metrics[self.evaluation_metric]
            elif 'total_profit' in metrics:
                value = metrics['total_profit']
            elif 'profit_improvement' in metrics:
                value = metrics['profit_improvement']
            elif 'total_weighted_return' in metrics:
                value = metrics['total_weighted_return']
            else:
                # No valid metric found
                continue
            
            # Compare based on direction (minimize for drawdown, maximize for others)
            if self.evaluation_metric == 'max_drawdown':
                if value < best_value:
                    best_value = value
                    best_idx = i
                    best_metrics = metrics
            else:
                if value > best_value:
                    best_value = value
                    best_idx = i
                    best_metrics = metrics
        
        return best_idx, best_metrics
    
    def _calculate_parameter_importance(self, parameter_sets: List[Dict[str, Any]], 
                                       metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate parameter importance based on correlation with performance"""
        # Same implementation as in GridSearchOptimizer
        importance = {}
        
        # Convert parameters to DataFrame
        param_df = pd.DataFrame(parameter_sets)
        
        # Extract metric values
        metric_values = []
        for metrics in metrics_list:
            # Skip if there was an error
            if 'error' in metrics:
                metric_values.append(np.nan)
                continue
            
            # Get metric value based on evaluation metric
            if self.evaluation_metric in metrics:
                value = metrics[self.evaluation_metric]
            elif 'total_profit' in metrics:
                value = metrics['total_profit']
            elif 'profit_improvement' in metrics:
                value = metrics['profit_improvement']
            elif 'total_weighted_return' in metrics:
                value = metrics['total_weighted_return']
            else:
                # No valid metric found
                value = np.nan
            
            metric_values.append(value)
        
        # Skip if we don't have enough valid values
        if sum(~np.isnan(metric_values)) < 5:
            return importance
        
        # Add metric values to DataFrame
        param_df['metric'] = metric_values
        
        # Drop rows with NaN values
        param_df = param_df.dropna()
        
        # Calculate correlation for numeric parameters
        for param in param_df.columns:
            if param == 'metric':
                continue
            
            # Skip non-numeric parameters
            if not pd.api.types.is_numeric_dtype(param_df[param]):
                continue
            
            # Calculate correlation
            corr = param_df[param].corr(param_df['metric'])
            
            # Use absolute correlation as importance
            importance[param] = abs(corr)
        
        # Normalize importance values
        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                for param, value in importance.items():
                    importance[param] = value / max_importance
        
        return importance


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization for strategy components"""
    
    def __init__(self):
        super().__init__()
        self.name = "bayesian"
        self.description = "Bayesian optimization"
        
        # Optional dependency warning
        try:
            import skopt
            self._skopt_available = True
        except ImportError:
            self._skopt_available = False
            logger.warning("scikit-optimize (skopt) not available. "
                          "Bayesian optimization will fall back to random search.")
    
    def optimize(self) -> OptimizationResult:
        """Perform Bayesian optimization if scikit-optimize is available"""
        # Call parent method to initialize
        result = super().optimize()
        
        if not self._skopt_available:
            logger.warning("Falling back to random search as scikit-optimize is not available")
            # Create a random search optimizer and use it instead
            random_optimizer = RandomSearchOptimizer()
            random_optimizer.component = self.component
            random_optimizer.historical_data = self.historical_data
            random_optimizer.parameter_ranges = self.parameter_ranges
            random_optimizer.evaluation_metric = self.evaluation_metric
            random_optimizer.max_evaluations = self.max_evaluations
            random_optimizer.parallel_jobs = self.parallel_jobs
            random_optimizer.save_path = self.save_path
            
            # Run optimization
            fallback_result = random_optimizer.optimize()
            
            # Update result with note about fallback
            fallback_result.optimization_method = "bayesian (fallback to random)"
            
            return fallback_result
        
        try:
            import skopt
            from skopt.space import Real, Integer, Categorical
            from skopt import gp_minimize, forest_minimize
            
            # Convert parameter ranges to skopt space
            space = []
            param_names = []
            
            for param_name, param_values in self.parameter_ranges.items():
                param_names.append(param_name)
                
                # Determine parameter type and create appropriate space
                if all(isinstance(v, int) for v in param_values):
                    # Integer parameter
                    space.append(Integer(min(param_values), max(param_values)))
                elif all(isinstance(v, float) for v in param_values):
                    # Float parameter
                    space.append(Real(min(param_values), max(param_values)))
                else:
                    # Categorical parameter
                    space.append(Categorical(param_values))
            
            # Define objective function
            def objective(x):
                # Convert skopt parameters to component parameters
                params = {name: value for name, value in zip(param_names, x)}
                
                # Evaluate parameters
                metrics = self._evaluate_parameters(params)
                
                # Extract relevant metric value
                if self.evaluation_metric in metrics:
                    value = metrics[self.evaluation_metric]
                elif 'total_profit' in metrics:
                    value = metrics['total_profit']
                elif 'profit_improvement' in metrics:
                    value = metrics['profit_improvement']
                elif 'total_weighted_return' in metrics:
                    value = metrics['total_weighted_return']
                else:
                    # No valid metric found
                    value = float('-inf')
                
                # Negate value for metrics we want to maximize (skopt minimizes)
                if self.evaluation_metric != 'max_drawdown':
                    value = -value
                
                return value
            
            # Start timer
            start_time = time.time()
            
            # Run Bayesian optimization
            res = gp_minimize(
                objective,
                space,
                n_calls=self.max_evaluations,
                n_random_starts=min(10, self.max_evaluations // 3),
                random_state=42
            )
            
            # End timer
            end_time = time.time()
            optimization_time = end_time - start_time
            
            # Convert results
            best_params = {name: value for name, value in zip(param_names, res.x)}
            
            # Evaluate best parameters to get metrics
            best_metrics = self._evaluate_parameters(best_params)
            
            # Update result
            result.best_parameters = best_params
            result.best_performance = best_metrics
            result.optimization_time = optimization_time
            result.status = "completed"
            
            # Extract all evaluated points for parameter_sets
            parameter_sets = []
            for i, x in enumerate(res.x_iters):
                params = {name: value for name, value in zip(param_names, x)}
                # Convert objective value back to original metric (negate if needed)
                y = res.func_vals[i]
                if self.evaluation_metric != 'max_drawdown':
                    y = -y
                
                parameter_sets.append({
                    'parameters': params,
                    'metrics': {self.evaluation_metric: y}
                })
            
            result.parameter_sets = parameter_sets
            
            # Save results if path is provided
            if self.save_path:
                self._save_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            result.status = "failed"
            result.error_message = str(e)
            return result
