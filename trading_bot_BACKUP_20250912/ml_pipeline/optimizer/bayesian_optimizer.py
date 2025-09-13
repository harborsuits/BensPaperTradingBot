"""
Bayesian Optimizer Module

Provides Bayesian optimization for trading strategies using Gaussian Process Regression.
This approach is especially efficient for expensive-to-evaluate objective functions.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from trading_bot.ml_pipeline.optimizer.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization for trading strategies
    
    Uses Gaussian Process Regression to build a surrogate model of the 
    objective function and acquisition functions to determine the next
    parameters to try, making it very efficient for costly evaluations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Bayesian optimizer
        
        Args:
            config: Configuration dictionary with parameters
        """
        super().__init__(config)
        
        if not SKOPT_AVAILABLE:
            logger.warning("scikit-optimize is not installed. Installing it with 'pip install scikit-optimize' is recommended for Bayesian optimization.")
        
        # Bayesian optimization parameters
        self.n_calls = self.config.get('n_calls', 50)  # Total number of evaluations
        self.n_initial_points = self.config.get('n_initial_points', 10)  # Initial random explorations
        self.acquisition_function = self.config.get('acquisition_function', 'EI')  # Expected Improvement
        self.acquisition_optimizer = self.config.get('acquisition_optimizer', 'auto')
        self.n_restarts_optimizer = self.config.get('n_restarts_optimizer', 5)
        
        logger.info(f"Bayesian Optimizer initialized with {self.n_calls} calls, {self.n_initial_points} initial points")
    
    def optimize(self, 
                strategy_class, 
                param_space: Dict[str, Union[List, Tuple]], 
                historical_data: Dict[str, pd.DataFrame],
                metric: str = 'total_profit',
                metric_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Bayesian optimization
        
        Args:
            strategy_class: Class of the strategy to optimize
            param_space: Dictionary of parameter names and possible values
            historical_data: Dictionary of symbol -> DataFrame with historical data
            metric: Metric to optimize ('total_profit', 'sortino', 'sharpe', etc.)
            metric_function: Optional custom function to calculate metric
            
        Returns:
            Dictionary with optimization results
        """
        if not SKOPT_AVAILABLE:
            error_msg = "scikit-optimize is required for Bayesian optimization"
            logger.error(error_msg)
            return {"error": error_msg}
        
        logger.info(f"Starting Bayesian optimization for {strategy_class.__name__}")
        
        start_time = datetime.now()
        np.random.seed(self.random_state)
        
        # Convert parameter space to scikit-optimize format
        skopt_space, param_names = self._convert_param_space(param_space)
        
        # Keep track of all evaluations
        all_results = []
        
        # Create objective function
        @use_named_args(skopt_space)
        def objective_function(**params):
            # Convert params back to original format
            original_params = {}
            for name, value in params.items():
                # Handle special case for categorical parameters
                if name in param_space and isinstance(param_space[name][0], (str, bool)):
                    # Find the original value in the param space
                    original_params[name] = value
                else:
                    original_params[name] = value
            
            # Evaluate the strategy
            evaluation_result = self._evaluate_strategy(
                strategy_class, 
                original_params, 
                historical_data, 
                metric, 
                metric_function
            )
            
            # Store result
            all_results.append({
                'params': original_params.copy(),
                'metrics': evaluation_result.copy()
            })
            
            # Extract the metric to optimize
            # For minimization, negate if metric should be maximized
            metric_value = evaluation_result.get(metric, float('inf'))
            if metric not in ['max_drawdown']:  # Metrics where lower is better
                return -metric_value  # Negate for maximization
            return metric_value
        
        # Run Bayesian optimization
        try:
            result = gp_minimize(
                objective_function,
                skopt_space,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                acq_func=self.acquisition_function,
                acq_optimizer=self.acquisition_optimizer,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state,
                verbose=True
            )
            
            # Extract best parameters
            best_params = {}
            for i, name in enumerate(param_names):
                best_params[name] = result.x[i]
            
            # Find the corresponding metrics
            best_metrics = {}
            for evaluation in all_results:
                if all(evaluation['params'].get(k) == v for k, v in best_params.items()):
                    best_metrics = evaluation['metrics']
                    break
        except Exception as e:
            logger.error(f"Error during Bayesian optimization: {e}")
            return {"error": str(e)}
        
        # Calculate elapsed time
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        # Prepare final results
        final_results = {
            'strategy': strategy_class.__name__,
            'optimization_method': 'bayesian',
            'best_params': best_params,
            'best_metrics': best_metrics,
            'n_calls': self.n_calls,
            'n_initial_points': self.n_initial_points,
            'elapsed_time': elapsed_time,
            'timestamp': end_time.isoformat(),
            'all_evaluations': all_results
        }
        
        # Store full results
        self.optimization_results.append(final_results)
        
        # Save results to disk
        self._save_results(final_results)
        
        logger.info(f"Bayesian optimization completed in {elapsed_time:.1f} seconds")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {metric}: {best_metrics.get(metric, 'N/A')}")
        
        return final_results
    
    def _convert_param_space(self, param_space: Dict[str, Union[List, Tuple]]) -> Tuple[List, List[str]]:
        """
        Convert parameter space to scikit-optimize format
        
        Args:
            param_space: Dictionary of parameter names and possible values
            
        Returns:
            Tuple of (skopt_space, param_names)
        """
        skopt_space = []
        param_names = []
        
        for name, values in param_space.items():
            param_names.append(name)
            
            # Handle different parameter types
            if all(isinstance(v, (int, np.integer)) for v in values):
                # Integer parameter
                skopt_space.append(Integer(min(values), max(values), name=name))
            elif all(isinstance(v, (float, np.floating)) for v in values):
                # Real parameter
                skopt_space.append(Real(min(values), max(values), name=name))
            else:
                # Categorical parameter
                skopt_space.append(Categorical(values, name=name))
        
        return skopt_space, param_names
