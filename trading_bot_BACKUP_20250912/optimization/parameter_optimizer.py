#!/usr/bin/env python3
"""
Parameter Optimizer Module

This module provides tools for optimizing trading system parameters through
grid search, Bayesian optimization, or genetic algorithms.

It can optimize:
- Anomaly detection thresholds
- Risk manager response levels
- Trading strategy parameters (MA lengths, etc.)
"""

import os
import logging
import itertools
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class OptimizationMethod(str, Enum):
    """Optimization methods supported by the optimizer."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"

class ParameterType(str, Enum):
    """Parameter types for optimization."""
    ANOMALY_DETECTION = "anomaly_detection"
    RISK_MANAGEMENT = "risk_management"
    TRADING_STRATEGY = "trading_strategy"
    COMBINED = "combined"

class ParameterOptimizer:
    """
    Optimizes trading system parameters using various search methods.
    
    Features:
    - Grid search across parameter space
    - Parallel processing for faster optimization
    - Result visualization
    - Parameter sensitivity analysis
    - Save/load optimization results
    """
    
    def __init__(self, 
                 optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                 parameter_type: ParameterType = ParameterType.COMBINED,
                 max_workers: int = None,
                 results_dir: str = "results/optimization"):
        """
        Initialize parameter optimizer.
        
        Args:
            optimization_method: Method to use for optimization
            parameter_type: Type of parameters to optimize
            max_workers: Maximum number of parallel workers (None = auto)
            results_dir: Directory to store optimization results
        """
        self.optimization_method = optimization_method
        self.parameter_type = parameter_type
        self.max_workers = max_workers
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Default parameter spaces
        self.anomaly_params = {
            "alert_threshold": np.linspace(0.6, 0.9, 7).tolist(),  # [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            "lookback_window": list(range(10, 31, 5)),  # [10, 15, 20, 25, 30]
            "contamination": [0.005, 0.01, 0.02, 0.03, 0.05]
        }
        
        self.risk_params = {
            "minimal_threshold": np.linspace(0.2, 0.4, 5).tolist(),  # [0.2, 0.25, 0.3, 0.35, 0.4]
            "moderate_threshold": np.linspace(0.4, 0.6, 5).tolist(),  # [0.4, 0.45, 0.5, 0.55, 0.6]
            "high_threshold": np.linspace(0.6, 0.8, 5).tolist(),  # [0.6, 0.65, 0.7, 0.75, 0.8]
            "critical_threshold": np.linspace(0.8, 0.95, 4).tolist(),  # [0.8, 0.85, 0.9, 0.95]
            "position_size_modifier_moderate": np.linspace(0.5, 0.8, 4).tolist(),  # [0.5, 0.6, 0.7, 0.8]
            "position_size_modifier_high": np.linspace(0.2, 0.5, 4).tolist(),  # [0.2, 0.3, 0.4, 0.5]
            "recovery_monitor_periods": list(range(2, 11, 2))  # [2, 4, 6, 8, 10]
        }
        
        self.strategy_params = {
            "short_ma_window": list(range(3, 16, 2)),  # [3, 5, 7, 9, 11, 13, 15]
            "long_ma_window": list(range(10, 31, 5)),  # [10, 15, 20, 25, 30]
            "risk_per_trade_pct": np.linspace(0.005, 0.02, 4).tolist(),  # [0.005, 0.01, 0.015, 0.02]
            "take_profit_atr_mult": np.linspace(1.5, 4.0, 6).tolist(),  # [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            "stop_loss_atr_mult": np.linspace(1.0, 3.0, 5).tolist()  # [1.0, 1.5, 2.0, 2.5, 3.0]
        }
        
        # Results storage
        self.results = []
        self.best_params = {}
        self.best_score = float('-inf')
        
        logger.info(f"Initialized ParameterOptimizer with {optimization_method} method for {parameter_type} parameters")
    
    def set_parameter_space(self, parameter_type: ParameterType, param_space: Dict[str, List]):
        """
        Set or override the parameter space for optimization.
        
        Args:
            parameter_type: Type of parameters to set
            param_space: Dictionary of parameter names and possible values
        """
        if parameter_type == ParameterType.ANOMALY_DETECTION:
            self.anomaly_params = param_space
        elif parameter_type == ParameterType.RISK_MANAGEMENT:
            self.risk_params = param_space
        elif parameter_type == ParameterType.TRADING_STRATEGY:
            self.strategy_params = param_space
        
        logger.info(f"Updated parameter space for {parameter_type}")
    
    def get_parameter_space(self, parameter_type: Optional[ParameterType] = None) -> Dict[str, List]:
        """
        Get the current parameter space.
        
        Args:
            parameter_type: Type of parameters to get (None = use instance type)
            
        Returns:
            Dictionary with parameter space
        """
        if parameter_type is None:
            parameter_type = self.parameter_type
        
        if parameter_type == ParameterType.ANOMALY_DETECTION:
            return self.anomaly_params
        elif parameter_type == ParameterType.RISK_MANAGEMENT:
            return self.risk_params
        elif parameter_type == ParameterType.TRADING_STRATEGY:
            return self.strategy_params
        elif parameter_type == ParameterType.COMBINED:
            # Prefix parameters to avoid name collisions
            combined = {}
            for k, v in self.anomaly_params.items():
                combined[f"anomaly_{k}"] = v
            for k, v in self.risk_params.items():
                combined[f"risk_{k}"] = v
            for k, v in self.strategy_params.items():
                combined[f"strategy_{k}"] = v
            return combined
        
        return {}
    
    def optimize(self, 
                evaluation_func: Callable[[Dict[str, Any]], float],
                parameter_type: Optional[ParameterType] = None,
                max_iterations: int = 100,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Run optimization using the specified method.
        
        Args:
            evaluation_func: Function that takes parameters and returns a score
            parameter_type: Type of parameters to optimize (None = use instance type)
            max_iterations: Maximum number of iterations for non-grid search methods
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with best parameters and score
        """
        if parameter_type is None:
            parameter_type = self.parameter_type
        
        param_space = self.get_parameter_space(parameter_type)
        
        if self.optimization_method == OptimizationMethod.GRID_SEARCH:
            return self._run_grid_search(param_space, evaluation_func, verbose)
        elif self.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            return self._run_random_search(param_space, evaluation_func, max_iterations, verbose)
        elif self.optimization_method == OptimizationMethod.BAYESIAN:
            return self._run_bayesian_optimization(param_space, evaluation_func, max_iterations, verbose)
        elif self.optimization_method == OptimizationMethod.GENETIC:
            return self._run_genetic_algorithm(param_space, evaluation_func, max_iterations, verbose)
        
        raise ValueError(f"Unsupported optimization method: {self.optimization_method}")
    
    def _run_grid_search(self, 
                       param_space: Dict[str, List],
                       evaluation_func: Callable[[Dict[str, Any]], float],
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Args:
            param_space: Dictionary of parameter names and possible values
            evaluation_func: Function that takes parameters and returns a score
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with best parameters and score
        """
        # Get parameter names and values
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        if verbose:
            logger.info(f"Running grid search with {total_combinations} parameter combinations")
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Run evaluation in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create future tasks
            future_to_params = {}
            for i, values in enumerate(combinations):
                # Create parameter dictionary
                params = dict(zip(param_names, values))
                
                # Submit task
                future = executor.submit(evaluation_func, params)
                future_to_params[future] = params
                
                if verbose and i % 100 == 0 and i > 0:
                    logger.info(f"Submitted {i}/{total_combinations} combinations")
            
            # Process results as they complete
            self.results = []
            for i, future in enumerate(as_completed(future_to_params)):
                params = future_to_params[future]
                try:
                    score = future.result()
                    
                    # Store result
                    result = {
                        "params": params,
                        "score": score
                    }
                    self.results.append(result)
                    
                    # Update best parameters if needed
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params
                    
                    if verbose and (i+1) % 100 == 0:
                        logger.info(f"Completed {i+1}/{total_combinations} evaluations. Current best: {self.best_score:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
        
        # Sort results by score (descending)
        self.results.sort(key=lambda x: x["score"], reverse=True)
        
        if verbose:
            logger.info(f"Grid search completed. Best score: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
    
    def _run_random_search(self,
                          param_space: Dict[str, List],
                          evaluation_func: Callable[[Dict[str, Any]], float],
                          max_iterations: int = 100,
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Run random search optimization.
        
        Args:
            param_space: Dictionary of parameter names and possible values
            evaluation_func: Function that takes parameters and returns a score
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with best parameters and score
        """
        param_names = list(param_space.keys())
        
        if verbose:
            logger.info(f"Running random search with {max_iterations} iterations")
        
        # Run evaluation in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create future tasks
            future_to_params = {}
            for i in range(max_iterations):
                # Create random parameter dictionary
                params = {}
                for name in param_names:
                    values = param_space[name]
                    params[name] = np.random.choice(values)
                
                # Submit task
                future = executor.submit(evaluation_func, params)
                future_to_params[future] = params
                
                if verbose and i % 10 == 0 and i > 0:
                    logger.info(f"Submitted {i}/{max_iterations} iterations")
            
            # Process results as they complete
            self.results = []
            for i, future in enumerate(as_completed(future_to_params)):
                params = future_to_params[future]
                try:
                    score = future.result()
                    
                    # Store result
                    result = {
                        "params": params,
                        "score": score
                    }
                    self.results.append(result)
                    
                    # Update best parameters if needed
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params
                    
                    if verbose and (i+1) % 10 == 0:
                        logger.info(f"Completed {i+1}/{max_iterations} evaluations. Current best: {self.best_score:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
        
        # Sort results by score (descending)
        self.results.sort(key=lambda x: x["score"], reverse=True)
        
        if verbose:
            logger.info(f"Random search completed. Best score: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
    
    def _run_bayesian_optimization(self,
                                 param_space: Dict[str, List],
                                 evaluation_func: Callable[[Dict[str, Any]], float],
                                 max_iterations: int = 100,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            param_space: Dictionary of parameter names and possible values
            evaluation_func: Function that takes parameters and returns a score
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with best parameters and score
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Categorical
            from skopt.utils import use_named_args
        except ImportError:
            logger.error("scikit-optimize package is required for Bayesian optimization")
            logger.error("Install with: pip install scikit-optimize")
            raise
        
        # Convert parameter space to skopt format
        dimensions = []
        dim_names = []
        
        for name, values in param_space.items():
            dimensions.append(Categorical(values, name=name))
            dim_names.append(name)
        
        # Create the objective function
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            try:
                score = evaluation_func(params)
                
                # Store result
                result = {
                    "params": params,
                    "score": score
                }
                self.results.append(result)
                
                # Update best score if needed
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                
                return -score  # Minimize negative score (maximize score)
            except Exception as e:
                logger.error(f"Error evaluating parameters {params}: {e}")
                return float('inf')  # Worst possible score
        
        if verbose:
            logger.info(f"Running Bayesian optimization with {max_iterations} iterations")
        
        # Reset results
        self.results = []
        
        # Run optimization
        res = gp_minimize(
            objective,
            dimensions=dimensions,
            n_calls=max_iterations,
            n_random_starts=min(10, max_iterations // 3),
            verbose=verbose
        )
        
        # Get best parameters
        best_params = dict(zip(dim_names, res.x))
        
        if verbose:
            logger.info(f"Bayesian optimization completed. Best score: {self.best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
        
        return {
            "best_params": best_params,
            "best_score": self.best_score
        }
    
    def _run_genetic_algorithm(self,
                             param_space: Dict[str, List],
                             evaluation_func: Callable[[Dict[str, Any]], float],
                             max_iterations: int = 100,
                             verbose: bool = True) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Args:
            param_space: Dictionary of parameter names and possible values
            evaluation_func: Function that takes parameters and returns a score
            max_iterations: Maximum number of iterations (generations)
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with best parameters and score
        """
        # Simple genetic algorithm implementation
        param_names = list(param_space.keys())
        population_size = min(50, max_iterations)
        mutation_rate = 0.1
        
        # Initialize population
        population = []
        for _ in range(population_size):
            # Create random parameter dictionary
            params = {}
            for name in param_names:
                values = param_space[name]
                params[name] = np.random.choice(values)
            
            population.append(params)
        
        if verbose:
            logger.info(f"Running genetic algorithm with {max_iterations} generations, population size {population_size}")
        
        # Reset results
        self.results = []
        
        # Run generations
        for generation in range(max_iterations):
            # Evaluate population
            scores = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all evaluations
                future_to_idx = {}
                for i, params in enumerate(population):
                    future = executor.submit(evaluation_func, params)
                    future_to_idx[future] = i
                
                # Process results
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        score = future.result()
                        scores.append((idx, score))
                        
                        # Store result
                        result = {
                            "params": population[idx],
                            "score": score,
                            "generation": generation
                        }
                        self.results.append(result)
                        
                        # Update best score if needed
                        if score > self.best_score:
                            self.best_score = score
                            self.best_params = population[idx]
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {population[idx]}: {e}")
                        scores.append((idx, float('-inf')))
            
            if verbose:
                logger.info(f"Generation {generation+1}/{max_iterations} completed. Best score: {self.best_score:.4f}")
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Check if we should continue
            if generation == max_iterations - 1:
                break
            
            # Select top half for reproduction
            top_indices = [idx for idx, _ in scores[:population_size//2]]
            parents = [population[idx] for idx in top_indices]
            
            # Create new population
            new_population = []
            
            # Elitism: Keep the best individual
            new_population.append(population[scores[0][0]])
            
            # Create offspring
            while len(new_population) < population_size:
                # Select two parents
                parent1 = np.random.choice(parents)
                parent2 = np.random.choice(parents)
                
                # Crossover
                child = {}
                for name in param_names:
                    # 50% chance of inheriting from each parent
                    if np.random.random() < 0.5:
                        child[name] = parent1[name]
                    else:
                        child[name] = parent2[name]
                
                # Mutation
                for name in param_names:
                    if np.random.random() < mutation_rate:
                        values = param_space[name]
                        child[name] = np.random.choice(values)
                
                new_population.append(child)
            
            # Replace population
            population = new_population
        
        if verbose:
            logger.info(f"Genetic algorithm completed. Best score: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
    
    def analyze_results(self, top_n: int = 10, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Analyze optimization results to find parameter importance.
        
        Args:
            top_n: Number of top results to analyze
            save_to_file: Whether to save results to file
            
        Returns:
            Dictionary with analysis results
        """
        if not self.results:
            logger.warning("No optimization results available for analysis")
            return {}
        
        # Sort results by score (descending)
        sorted_results = sorted(self.results, key=lambda x: x["score"], reverse=True)
        
        # Take top N results
        top_results = sorted_results[:top_n]
        
        # Analyze parameter ranges in top results
        param_ranges = {}
        param_values = {}
        
        for result in top_results:
            params = result["params"]
            for name, value in params.items():
                if name not in param_ranges:
                    param_ranges[name] = {"min": float('inf'), "max": float('-inf')}
                    param_values[name] = []
                
                # Update range
                value_float = float(value)
                param_ranges[name]["min"] = min(param_ranges[name]["min"], value_float)
                param_ranges[name]["max"] = max(param_ranges[name]["max"], value_float)
                
                # Store value
                param_values[name].append(value)
        
        # Calculate statistics
        param_stats = {}
        for name, values in param_values.items():
            value_floats = [float(v) for v in values]
            param_stats[name] = {
                "mean": np.mean(value_floats),
                "std": np.std(value_floats),
                "median": np.median(value_floats),
                "range": param_ranges[name],
                "values": values,
                "unique_values": list(set(values))
            }
        
        # Analyze parameter sensitivity
        sensitivity = {}
        for name in param_values.keys():
            # Group results by parameter value
            grouped = {}
            for result in self.results:
                value = result["params"].get(name)
                if value is not None:
                    if value not in grouped:
                        grouped[value] = []
                    grouped[value].append(result["score"])
            
            # Calculate average score for each value
            avg_scores = {}
            for value, scores in grouped.items():
                avg_scores[value] = np.mean(scores)
            
            # Calculate max difference in average scores
            if avg_scores:
                max_score = max(avg_scores.values())
                min_score = min(avg_scores.values())
                score_range = max_score - min_score
                
                # Store sensitivity
                sensitivity[name] = {
                    "score_range": score_range,
                    "avg_scores": avg_scores
                }
        
        # Calculate relative sensitivity
        if sensitivity:
            max_range = max(s["score_range"] for s in sensitivity.values())
            for name, data in sensitivity.items():
                data["relative_sensitivity"] = data["score_range"] / max_range if max_range > 0 else 0
        
        # Sort parameters by sensitivity
        sorted_sensitivity = sorted(
            sensitivity.items(),
            key=lambda x: x[1]["relative_sensitivity"],
            reverse=True
        )
        
        # Prepare analysis results
        analysis = {
            "top_results": top_results,
            "param_stats": param_stats,
            "sensitivity": sensitivity,
            "sorted_sensitivity": [name for name, _ in sorted_sensitivity]
        }
        
        if save_to_file:
            # Save analysis to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.results_dir, f"optimization_analysis_{timestamp}.json")
            
            # Prepare for JSON serialization
            json_analysis = {
                "top_results": top_results,
                "param_stats": {name: {k: v for k, v in stats.items() if k != "values"} 
                                for name, stats in param_stats.items()},
                "sorted_sensitivity": analysis["sorted_sensitivity"]
            }
            
            try:
                with open(filename, 'w') as f:
                    json.dump(json_analysis, f, indent=2)
                logger.info(f"Saved analysis to {filename}")
            except Exception as e:
                logger.error(f"Error saving analysis to file: {e}")
        
        return analysis
    
    def plot_results(self, 
                   top_n: int = 20, 
                   parameter_names: Optional[List[str]] = None,
                   save_to_file: bool = True) -> None:
        """
        Plot optimization results.
        
        Args:
            top_n: Number of top results to include
            parameter_names: List of parameter names to include (None = all)
            save_to_file: Whether to save plots to file
        """
        if not self.results:
            logger.warning("No optimization results available for plotting")
            return
        
        # Sort results by score (descending)
        sorted_results = sorted(self.results, key=lambda x: x["score"], reverse=True)
        
        # Take top N results
        top_results = sorted_results[:top_n]
        
        # Get all parameter names
        all_params = set()
        for result in top_results:
            all_params.update(result["params"].keys())
        
        # Filter parameter names if specified
        if parameter_names is not None:
            param_names = [name for name in parameter_names if name in all_params]
        else:
            param_names = list(all_params)
        
        # Sort by parameter name
        param_names.sort()
        
        # Create figure for parameter values
        n_params = len(param_names)
        if n_params == 0:
            logger.warning("No parameters to plot")
            return
        
        # Scatter plot for top results
        plt.figure(figsize=(12, 8))
        
        # Get scores
        scores = [r["score"] for r in top_results]
        
        # Create index column
        indices = list(range(len(top_results)))
        
        # Plot score
        plt.subplot(n_params + 1, 1, 1)
        plt.plot(indices, scores, 'o-', color='blue')
        plt.title("Top Optimization Results")
        plt.ylabel("Score")
        plt.grid(True)
        
        # Plot parameters
        for i, param in enumerate(param_names):
            plt.subplot(n_params + 1, 1, i + 2)
            
            # Get parameter values
            values = []
            for result in top_results:
                values.append(float(result["params"].get(param, float('nan'))))
            
            plt.plot(indices, values, 'o-', color='green')
            plt.ylabel(param)
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_to_file:
            # Save plot to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.results_dir, f"optimization_plot_{timestamp}.png")
            plt.savefig(filename)
            logger.info(f"Saved plot to {filename}")
        
        # Create sensitivity plot
        plt.figure(figsize=(12, 6))
        
        # Analyze parameter sensitivity
        sensitivity = {}
        for name in param_names:
            # Group results by parameter value
            grouped = {}
            for result in self.results:
                value = result["params"].get(name)
                if value is not None:
                    if value not in grouped:
                        grouped[value] = []
                    grouped[value].append(result["score"])
            
            # Calculate average score for each value
            avg_scores = {}
            for value, scores in grouped.items():
                avg_scores[value] = np.mean(scores)
            
            # Calculate max difference in average scores
            if avg_scores:
                max_score = max(avg_scores.values())
                min_score = min(avg_scores.values())
                score_range = max_score - min_score
                
                # Store sensitivity
                sensitivity[name] = {
                    "score_range": score_range,
                    "avg_scores": avg_scores
                }
        
        # Calculate relative sensitivity
        if sensitivity:
            max_range = max(s["score_range"] for s in sensitivity.values())
            for name, data in sensitivity.items():
                data["relative_sensitivity"] = data["score_range"] / max_range if max_range > 0 else 0
        
        # Sort parameters by sensitivity
        sorted_params = sorted(
            sensitivity.items(),
            key=lambda x: x[1]["relative_sensitivity"],
            reverse=True
        )
        
        # Plot sensitivity
        names = [name for name, _ in sorted_params]
        sensitivities = [data["relative_sensitivity"] for _, data in sorted_params]
        
        plt.barh(names, sensitivities, color='orange')
        plt.title("Parameter Sensitivity")
        plt.xlabel("Relative Sensitivity")
        plt.tight_layout()
        
        if save_to_file:
            # Save plot to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.results_dir, f"sensitivity_plot_{timestamp}.png")
            plt.savefig(filename)
            logger.info(f"Saved sensitivity plot to {filename}")
        
        # Show plots
        plt.show()
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save optimization results to file.
        
        Args:
            filename: Output filename (None = auto-generate)
            
        Returns:
            Path to saved file
        """
        if not self.results:
            logger.warning("No optimization results available to save")
            return ""
        
        if filename is None:
            # Auto-generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.results_dir, f"optimization_results_{timestamp}.json")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save results to file
        try:
            with open(filename, 'w') as f:
                json.dump({
                    "method": self.optimization_method,
                    "parameter_type": self.parameter_type,
                    "best_params": self.best_params,
                    "best_score": self.best_score,
                    "results": self.results[:1000],  # Limit to top 1000 results to avoid huge files
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Saved optimization results to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving results to file: {e}")
            return ""
    
    def load_results(self, filename: str) -> bool:
        """
        Load optimization results from file.
        
        Args:
            filename: Input filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Validate data
            if "method" not in data or "parameter_type" not in data:
                logger.error(f"Invalid optimization results file: {filename}")
                return False
            
            # Update instance
            self.optimization_method = data["method"]
            self.parameter_type = data["parameter_type"]
            self.best_params = data["best_params"]
            self.best_score = data["best_score"]
            self.results = data["results"]
            
            logger.info(f"Loaded optimization results from {filename}")
            logger.info(f"Best score: {self.best_score:.4f}")
            return True
        except Exception as e:
            logger.error(f"Error loading results from file: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example evaluation function
    def evaluate_parameters(params):
        # This would be your backtesting function
        # Return a performance metric (higher is better)
        
        # For testing, return a simple function of parameters
        score = 0.0
        
        if "anomaly_alert_threshold" in params:
            # Prefer higher threshold (fewer false positives)
            score += params["anomaly_alert_threshold"] * 0.5
        
        if "anomaly_lookback_window" in params:
            # Prefer moderate window size
            window = params["anomaly_lookback_window"]
            score += (1.0 - abs(window - 20) / 20) * 0.3
        
        if "risk_moderate_threshold" in params:
            # Prefer lower moderate threshold (more sensitive)
            score += (1.0 - params["risk_moderate_threshold"]) * 0.2
        
        # Add random noise
        score += np.random.normal(0, 0.1)
        
        return score
    
    # Create optimizer
    optimizer = ParameterOptimizer(
        optimization_method=OptimizationMethod.GRID_SEARCH,
        parameter_type=ParameterType.COMBINED,
        max_workers=4
    )
    
    # Set simplified parameter space for testing
    optimizer.set_parameter_space(
        ParameterType.ANOMALY_DETECTION,
        {
            "anomaly_alert_threshold": [0.6, 0.7, 0.8, 0.9],
            "anomaly_lookback_window": [10, 15, 20, 25, 30]
        }
    )
    
    # Run optimization
    print("Running optimization...")
    best = optimizer.optimize(evaluate_parameters, max_iterations=50, verbose=True)
    
    print(f"Best parameters: {best['best_params']}")
    print(f"Best score: {best['best_score']:.4f}")
    
    # Analyze results
    analysis = optimizer.analyze_results()
    
    # Plot results
    optimizer.plot_results()
    
    # Save results
    optimizer.save_results() 