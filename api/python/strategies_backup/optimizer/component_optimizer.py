"""
Component Optimizer

This module provides optimization utilities for strategy components.
It supports various optimization methods, including grid search, Bayesian optimization, and genetic algorithms.
"""

import numpy as np
import pandas as pd
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import multiprocessing
from functools import partial
import itertools

from trading_bot.strategies.modular_strategy import ModularStrategy
from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, 
    SignalGeneratorComponent, FilterComponent, 
    PositionSizerComponent, ExitManagerComponent
)
from trading_bot.strategies.components.component_registry import get_component_registry
from trading_bot.strategies.modular_strategy_integration import (
    ModularStrategyFactory, StrategyConfigGenerator
)

logger = logging.getLogger(__name__)

class OptimizationSpace:
    """Defines parameter space for optimization."""
    
    def __init__(self, param_name: str, param_type: str):
        """
        Initialize optimization space
        
        Args:
            param_name: Parameter name
            param_type: Parameter type ('int', 'float', 'bool', 'categorical')
        """
        self.param_name = param_name
        self.param_type = param_type
        self.values = []
        self.min_value = None
        self.max_value = None
        self.step = None
    
    def discrete(self, values: List[Any]) -> 'OptimizationSpace':
        """
        Set discrete values for parameter
        
        Args:
            values: List of possible values
            
        Returns:
            Self for chaining
        """
        self.values = values
        return self
    
    def range(self, min_value: Union[int, float], max_value: Union[int, float], 
             step: Optional[Union[int, float]] = None) -> 'OptimizationSpace':
        """
        Set range for parameter
        
        Args:
            min_value: Minimum value
            max_value: Maximum value
            step: Step size (for grid search)
            
        Returns:
            Self for chaining
        """
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        return self
    
    def sample(self, size: int = 1) -> List[Any]:
        """
        Sample values from parameter space
        
        Args:
            size: Number of samples
            
        Returns:
            List of sampled values
        """
        if self.values:
            # Sample from discrete values
            indices = np.random.randint(0, len(self.values), size=size)
            return [self.values[i] for i in indices]
        
        if self.min_value is not None and self.max_value is not None:
            # Sample from range
            if self.param_type == 'int':
                return list(np.random.randint(self.min_value, self.max_value + 1, size=size))
            elif self.param_type == 'float':
                return list(np.random.uniform(self.min_value, self.max_value, size=size))
            elif self.param_type == 'bool':
                return list(np.random.choice([True, False], size=size))
        
        # Default case
        return [None] * size
    
    def get_grid_points(self, num_points: int = 10) -> List[Any]:
        """
        Get grid points for parameter space
        
        Args:
            num_points: Number of points for range parameters
            
        Returns:
            List of grid points
        """
        if self.values:
            # Use discrete values directly
            return self.values
        
        if self.min_value is not None and self.max_value is not None:
            # Generate grid points from range
            if self.step is not None:
                # Use specified step
                if self.param_type == 'int':
                    return list(range(self.min_value, self.max_value + 1, self.step))
                elif self.param_type == 'float':
                    return list(np.arange(self.min_value, self.max_value + self.step/2, self.step))
            else:
                # Generate evenly spaced points
                if self.param_type == 'int':
                    return list(np.linspace(self.min_value, self.max_value, num_points, dtype=int))
                elif self.param_type == 'float':
                    return list(np.linspace(self.min_value, self.max_value, num_points))
            
            if self.param_type == 'bool':
                return [True, False]
        
        # Default case
        return []

class ComponentOptimizer:
    """Base class for component optimization."""
    
    def __init__(self, 
                component_type: ComponentType,
                component_class: str,
                base_parameters: Dict[str, Any] = None,
                data_provider: Callable = None,
                evaluation_func: Callable = None,
                max_evaluations: int = 100,
                n_jobs: int = -1,
                random_seed: int = 42):
        """
        Initialize component optimizer
        
        Args:
            component_type: Type of component to optimize
            component_class: Class name of component
            base_parameters: Base parameters for component
            data_provider: Function to provide data for evaluation
            evaluation_func: Function to evaluate component performance
            max_evaluations: Maximum number of evaluations
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_seed: Random seed for reproducibility
        """
        self.component_type = component_type
        self.component_class = component_class
        self.base_parameters = base_parameters or {}
        self.data_provider = data_provider
        self.evaluation_func = evaluation_func
        self.max_evaluations = max_evaluations
        self.parameters_space = {}
        
        # Set number of jobs
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Get component registry
        self.registry = get_component_registry()
        
        # Optimization results
        self.results = []
    
    def add_parameter_space(self, param_space: OptimizationSpace) -> None:
        """
        Add parameter space for optimization
        
        Args:
            param_space: Parameter space
        """
        self.parameters_space[param_space.param_name] = param_space
    
    def get_parameters_to_optimize(self) -> List[str]:
        """
        Get list of parameters to optimize
        
        Returns:
            List of parameter names
        """
        return list(self.parameters_space.keys())
    
    def _create_component(self, parameters: Dict[str, Any]) -> StrategyComponent:
        """
        Create component with given parameters
        
        Args:
            parameters: Component parameters
            
        Returns:
            Component instance
        """
        # Combine base parameters with optimization parameters
        combined_params = {**self.base_parameters, **parameters}
        
        # Create component
        component = self.registry.create_component(
            self.component_type, 
            self.component_class, 
            combined_params
        )
        
        return component
    
    def evaluate_parameters(self, parameters: Dict[str, Any]) -> float:
        """
        Evaluate component performance with given parameters
        
        Args:
            parameters: Component parameters
            
        Returns:
            Performance score
        """
        if not self.evaluation_func:
            raise ValueError("Evaluation function not specified")
        
        # Create component
        component = self._create_component(parameters)
        
        # Get data for evaluation
        if self.data_provider:
            data = self.data_provider()
        else:
            data = {}
        
        # Evaluate component
        try:
            score = self.evaluation_func(component, data)
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            score = float('-inf')
        
        # Store result
        self.results.append({
            'parameters': parameters,
            'score': score
        })
        
        return score
    
    def _log_progress(self, iteration: int, best_score: float, best_params: Dict[str, Any]) -> None:
        """
        Log optimization progress
        
        Args:
            iteration: Current iteration
            best_score: Best score so far
            best_params: Best parameters so far
        """
        logger.info(f"Iteration {iteration}/{self.max_evaluations}, Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get best parameters from optimization
        
        Returns:
            Best parameters
        """
        if not self.results:
            return {}
        
        # Find best result
        best_result = max(self.results, key=lambda x: x['score'])
        
        return best_result['parameters']
    
    def get_optimization_results(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get optimization results as DataFrame
        
        Args:
            top_n: Number of top results to include
            
        Returns:
            DataFrame of results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Sort by score
        df = df.sort_values('score', ascending=False)
        
        # Limit to top N
        if top_n > 0:
            df = df.head(top_n)
        
        # Explode parameters dictionary
        params_df = pd.json_normalize(df['parameters'])
        
        # Combine with scores
        result_df = pd.concat([params_df, df['score']], axis=1)
        
        return result_df
    
    def save_results(self, file_path: str) -> None:
        """
        Save optimization results to file
        
        Args:
            file_path: Path to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Get results as DataFrame
        df = self.get_optimization_results(top_n=-1)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization (to be implemented by subclasses)
        
        Returns:
            Best parameters
        """
        raise NotImplementedError("Optimize method must be implemented by subclasses")

class GridSearchOptimizer(ComponentOptimizer):
    """Grid search optimization for component parameters."""
    
    def __init__(self, 
                component_type: ComponentType,
                component_class: str,
                base_parameters: Dict[str, Any] = None,
                data_provider: Callable = None,
                evaluation_func: Callable = None,
                max_evaluations: int = 100,
                n_jobs: int = -1,
                random_seed: int = 42):
        """Initialize grid search optimizer."""
        super().__init__(
            component_type=component_type,
            component_class=component_class,
            base_parameters=base_parameters,
            data_provider=data_provider,
            evaluation_func=evaluation_func,
            max_evaluations=max_evaluations,
            n_jobs=n_jobs,
            random_seed=random_seed
        )
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run grid search optimization
        
        Returns:
            Best parameters
        """
        logger.info(f"Starting grid search optimization for {self.component_class}")
        
        # Get parameter grids
        param_grids = {}
        for param_name, param_space in self.parameters_space.items():
            param_grids[param_name] = param_space.get_grid_points()
        
        # Generate parameter combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        logger.info(f"Total combinations: {total_combinations}")
        
        # Limit combinations if needed
        if total_combinations > self.max_evaluations:
            logger.warning(
                f"Total combinations ({total_combinations}) exceed max evaluations "
                f"({self.max_evaluations}). Using random subset."
            )
            
            # Sample combinations randomly
            all_combinations = list(itertools.product(*param_values))
            indices = np.random.choice(
                len(all_combinations), 
                size=self.max_evaluations, 
                replace=False
            )
            combinations = [all_combinations[i] for i in indices]
        else:
            # Use all combinations
            combinations = itertools.product(*param_values)
        
        # Convert combinations to parameter dictionaries
        param_dicts = []
        for combination in combinations:
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = combination[i]
            param_dicts.append(param_dict)
        
        # Evaluate parameters in parallel
        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                scores = pool.map(self.evaluate_parameters, param_dicts)
        else:
            scores = []
            for i, params in enumerate(param_dicts):
                score = self.evaluate_parameters(params)
                scores.append(score)
                
                # Log progress
                if (i + 1) % 10 == 0 or i == len(param_dicts) - 1:
                    best_idx = np.argmax(scores[:i+1])
                    self._log_progress(i + 1, scores[best_idx], param_dicts[best_idx])
        
        # Get best parameters
        best_params = self.get_best_parameters()
        
        logger.info(f"Grid search optimization complete. Best parameters: {best_params}")
        
        return best_params

class RandomSearchOptimizer(ComponentOptimizer):
    """Random search optimization for component parameters."""
    
    def __init__(self, 
                component_type: ComponentType,
                component_class: str,
                base_parameters: Dict[str, Any] = None,
                data_provider: Callable = None,
                evaluation_func: Callable = None,
                max_evaluations: int = 100,
                n_jobs: int = -1,
                random_seed: int = 42):
        """Initialize random search optimizer."""
        super().__init__(
            component_type=component_type,
            component_class=component_class,
            base_parameters=base_parameters,
            data_provider=data_provider,
            evaluation_func=evaluation_func,
            max_evaluations=max_evaluations,
            n_jobs=n_jobs,
            random_seed=random_seed
        )
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run random search optimization
        
        Returns:
            Best parameters
        """
        logger.info(f"Starting random search optimization for {self.component_class}")
        
        # Generate random parameter combinations
        param_dicts = []
        
        for _ in range(self.max_evaluations):
            param_dict = {}
            for param_name, param_space in self.parameters_space.items():
                param_dict[param_name] = param_space.sample(1)[0]
            param_dicts.append(param_dict)
        
        # Evaluate parameters in parallel
        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                scores = pool.map(self.evaluate_parameters, param_dicts)
        else:
            scores = []
            for i, params in enumerate(param_dicts):
                score = self.evaluate_parameters(params)
                scores.append(score)
                
                # Log progress
                if (i + 1) % 10 == 0 or i == len(param_dicts) - 1:
                    best_idx = np.argmax(scores[:i+1])
                    self._log_progress(i + 1, scores[best_idx], param_dicts[best_idx])
        
        # Get best parameters
        best_params = self.get_best_parameters()
        
        logger.info(f"Random search optimization complete. Best parameters: {best_params}")
        
        return best_params

# Add Bayesian optimization if scikit-optimize is available
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    
    class BayesianOptimizer(ComponentOptimizer):
        """Bayesian optimization for component parameters using scikit-optimize."""
        
        def __init__(self, 
                    component_type: ComponentType,
                    component_class: str,
                    base_parameters: Dict[str, Any] = None,
                    data_provider: Callable = None,
                    evaluation_func: Callable = None,
                    max_evaluations: int = 50,
                    n_jobs: int = 1,  # Bayesian opt is sequential
                    random_seed: int = 42):
            """Initialize Bayesian optimizer."""
            super().__init__(
                component_type=component_type,
                component_class=component_class,
                base_parameters=base_parameters,
                data_provider=data_provider,
                evaluation_func=evaluation_func,
                max_evaluations=max_evaluations,
                n_jobs=n_jobs,
                random_seed=random_seed
            )
        
        def _create_search_space(self) -> Tuple[List[Any], List[str]]:
            """
            Create search space for scikit-optimize
            
            Returns:
                Tuple of (search space, parameter names)
            """
            search_space = []
            param_names = []
            
            for param_name, param_space in self.parameters_space.items():
                param_names.append(param_name)
                
                if param_space.values:
                    # Categorical parameter
                    search_space.append(Categorical(param_space.values, name=param_name))
                elif param_space.min_value is not None and param_space.max_value is not None:
                    # Continuous or integer parameter
                    if param_space.param_type == 'int':
                        search_space.append(
                            Integer(param_space.min_value, param_space.max_value, name=param_name)
                        )
                    elif param_space.param_type == 'float':
                        search_space.append(
                            Real(param_space.min_value, param_space.max_value, name=param_name)
                        )
                    elif param_space.param_type == 'bool':
                        search_space.append(Categorical([True, False], name=param_name))
            
            return search_space, param_names
        
        def _objective_function(self, params_list: List[Any]) -> float:
            """
            Objective function for Bayesian optimization
            
            Args:
                params_list: List of parameter values
                
            Returns:
                Negative score (minimization)
            """
            # Convert list to dictionary
            params = {}
            for i, param_name in enumerate(self.param_names):
                params[param_name] = params_list[i]
            
            # Evaluate parameters
            score = self.evaluate_parameters(params)
            
            # Return negative score (for minimization)
            return -score
        
        def optimize(self) -> Dict[str, Any]:
            """
            Run Bayesian optimization
            
            Returns:
                Best parameters
            """
            logger.info(f"Starting Bayesian optimization for {self.component_class}")
            
            # Create search space
            search_space, self.param_names = self._create_search_space()
            
            if not search_space:
                logger.warning("No parameters to optimize")
                return {}
            
            # Run optimization
            result = gp_minimize(
                self._objective_function, 
                search_space, 
                n_calls=self.max_evaluations,
                random_state=self.random_seed,
                verbose=True
            )
            
            # Convert result to parameters dictionary
            best_params = {}
            for i, param_name in enumerate(self.param_names):
                best_params[param_name] = result.x[i]
            
            logger.info(f"Bayesian optimization complete. Best parameters: {best_params}")
            
            return best_params
except ImportError:
    logger.warning("scikit-optimize not available. Bayesian optimization not supported.")

# Try to import genetic algorithm optimization if DEAP is available
try:
    import deap
    from deap import base, creator, tools, algorithms
    
    class GeneticOptimizer(ComponentOptimizer):
        """Genetic algorithm optimization for component parameters using DEAP."""
        
        def __init__(self, 
                  component_type: ComponentType,
                  component_class: str,
                  base_parameters: Dict[str, Any] = None,
                  data_provider: Callable = None,
                  evaluation_func: Callable = None,
                  max_evaluations: int = 100,
                  population_size: int = 20,
                  n_generations: int = 5,
                  crossover_prob: float = 0.5,
                  mutation_prob: float = 0.2,
                  n_jobs: int = -1,
                  random_seed: int = 42):
            """Initialize genetic algorithm optimizer."""
            super().__init__(
                component_type=component_type,
                component_class=component_class,
                base_parameters=base_parameters,
                data_provider=data_provider,
                evaluation_func=evaluation_func,
                max_evaluations=max_evaluations,
                n_jobs=n_jobs,
                random_seed=random_seed
            )
            self.population_size = min(population_size, max_evaluations)
            self.n_generations = n_generations
            self.crossover_prob = crossover_prob
            self.mutation_prob = mutation_prob
        
        def optimize(self) -> Dict[str, Any]:
            """
            Run genetic algorithm optimization
            
            Returns:
                Best parameters
            """
            # Not implemented yet - placeholder for future development
            logger.warning("Genetic algorithm optimization not implemented yet")
            return {}
except ImportError:
    logger.warning("DEAP not available. Genetic algorithm optimization not supported.")
