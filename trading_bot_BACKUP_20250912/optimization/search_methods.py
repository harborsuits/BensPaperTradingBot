import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional, Callable, Iterator
import logging
import abc
import random
from enum import Enum
from collections import deque

from .parameter_space import ParameterSpace

# Optional imports for advanced methods
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKLEARN_OPT_AVAILABLE = True
except ImportError:
    SKLEARN_OPT_AVAILABLE = False

logger = logging.getLogger(__name__)

class SearchMethod(abc.ABC):
    """Base class for parameter search methods"""
    
    def __init__(self, parameter_space: ParameterSpace):
        """
        Initialize search method
        
        Args:
            parameter_space: Parameter space to search
        """
        self.parameter_space = parameter_space
        self.evaluated_params = []
        self.objective_values = []
        self.best_params = None
        self.best_objective = float('inf')  # Assuming minimization
        self.minimizing = True  # Whether we're minimizing or maximizing
    
    @abc.abstractmethod
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next parameter set to evaluate
        
        Returns:
            Dictionary of parameter values
        """
        pass
    
    def register_result(self, params: Dict[str, Any], objective_value: float) -> None:
        """
        Register evaluation result
        
        Args:
            params: Parameter values
            objective_value: Objective function value
        """
        self.evaluated_params.append(params.copy())
        self.objective_values.append(objective_value)
        
        # Update best result (minimization)
        if self.minimizing and objective_value < self.best_objective:
            self.best_objective = objective_value
            self.best_params = params.copy()
        # Update best result (maximization)
        elif not self.minimizing and objective_value > self.best_objective:
            self.best_objective = objective_value
            self.best_params = params.copy()
    
    def set_max_objective(self) -> None:
        """Set to maximize objective (higher is better)"""
        self.minimizing = False
        self.best_objective = float('-inf')
    
    def set_min_objective(self) -> None:
        """Set to minimize objective (lower is better)"""
        self.minimizing = True
        self.best_objective = float('inf')
    
    def get_results_df(self) -> pd.DataFrame:
        """
        Get results as a DataFrame
        
        Returns:
            DataFrame of parameter values and objective values
        """
        if not self.evaluated_params:
            return pd.DataFrame()
        
        # Convert list of parameter dictionaries to DataFrame
        results_df = pd.DataFrame(self.evaluated_params)
        results_df['objective'] = self.objective_values
        
        # Sort by objective (ascending for minimization, descending for maximization)
        ascending = self.minimizing
        results_df = results_df.sort_values('objective', ascending=ascending)
        
        return results_df
    
    def get_best_params(self) -> Tuple[Dict[str, Any], float]:
        """
        Get best parameters found
        
        Returns:
            Tuple of (best_params, best_objective)
        """
        return self.best_params, self.best_objective


class GridSearch(SearchMethod):
    """Grid search for parameter optimization"""
    
    def __init__(self, parameter_space: ParameterSpace, num_points: int = 10):
        """
        Initialize grid search
        
        Args:
            parameter_space: Parameter space to search
            num_points: Number of points to sample for range/integer parameters
        """
        super().__init__(parameter_space)
        self.num_points = num_points
        self.iterator = parameter_space.grid_search_iterator(num_points)
        self.has_next = True
    
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next parameter set to evaluate
        
        Returns:
            Dictionary of parameter values
        """
        try:
            params = next(self.iterator)
            return params
        except StopIteration:
            self.has_next = False
            # Return best params if we've evaluated at least one parameter set
            if self.best_params is not None:
                return self.best_params
            # Otherwise, return default params
            return self.parameter_space.get_default_params()


class RandomSearch(SearchMethod):
    """Random search for parameter optimization"""
    
    def __init__(self, parameter_space: ParameterSpace, num_iterations: int = 100):
        """
        Initialize random search
        
        Args:
            parameter_space: Parameter space to search
            num_iterations: Number of random samples to evaluate
        """
        super().__init__(parameter_space)
        self.num_iterations = num_iterations
        self.current_iteration = 0
    
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next parameter set to evaluate
        
        Returns:
            Dictionary of parameter values
        """
        if self.current_iteration < self.num_iterations:
            self.current_iteration += 1
            return self.parameter_space.sample()
        else:
            # Return best params if we've evaluated at least one parameter set
            if self.best_params is not None:
                return self.best_params
            # Otherwise, return default params
            return self.parameter_space.get_default_params()


class BayesianOptimization(SearchMethod):
    """Bayesian optimization for parameter search"""
    
    def __init__(
        self, 
        parameter_space: ParameterSpace, 
        num_iterations: int = 50,
        initial_random_points: int = 10,
        n_random_starts: int = 10,
        random_state: int = None
    ):
        """
        Initialize Bayesian optimization
        
        Args:
            parameter_space: Parameter space to search
            num_iterations: Maximum number of iterations
            initial_random_points: Number of initial random evaluations
            n_random_starts: Number of random starts for optimization
            random_state: Random state for reproducibility
        """
        super().__init__(parameter_space)
        self.num_iterations = num_iterations
        self.initial_random_points = initial_random_points
        self.n_random_starts = n_random_starts
        self.random_state = random_state
        self.current_iteration = 0
        
        # Check if scikit-optimize is available
        if not SKLEARN_OPT_AVAILABLE:
            logger.warning("scikit-optimize not available, falling back to random search")
            self.fallback_to_random = True
            self.random_search = RandomSearch(parameter_space, num_iterations)
        else:
            self.fallback_to_random = False
            self._setup_skopt_space()
    
    def _setup_skopt_space(self) -> None:
        """Setup scikit-optimize search space"""
        if not SKLEARN_OPT_AVAILABLE:
            return
        
        # Create scikit-optimize search space
        self.skopt_space = []
        self.param_names = []
        
        for param in self.parameter_space.parameters:
            self.param_names.append(param.name)
            
            if param.param_type.value == "range":
                self.skopt_space.append(Real(param.values[0], param.values[1], name=param.name))
            elif param.param_type.value == "integer":
                self.skopt_space.append(Integer(param.values[0], param.values[1], name=param.name))
            elif param.param_type.value in ["discrete", "categorical", "boolean"]:
                self.skopt_space.append(Categorical(param.values, name=param.name))
    
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next parameter set to evaluate
        
        Returns:
            Dictionary of parameter values
        """
        if self.fallback_to_random:
            return self.random_search.suggest()
        
        # Do initial random exploration
        if self.current_iteration < self.initial_random_points:
            self.current_iteration += 1
            return self.parameter_space.sample()
        
        # We have enough data, use Bayesian optimization
        if self.current_iteration < self.num_iterations:
            self.current_iteration += 1
            
            # Use scikit-optimize to suggest next point
            if len(self.evaluated_params) >= self.initial_random_points:
                try:
                    # Convert previous evaluations to the format expected by skopt
                    X = []
                    y = []
                    
                    for params, objective in zip(self.evaluated_params, self.objective_values):
                        X.append([params[name] for name in self.param_names])
                        y.append(objective if self.minimizing else -objective)
                    
                    # Use gp_minimize to get next suggested point
                    res = gp_minimize(
                        lambda x: 0,  # Dummy objective function
                        self.skopt_space,
                        x0=X,
                        y0=y,
                        n_calls=1,
                        n_random_starts=0,
                        random_state=self.random_state
                    )
                    
                    # Convert result back to parameter dictionary
                    next_point = res.x_iters[-1]
                    return {name: value for name, value in zip(self.param_names, next_point)}
                except Exception as e:
                    logger.warning(f"Error in Bayesian optimization: {e}, falling back to random search")
                    return self.parameter_space.sample()
            else:
                return self.parameter_space.sample()
        else:
            # Return best params if we've evaluated at least one parameter set
            if self.best_params is not None:
                return self.best_params
            # Otherwise, return default params
            return self.parameter_space.get_default_params()
    
    def register_result(self, params: Dict[str, Any], objective_value: float) -> None:
        """
        Register evaluation result
        
        Args:
            params: Parameter values
            objective_value: Objective function value
        """
        super().register_result(params, objective_value)
        
        # If using fallback, also register with random search
        if self.fallback_to_random:
            self.random_search.register_result(params, objective_value)


class GeneticAlgorithm(SearchMethod):
    """Genetic algorithm for parameter optimization"""
    
    def __init__(
        self, 
        parameter_space: ParameterSpace, 
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_size: int = 2,
        tournament_size: int = 3,
        random_state: int = None
    ):
        """
        Initialize genetic algorithm
        
        Args:
            parameter_space: Parameter space to search
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of best individuals to keep unchanged
            tournament_size: Size of tournament for selection
            random_state: Random state for reproducibility
        """
        super().__init__(parameter_space)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.random_state = random_state
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        
        # Initialize state
        self.current_generation = 0
        self.current_individual = 0
        self.population = []
        self.fitness = []
        
        # Generate initial population
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Initialize random population"""
        self.population = [self.parameter_space.sample() for _ in range(self.population_size)]
        self.fitness = [float('inf') if self.minimizing else float('-inf')] * self.population_size
        self.evaluated = [False] * self.population_size
    
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next parameter set to evaluate
        
        Returns:
            Dictionary of parameter values
        """
        # If we haven't finished evaluating the current population, return the next individual
        if self.current_generation < self.generations:
            if self.current_individual < self.population_size:
                params = self.population[self.current_individual]
                self.current_individual += 1
                return params
            else:
                # We've evaluated the entire population, create the next generation
                self._create_next_generation()
                self.current_individual = 1  # Start from 1 because we return the 0th element below
                return self.population[0]
        
        # We've finished all generations, return the best individual
        if self.best_params is not None:
            return self.best_params
        else:
            return self.parameter_space.get_default_params()
    
    def register_result(self, params: Dict[str, Any], objective_value: float) -> None:
        """
        Register evaluation result
        
        Args:
            params: Parameter values
            objective_value: Objective function value
        """
        super().register_result(params, objective_value)
        
        # Update fitness of the corresponding individual
        # Find which individual in the population is closest to these params
        for i, individual in enumerate(self.population):
            if all(individual[key] == params[key] for key in params):
                self.fitness[i] = objective_value
                self.evaluated[i] = True
                break
    
    def _create_next_generation(self) -> None:
        """Create the next generation of individuals"""
        self.current_generation += 1
        
        # Create new population
        new_population = []
        
        # Add elite individuals
        elite_indices = self._get_elite_indices()
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Create rest of the population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1_idx = self._tournament_selection()
            parent2_idx = self._tournament_selection()
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Update population
        self.population = new_population
        self.fitness = [float('inf') if self.minimizing else float('-inf')] * self.population_size
        self.evaluated = [False] * self.population_size
    
    def _get_elite_indices(self) -> List[int]:
        """Get indices of elite individuals"""
        # Sort indices by fitness (ascending for minimization, descending for maximization)
        sorted_indices = sorted(range(len(self.fitness)), 
                               key=lambda i: self.fitness[i],
                               reverse=not self.minimizing)
        
        return sorted_indices[:self.elite_size]
    
    def _tournament_selection(self) -> int:
        """
        Select individual using tournament selection
        
        Returns:
            Index of selected individual
        """
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(len(self.population)), 
                                          min(self.tournament_size, len(self.population)))
        
        # Get their fitness values
        tournament_fitness = [self.fitness[i] for i in tournament_indices]
        
        # Select the best (min for minimization, max for maximization)
        if self.minimizing:
            best_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        else:
            best_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        
        return best_idx
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two children
        """
        child1 = {}
        child2 = {}
        
        # For each parameter, randomly choose from which parent to inherit
        for param_name in parent1:
            if random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate an individual
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        # For each parameter, randomly decide whether to mutate
        for param_name, param_value in individual.items():
            if random.random() < self.mutation_rate:
                # Replace with a random value
                param = self.parameter_space.get_parameter(param_name)
                mutated[param_name] = param.sample()
        
        return mutated 