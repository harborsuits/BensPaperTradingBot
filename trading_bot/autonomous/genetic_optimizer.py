#!/usr/bin/env python3
"""
Genetic Algorithm Optimizer

This module implements genetic algorithm optimization for strategy parameters,
building on our existing optimization framework. Genetic algorithms are particularly
well-suited for complex parameter spaces where gradient-based methods might struggle.

Key features:
1. Population-based search with crossover and mutation
2. Tournament selection to maintain diversity
3. Elitism to preserve best solutions
4. Adaptive mutation rates based on population diversity
5. Integration with our parameter space representation
"""

import os
import json
import logging
import time
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime
from enum import Enum
from copy import deepcopy

# Import parameter space from existing optimizer
from trading_bot.autonomous.bayesian_optimizer import (
    ParameterSpace, ParameterType, OptimizationDirection
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectionMethod(str, Enum):
    """Selection methods for genetic algorithms."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


class CrossoverMethod(str, Enum):
    """Crossover methods for genetic algorithms."""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"


class GeneticOptimizer:
    """
    Genetic algorithm optimization for strategy parameters.
    
    This class implements genetic algorithm optimization using a population-based
    approach with selection, crossover, and mutation operations. It is well-suited
    for complex, non-convex parameter spaces.
    """
    
    def __init__(
        self, 
        parameter_space: ParameterSpace,
        population_size: int = 50,
        elite_size: int = 5,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        selection_method: Union[SelectionMethod, str] = SelectionMethod.TOURNAMENT,
        crossover_method: Union[CrossoverMethod, str] = CrossoverMethod.UNIFORM,
        minimize: bool = False,
        adaptive_mutation: bool = True
    ):
        """
        Initialize the genetic optimizer.
        
        Args:
            parameter_space: Parameter space to optimize
            population_size: Size of the population
            elite_size: Number of elite individuals to preserve
            tournament_size: Size of tournament for selection
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            selection_method: Method for selecting parents
            crossover_method: Method for crossover operation
            minimize: Whether to minimize (True) or maximize (False)
            adaptive_mutation: Whether to adapt mutation rate based on diversity
        """
        self.parameter_space = parameter_space
        self.param_names = parameter_space.param_names
        self.param_types = parameter_space.param_types
        self.bounds = parameter_space.bounds
        self.categories = parameter_space.categories
        
        # Genetic algorithm parameters
        self.population_size = population_size
        self.elite_size = min(elite_size, population_size // 2)
        self.tournament_size = min(tournament_size, population_size // 2)
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.minimize = minimize
        self.adaptive_mutation = adaptive_mutation
        
        # Convert string enums to enum values if needed
        if isinstance(selection_method, str):
            self.selection_method = SelectionMethod(selection_method)
        else:
            self.selection_method = selection_method
            
        if isinstance(crossover_method, str):
            self.crossover_method = CrossoverMethod(crossover_method)
        else:
            self.crossover_method = crossover_method
        
        # Population and fitness tracking
        self.population = []
        self.fitness_values = []
        self.best_individual = None
        self.best_fitness = float('-inf') if not minimize else float('inf')
        
        # History for tracking progress
        self.parameters_history = []
        self.generation_stats = []
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with random individuals."""
        # Generate random parameter sets
        self.population = self.parameter_space.get_random_parameters(self.population_size)
        
        # Initialize fitness values with placeholders
        self.fitness_values = [None] * self.population_size
    
    def evaluate_population(self, objective_function: Callable[[Dict[str, Any]], float]):
        """
        Evaluate fitness for all individuals in the population.
        
        Args:
            objective_function: Function to evaluate fitness
        """
        for i, individual in enumerate(self.population):
            if self.fitness_values[i] is None:  # Only evaluate if not already evaluated
                try:
                    fitness = objective_function(individual)
                except Exception as e:
                    logger.error(f"Error evaluating individual: {str(e)}")
                    # Assign a penalty value
                    fitness = float('-inf') if not self.minimize else float('inf')
                
                self.fitness_values[i] = fitness
                
                # Track best individual
                if (not self.minimize and fitness > self.best_fitness) or \
                   (self.minimize and fitness < self.best_fitness):
                    self.best_fitness = fitness
                    self.best_individual = deepcopy(individual)
                    
                    # Add to history
                    self.parameters_history.append({
                        'parameters': individual,
                        'value': fitness,
                        'timestamp': datetime.now().isoformat()
                    })
    
    def _select_parents(self) -> List[Dict[str, Any]]:
        """
        Select parents for reproduction.
        
        Returns:
            List of selected parents
        """
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection()
        elif self.selection_method == SelectionMethod.RANK:
            return self._rank_selection()
        else:
            return self._tournament_selection()  # Default
    
    def _tournament_selection(self) -> List[Dict[str, Any]]:
        """
        Tournament selection method.
        
        Returns:
            Selected parents
        """
        parents = []
        
        # Keep elite individuals
        if self.elite_size > 0:
            # Sort population by fitness
            sorted_indices = np.argsort(self.fitness_values)
            if not self.minimize:
                sorted_indices = sorted_indices[::-1]  # Reverse for maximization
            
            # Add elite individuals
            for i in range(self.elite_size):
                parents.append(deepcopy(self.population[sorted_indices[i]]))
        
        # Fill the rest with tournament selection
        while len(parents) < self.population_size:
            # Select tournament participants
            tournament_indices = random.sample(range(self.population_size), self.tournament_size)
            
            # Find winner (best fitness in tournament)
            if self.minimize:
                winner_idx = min(tournament_indices, key=lambda i: self.fitness_values[i])
            else:
                winner_idx = max(tournament_indices, key=lambda i: self.fitness_values[i])
            
            # Add winner to parents
            parents.append(deepcopy(self.population[winner_idx]))
        
        return parents
    
    def _roulette_selection(self) -> List[Dict[str, Any]]:
        """
        Roulette wheel selection method.
        
        Returns:
            Selected parents
        """
        parents = []
        
        # Keep elite individuals
        if self.elite_size > 0:
            # Sort population by fitness
            sorted_indices = np.argsort(self.fitness_values)
            if not self.minimize:
                sorted_indices = sorted_indices[::-1]  # Reverse for maximization
            
            # Add elite individuals
            for i in range(self.elite_size):
                parents.append(deepcopy(self.population[sorted_indices[i]]))
        
        # Adjust fitness for minimization and negative values
        adjusted_fitness = np.array(self.fitness_values)
        
        if self.minimize:
            # For minimization, invert fitness (smaller values become larger)
            max_fitness = max(adjusted_fitness)
            adjusted_fitness = max_fitness - adjusted_fitness
        
        # Handle negative fitness values
        min_fitness = min(adjusted_fitness)
        if min_fitness < 0:
            adjusted_fitness = adjusted_fitness - min_fitness + 1e-6
        
        # Calculate selection probabilities
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            # If all fitness values are zero, use uniform selection
            probabilities = np.ones(len(adjusted_fitness)) / len(adjusted_fitness)
        else:
            probabilities = adjusted_fitness / total_fitness
        
        # Fill the rest with roulette selection
        while len(parents) < self.population_size:
            # Select parent based on fitness probability
            parent_idx = np.random.choice(range(self.population_size), p=probabilities)
            parents.append(deepcopy(self.population[parent_idx]))
        
        return parents
    
    def _rank_selection(self) -> List[Dict[str, Any]]:
        """
        Rank-based selection method.
        
        Returns:
            Selected parents
        """
        parents = []
        
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_values)
        if not self.minimize:
            sorted_indices = sorted_indices[::-1]  # Reverse for maximization
        
        # Assign ranks (better individuals get higher ranks)
        ranks = np.arange(1, self.population_size + 1)
        
        # Calculate selection probabilities based on rank
        total_rank = sum(ranks)
        probabilities = ranks / total_rank
        
        # Apply probabilities to sorted indices
        rank_probabilities = np.zeros(self.population_size)
        for i, idx in enumerate(sorted_indices):
            rank_probabilities[idx] = probabilities[i]
        
        # Keep elite individuals
        if self.elite_size > 0:
            # Add elite individuals
            for i in range(self.elite_size):
                parents.append(deepcopy(self.population[sorted_indices[i]]))
        
        # Fill the rest with rank selection
        while len(parents) < self.population_size:
            # Select parent based on rank probability
            parent_idx = np.random.choice(range(self.population_size), p=rank_probabilities)
            parents.append(deepcopy(self.population[parent_idx]))
        
        return parents
    
    def _crossover(self, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform crossover to create offspring.
        
        Args:
            parents: List of parent individuals
            
        Returns:
            List of offspring
        """
        offspring = []
        
        # Keep elite parents unchanged
        for i in range(self.elite_size):
            offspring.append(parents[i])
        
        # Create offspring through crossover
        while len(offspring) < self.population_size:
            # Select two parents
            parent1_idx = random.randint(0, len(parents) - 1)
            parent2_idx = random.randint(0, len(parents) - 1)
            
            # Ensure parents are different
            while parent2_idx == parent1_idx:
                parent2_idx = random.randint(0, len(parents) - 1)
            
            parent1 = parents[parent1_idx]
            parent2 = parents[parent2_idx]
            
            # Perform crossover with probability
            if random.random() < self.crossover_rate:
                if self.crossover_method == CrossoverMethod.UNIFORM:
                    child = self._uniform_crossover(parent1, parent2)
                elif self.crossover_method == CrossoverMethod.SINGLE_POINT:
                    child = self._single_point_crossover(parent1, parent2)
                elif self.crossover_method == CrossoverMethod.TWO_POINT:
                    child = self._two_point_crossover(parent1, parent2)
                else:
                    child = self._uniform_crossover(parent1, parent2)  # Default
            else:
                # No crossover, just copy a parent
                child = deepcopy(parent1 if random.random() < 0.5 else parent2)
            
            offspring.append(child)
        
        return offspring
    
    def _uniform_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child individual
        """
        child = {}
        
        for param_name in self.param_names:
            # For each parameter, randomly choose from either parent
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _single_point_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child individual
        """
        child = {}
        
        # Choose crossover point
        crossover_point = random.randint(1, len(self.param_names) - 1)
        
        for i, param_name in enumerate(self.param_names):
            if i < crossover_point:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _two_point_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Two-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child individual
        """
        child = {}
        
        # Choose crossover points
        point1 = random.randint(1, len(self.param_names) - 2)
        point2 = random.randint(point1 + 1, len(self.param_names) - 1)
        
        for i, param_name in enumerate(self.param_names):
            if i < point1 or i >= point2:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _mutate(self, offspring: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Mutate offspring.
        
        Args:
            offspring: List of offspring to mutate
            
        Returns:
            Mutated offspring
        """
        # Adjust mutation rate if adaptive
        if self.adaptive_mutation:
            self._adjust_mutation_rate()
        
        # Keep elite individuals unchanged
        for i in range(self.elite_size, len(offspring)):
            individual = offspring[i]
            
            for param_name in self.param_names:
                # Mutate each parameter with probability
                if random.random() < self.mutation_rate:
                    param_type = self.param_types[param_name]
                    
                    if param_type == ParameterType.REAL:
                        # Mutate real parameter
                        lower, upper = self.bounds[param_name]
                        sigma = (upper - lower) * 0.1  # 10% of range
                        new_value = individual[param_name] + random.gauss(0, sigma)
                        individual[param_name] = max(lower, min(upper, new_value))
                        
                    elif param_type == ParameterType.INTEGER:
                        # Mutate integer parameter
                        lower, upper = self.bounds[param_name]
                        sigma = max(1, (upper - lower) * 0.1)  # 10% of range
                        offset = int(random.gauss(0, sigma))
                        new_value = individual[param_name] + offset
                        individual[param_name] = max(lower, min(upper, new_value))
                        
                    elif param_type == ParameterType.CATEGORICAL:
                        # Mutate categorical parameter
                        categories = self.categories[param_name]
                        if len(categories) > 1:
                            # Select a different category
                            current_idx = categories.index(individual[param_name])
                            new_idx = current_idx
                            while new_idx == current_idx:
                                new_idx = random.randint(0, len(categories) - 1)
                            individual[param_name] = categories[new_idx]
                            
                    elif param_type == ParameterType.BOOLEAN:
                        # Flip boolean parameter
                        individual[param_name] = not individual[param_name]
        
        return offspring
    
    def _adjust_mutation_rate(self):
        """Adjust mutation rate based on population diversity."""
        # Calculate diversity as average normalized distance between individuals
        diversity = self._calculate_diversity()
        
        # Scale mutation rate based on diversity
        # Higher diversity -> lower mutation rate
        # Lower diversity -> higher mutation rate
        diversity_factor = 1.0 - min(1.0, max(0.0, diversity))
        
        # Apply scaling, allowing mutation rate to vary between 50% and 150% of base rate
        self.mutation_rate = self.base_mutation_rate * (0.5 + diversity_factor)
        
        logger.debug(f"Diversity: {diversity:.4f}, Adjusted mutation rate: {self.mutation_rate:.4f}")
    
    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity.
        
        Returns:
            Diversity measure between 0 and 1
        """
        if len(self.population) <= 1:
            return 0.0
        
        # Calculate normalized distances for each parameter
        normalized_distances = []
        
        for param_name in self.param_names:
            param_type = self.param_types[param_name]
            
            if param_type == ParameterType.REAL or param_type == ParameterType.INTEGER:
                # For numerical parameters, calculate range
                values = [ind[param_name] for ind in self.population]
                lower, upper = self.bounds[param_name]
                range_size = upper - lower
                
                if range_size > 0:
                    # Calculate normalized standard deviation
                    std_dev = np.std(values)
                    normalized_std = std_dev / range_size
                    normalized_distances.append(normalized_std)
                
            elif param_type == ParameterType.CATEGORICAL:
                # For categorical parameters, calculate entropy
                categories = self.categories[param_name]
                values = [ind[param_name] for ind in self.population]
                
                # Count occurrences of each category
                counts = [values.count(cat) for cat in categories]
                probabilities = [count / len(values) for count in counts]
                
                # Calculate entropy (higher is more diverse)
                entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)
                max_entropy = np.log(len(categories))
                
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                    normalized_distances.append(normalized_entropy)
                
            elif param_type == ParameterType.BOOLEAN:
                # For boolean parameters, calculate proportion of true values
                values = [ind[param_name] for ind in self.population]
                p_true = sum(values) / len(values)
                
                # Maximum diversity is p_true = 0.5
                boolean_diversity = 1.0 - abs(p_true - 0.5) * 2.0
                normalized_distances.append(boolean_diversity)
        
        # Average all normalized distances
        if normalized_distances:
            return sum(normalized_distances) / len(normalized_distances)
        else:
            return 0.0
    
    def evolve(self):
        """Perform one generation of evolution."""
        # Select parents
        parents = self._select_parents()
        
        # Create offspring through crossover
        offspring = self._crossover(parents)
        
        # Mutate offspring
        offspring = self._mutate(offspring)
        
        # Replace population with offspring
        self.population = offspring
        
        # Reset fitness values for new population (except elite individuals)
        preserved_fitness = self.fitness_values[:self.elite_size] if self.elite_size > 0 else []
        self.fitness_values = preserved_fitness + [None] * (self.population_size - self.elite_size)
    
    def get_best_parameters(self) -> Tuple[Dict[str, Any], float]:
        """
        Get the best parameters found so far.
        
        Returns:
            Tuple of (best_parameters, best_value)
        """
        if not self.parameters_history:
            # No evaluations yet, return default parameters
            return self.parameter_space.get_default_parameters(), 0.0
        
        return self.best_individual, self.best_fitness
    
    def optimize(
        self, 
        objective_function: Callable[[Dict[str, Any]], float],
        n_generations: int = 20,
        callback: Optional[Callable[[Dict[str, Any], float, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the optimization process for a given objective function.
        
        Args:
            objective_function: Function that evaluates parameters
            n_generations: Number of generations to run
            callback: Optional callback function after each generation
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        for generation in range(n_generations):
            # Evaluate current population
            self.evaluate_population(objective_function)
            
            # Calculate statistics for this generation
            mean_fitness = np.mean(self.fitness_values)
            best_idx = np.argmax(self.fitness_values) if not self.minimize else np.argmin(self.fitness_values)
            best_fitness = self.fitness_values[best_idx]
            
            # Record statistics
            self.generation_stats.append({
                'generation': generation,
                'mean_fitness': float(mean_fitness),
                'best_fitness': float(best_fitness),
                'diversity': float(self._calculate_diversity()),
                'mutation_rate': float(self.mutation_rate)
            })
            
            # Log progress
            logger.info(
                f"Generation {generation+1}/{n_generations}, "
                f"Best: {best_fitness:.4f}, Mean: {mean_fitness:.4f}"
            )
            
            # Call callback if provided
            if callback:
                callback(self.population[best_idx], best_fitness, generation)
            
            # Evolve population (except for last generation)
            if generation < n_generations - 1:
                self.evolve()
        
        # Ensure final population is evaluated
        self.evaluate_population(objective_function)
        
        # Get best result
        best_params, best_fitness = self.get_best_parameters()
        
        optimization_time = time.time() - start_time
        
        return {
            'best_parameters': best_params,
            'best_value': best_fitness,
            'n_generations': n_generations,
            'optimization_time': optimization_time,
            'all_parameters': self.parameters_history,
            'generation_stats': self.generation_stats,
            'parameter_space': self.parameter_space.get_parameters_dict()
        }


if __name__ == "__main__":
    # Example usage
    def example_objective(params):
        # Rosenbrock function (minimization)
        x = params['x']
        y = params['y']
        return -((1 - x)**2 + 100 * (y - x**2)**2)
    
    # Create parameter space
    param_space = ParameterSpace()
    param_space.add_real_parameter('x', -5.0, 5.0)
    param_space.add_real_parameter('y', -5.0, 5.0)
    
    # Create optimizer
    optimizer = GeneticOptimizer(
        parameter_space=param_space,
        population_size=50,
        elite_size=5,
        tournament_size=3,
        mutation_rate=0.1,
        minimize=False  # Maximize negative Rosenbrock
    )
    
    # Run optimization
    results = optimizer.optimize(example_objective, n_generations=20)
    
    # Print results
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best value: {results['best_value']}")
    print(f"Optimization time: {results['optimization_time']:.2f} seconds")
