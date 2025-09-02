#!/usr/bin/env python3
"""
Multi-Objective Optimizer

This module implements multi-objective optimization for strategy parameters,
building directly on our existing optimization framework. It uses Pareto-based
selection to optimize for multiple competing objectives simultaneously.

Key features:
1. Non-dominated sorting to identify Pareto-optimal solutions
2. Crowding distance calculation to maintain diversity
3. Multiple optimization strategies (NSGA-II and MOEA/D)
4. Support for weight preferences to prioritize certain objectives
5. Integration with our existing parameter space representation
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


class MultiObjectiveAlgorithm(str, Enum):
    """Multi-objective optimization algorithms."""
    NSGA_II = "nsga_ii"  # Non-dominated Sorting Genetic Algorithm II
    MOEA_D = "moea_d"    # Multi-objective Evolutionary Algorithm with Decomposition


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for strategy parameters.
    
    This class implements multi-objective optimization using Pareto-based methods
    to handle multiple competing objectives simultaneously. It is particularly
    useful for optimization where trade-offs between objectives must be considered.
    """
    
    def __init__(
        self, 
        parameter_space: ParameterSpace,
        objective_names: List[str],
        objective_directions: List[Union[OptimizationDirection, str]],
        weights: Optional[List[float]] = None,
        population_size: int = 50,
        algorithm: Union[MultiObjectiveAlgorithm, str] = MultiObjectiveAlgorithm.NSGA_II,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        adaptive_mutation: bool = True
    ):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            parameter_space: Parameter space to optimize
            objective_names: Names of the objectives to optimize
            objective_directions: Direction for each objective (minimize/maximize)
            weights: Optional priority weights for objectives (default: equal)
            population_size: Size of the population
            algorithm: Multi-objective algorithm to use
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            adaptive_mutation: Whether to adapt mutation rate based on diversity
        """
        self.parameter_space = parameter_space
        self.param_names = parameter_space.param_names
        self.param_types = parameter_space.param_types
        self.bounds = parameter_space.bounds
        self.categories = parameter_space.categories
        
        # Objective information
        self.objective_names = objective_names
        self.n_objectives = len(objective_names)
        
        # Validate and convert directions
        self.objective_directions = []
        for direction in objective_directions:
            if isinstance(direction, str):
                self.objective_directions.append(OptimizationDirection(direction))
            else:
                self.objective_directions.append(direction)
        
        # Initialize weights (default: equal weighting)
        if weights is None:
            self.weights = [1.0 / self.n_objectives] * self.n_objectives
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # Algorithm selection
        if isinstance(algorithm, str):
            self.algorithm = MultiObjectiveAlgorithm(algorithm)
        else:
            self.algorithm = algorithm
        
        # Optimization parameters
        self.population_size = max(population_size, self.n_objectives * 10)  # Ensure sufficient population
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.adaptive_mutation = adaptive_mutation
        
        # Population and fitness tracking
        self.population = []
        self.objective_values = []  # List of lists: [ind1:[obj1, obj2], ind2:[obj1, obj2], ...]
        self.pareto_fronts = []     # List of lists of indices, each list is a front
        self.crowding_distances = []
        
        # History for tracking progress
        self.parameters_history = []
        self.generation_stats = []
        
        # Track the Pareto front
        self.pareto_optimal = []    # List of dictionaries with parameters and objectives
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with random individuals."""
        # Generate random parameter sets
        self.population = self.parameter_space.get_random_parameters(self.population_size)
        
        # Initialize objective values with placeholders
        self.objective_values = [[None] * self.n_objectives for _ in range(self.population_size)]
    
    def evaluate_population(self, objective_function: Callable[[Dict[str, Any]], List[float]]):
        """
        Evaluate objectives for all individuals in the population.
        
        Args:
            objective_function: Function to evaluate objectives, must return list of values
        """
        for i, individual in enumerate(self.population):
            if None in self.objective_values[i]:  # Only evaluate if not already evaluated
                try:
                    values = objective_function(individual)
                    
                    # Ensure correct number of objectives
                    if len(values) != self.n_objectives:
                        raise ValueError(f"Expected {self.n_objectives} objectives, got {len(values)}")
                    
                    self.objective_values[i] = values
                    
                except Exception as e:
                    logger.error(f"Error evaluating individual: {str(e)}")
                    # Assign penalty values
                    penalties = []
                    for j, direction in enumerate(self.objective_directions):
                        if direction == OptimizationDirection.MAXIMIZE:
                            penalties.append(float('-inf'))
                        else:
                            penalties.append(float('inf'))
                    
                    self.objective_values[i] = penalties
                
                # Add to history
                history_entry = {
                    'parameters': deepcopy(individual),
                    'objectives': self.objective_values[i],
                    'timestamp': datetime.now().isoformat()
                }
                self.parameters_history.append(history_entry)
        
        # Update Pareto front
        self._update_pareto_front()
    
    def _dominates(self, values1: List[float], values2: List[float]) -> bool:
        """
        Check if values1 dominates values2.
        
        Args:
            values1: First set of objective values
            values2: Second set of objective values
            
        Returns:
            True if values1 dominates values2
        """
        better_in_any = False
        for i in range(self.n_objectives):
            # Check if worse in any objective
            if self.objective_directions[i] == OptimizationDirection.MAXIMIZE:
                if values1[i] < values2[i]:
                    return False
                elif values1[i] > values2[i]:
                    better_in_any = True
            else:  # MINIMIZE
                if values1[i] > values2[i]:
                    return False
                elif values1[i] < values2[i]:
                    better_in_any = True
        
        # Must be better in at least one objective to dominate
        return better_in_any
    
    def _non_dominated_sort(self) -> List[List[int]]:
        """
        Perform non-dominated sorting of the population.
        
        Returns:
            List of Pareto fronts, each containing indices of individuals
        """
        n = len(self.population)
        domination_count = [0] * n  # Number of individuals that dominate this one
        dominated_set = [[] for _ in range(n)]  # Set of individuals dominated by this one
        fronts = [[]]  # First front
        
        # For each individual
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Check if i dominates j
                if self._dominates(self.objective_values[i], self.objective_values[j]):
                    dominated_set[i].append(j)
                # Check if j dominates i
                elif self._dominates(self.objective_values[j], self.objective_values[i]):
                    domination_count[i] += 1
            
            # If no one dominates i, it belongs to the first front
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Find the rest of the fronts
        i = 0
        while fronts[i]:
            next_front = []
            
            # For each individual in the current front
            for j in fronts[i]:
                # For each individual dominated by j
                for k in dominated_set[j]:
                    domination_count[k] -= 1
                    
                    # If k is not dominated by anyone else, it belongs to the next front
                    if domination_count[k] == 0:
                        next_front.append(k)
            
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def _calculate_crowding_distance(self, front: List[int]) -> List[float]:
        """
        Calculate crowding distance for individuals in a front.
        
        Args:
            front: List of indices forming a Pareto front
            
        Returns:
            List of crowding distances
        """
        if not front:
            return []
        
        n = len(front)
        distances = [0.0] * n
        
        # For each objective
        for obj_idx in range(self.n_objectives):
            # Sort front by this objective
            sorted_front = sorted(front, key=lambda i: self.objective_values[i][obj_idx])
            
            # Set boundary points to infinity
            distances[0] = float('inf')
            distances[n - 1] = float('inf')
            
            # Calculate normalization factor to handle different scales
            obj_min = min(self.objective_values[i][obj_idx] for i in front)
            obj_max = max(self.objective_values[i][obj_idx] for i in front)
            scale = max(1e-10, obj_max - obj_min)  # Avoid division by zero
            
            # Calculate distance for intermediate points
            for i in range(1, n - 1):
                # Add normalized distance to the distance measure
                prev_val = self.objective_values[sorted_front[i - 1]][obj_idx]
                next_val = self.objective_values[sorted_front[i + 1]][obj_idx]
                
                # Add normalized distance contribution
                distances[i] += (next_val - prev_val) / scale
        
        return distances
    
    def _update_pareto_front(self):
        """Update the Pareto front based on current population."""
        # Perform non-dominated sorting
        self.pareto_fronts = self._non_dominated_sort()
        
        # Calculate crowding distance for each front
        self.crowding_distances = []
        for front in self.pareto_fronts:
            distances = self._calculate_crowding_distance(front)
            self.crowding_distances.append(distances)
        
        # Update the Pareto optimal set (first front)
        self.pareto_optimal = []
        for idx in self.pareto_fronts[0]:
            self.pareto_optimal.append({
                'parameters': deepcopy(self.population[idx]),
                'objectives': self.objective_values[idx]
            })
    
    def _select_parents(self) -> List[Dict[str, Any]]:
        """
        Select parents for reproduction using tournament selection.
        
        Returns:
            List of selected parents
        """
        if self.algorithm == MultiObjectiveAlgorithm.NSGA_II:
            return self._nsga_ii_selection()
        elif self.algorithm == MultiObjectiveAlgorithm.MOEA_D:
            return self._moea_d_selection()
        else:
            return self._nsga_ii_selection()  # Default
    
    def _nsga_ii_selection(self) -> List[Dict[str, Any]]:
        """
        NSGA-II selection method.
        
        Returns:
            Selected parents
        """
        parents = []
        
        # Create a flattened list of individuals sorted by rank and crowding distance
        ranked_indices = []
        for front_idx, front in enumerate(self.pareto_fronts):
            for i, idx in enumerate(front):
                ranked_indices.append((idx, front_idx, self.crowding_distances[front_idx][i]))
        
        # Sort by rank and then by crowding distance (descending)
        ranked_indices.sort(key=lambda x: (x[1], -x[2]))
        
        # Take the best individuals as parents
        for i in range(min(self.population_size, len(ranked_indices))):
            idx = ranked_indices[i][0]
            parents.append(deepcopy(self.population[idx]))
        
        # If we don't have enough, fill with random individuals
        while len(parents) < self.population_size:
            random_idx = random.randint(0, len(self.population) - 1)
            parents.append(deepcopy(self.population[random_idx]))
        
        return parents
    
    def _moea_d_selection(self) -> List[Dict[str, Any]]:
        """
        MOEA/D selection method using decomposition.
        
        Returns:
            Selected parents
        """
        parents = []
        
        # Create weight vectors (evenly distributed)
        weight_vectors = self._generate_weight_vectors(self.population_size)
        
        # For each weight vector, find the best individual
        for weights in weight_vectors:
            best_idx = self._find_best_for_weights(weights)
            parents.append(deepcopy(self.population[best_idx]))
        
        return parents
    
    def _generate_weight_vectors(self, n: int) -> List[List[float]]:
        """
        Generate evenly distributed weight vectors.
        
        Args:
            n: Number of weight vectors to generate
            
        Returns:
            List of weight vectors
        """
        if self.n_objectives == 2:
            # For 2 objectives, we can use a simple approach
            vectors = []
            for i in range(n):
                w1 = i / (n - 1) if n > 1 else 0.5
                w2 = 1.0 - w1
                vectors.append([w1, w2])
            return vectors
        else:
            # For more objectives, use a simple random approach
            # In a real implementation, a more sophisticated approach would be used
            vectors = []
            for _ in range(n):
                # Generate random weights and normalize
                weights = [random.random() for _ in range(self.n_objectives)]
                total = sum(weights)
                normalized = [w / total for w in weights]
                vectors.append(normalized)
            return vectors
    
    def _find_best_for_weights(self, weights: List[float]) -> int:
        """
        Find the best individual for a specific weight vector.
        
        Args:
            weights: Weight vector for objectives
            
        Returns:
            Index of the best individual
        """
        best_idx = 0
        best_value = float('-inf')
        
        for i in range(len(self.population)):
            # Calculate weighted sum of objectives
            value = self._calculate_weighted_sum(self.objective_values[i], weights)
            
            if value > best_value:
                best_value = value
                best_idx = i
        
        return best_idx
    
    def _calculate_weighted_sum(self, objectives: List[float], weights: List[float]) -> float:
        """
        Calculate weighted sum of objectives.
        
        Args:
            objectives: Objective values
            weights: Weight vector
            
        Returns:
            Weighted sum
        """
        weighted_sum = 0.0
        
        for i in range(self.n_objectives):
            # Adjust value based on direction (maximize or minimize)
            if self.objective_directions[i] == OptimizationDirection.MAXIMIZE:
                value = objectives[i]
            else:
                # For minimization, negate the value
                value = -objectives[i]
            
            weighted_sum += weights[i] * value
        
        return weighted_sum
    
    def _crossover(self, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform crossover to create offspring.
        
        Args:
            parents: List of parent individuals
            
        Returns:
            List of offspring
        """
        offspring = []
        
        # Keep some parents unchanged (elitism)
        elite_count = min(len(self.pareto_fronts[0]), self.population_size // 10)
        for i in range(elite_count):
            if i < len(parents):
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
                child = self._uniform_crossover(parent1, parent2)
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
        elite_count = min(len(self.pareto_fronts[0]), self.population_size // 10)
        
        for i in range(elite_count, len(offspring)):
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
        """Adjust mutation rate based on diversity."""
        # Calculate diversity as spread of Pareto front
        if self.pareto_fronts and self.pareto_fronts[0]:
            # If we have more than one individual in the Pareto front
            if len(self.pareto_fronts[0]) > 1:
                # Calculate diversity as average crowding distance
                avg_crowding = sum(self.crowding_distances[0]) / len(self.crowding_distances[0])
                normalized_crowding = min(1.0, avg_crowding / (self.n_objectives * 2.0))
                
                # Low crowding -> higher mutation to increase diversity
                diversity_factor = 1.0 - normalized_crowding
                
                # Apply scaling, allowing mutation rate to vary between 75% and 150% of base rate
                self.mutation_rate = self.base_mutation_rate * (0.75 + 0.75 * diversity_factor)
                
                logger.debug(f"Crowding: {avg_crowding:.4f}, Adjusted mutation rate: {self.mutation_rate:.4f}")
            else:
                # Only one individual, increase mutation to explore
                self.mutation_rate = self.base_mutation_rate * 1.5
    
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
        
        # Reset objective values for new population
        # (but preserve values for elite individuals if possible)
        elite_count = min(len(self.pareto_fronts[0]), self.population_size // 10)
        preserved_values = self.objective_values[:elite_count] if elite_count > 0 else []
        self.objective_values = preserved_values + [[None] * self.n_objectives for _ in range(self.population_size - elite_count)]
    
    def get_pareto_optimal_solutions(self) -> List[Dict[str, Any]]:
        """
        Get the Pareto optimal solutions.
        
        Returns:
            List of dictionaries with parameters and objectives
        """
        return self.pareto_optimal
    
    def get_best_compromise_solution(self) -> Dict[str, Any]:
        """
        Get the best compromise solution using weighted sum.
        
        Returns:
            Dictionary with parameters and objectives
        """
        if not self.pareto_optimal:
            # No solutions yet, return default parameters
            return {
                'parameters': self.parameter_space.get_default_parameters(),
                'objectives': [0.0] * self.n_objectives
            }
        
        # Find best solution using weights
        best_idx = 0
        best_value = float('-inf')
        
        for i, solution in enumerate(self.pareto_optimal):
            value = self._calculate_weighted_sum(solution['objectives'], self.weights)
            
            if value > best_value:
                best_value = value
                best_idx = i
        
        return self.pareto_optimal[best_idx]
    
    def optimize(
        self, 
        objective_function: Callable[[Dict[str, Any]], List[float]],
        n_generations: int = 20,
        callback: Optional[Callable[[List[Dict[str, Any]], Dict[str, Any], int], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the optimization process for a given objective function.
        
        Args:
            objective_function: Function that evaluates parameters and returns objective values
            n_generations: Number of generations to run
            callback: Optional callback after each generation
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        for generation in range(n_generations):
            # Evaluate current population
            self.evaluate_population(objective_function)
            
            # Update Pareto front
            self._update_pareto_front()
            
            # Get best compromise solution
            best_compromise = self.get_best_compromise_solution()
            
            # Calculate statistics for this generation
            gen_stats = {
                'generation': generation,
                'pareto_size': len(self.pareto_optimal),
                'best_compromise': best_compromise,
                'mutation_rate': self.mutation_rate
            }
            
            # For each objective, track best, worst, and average
            for i, name in enumerate(self.objective_names):
                values = [ind[i] for ind in self.objective_values if None not in ind]
                if values:
                    gen_stats[f"{name}_best"] = max(values)
                    gen_stats[f"{name}_worst"] = min(values)
                    gen_stats[f"{name}_avg"] = sum(values) / len(values)
            
            # Record statistics
            self.generation_stats.append(gen_stats)
            
            # Log progress
            logger.info(
                f"Generation {generation+1}/{n_generations}, "
                f"Pareto front size: {len(self.pareto_optimal)}"
            )
            
            # Call callback if provided
            if callback:
                callback(self.pareto_optimal, best_compromise, generation)
            
            # Evolve population (except for last generation)
            if generation < n_generations - 1:
                self.evolve()
        
        # Ensure final population is evaluated
        self.evaluate_population(objective_function)
        self._update_pareto_front()
        
        # Get final results
        optimization_time = time.time() - start_time
        best_compromise = self.get_best_compromise_solution()
        
        return {
            'pareto_front': self.pareto_optimal,
            'best_compromise': best_compromise,
            'n_generations': n_generations,
            'optimization_time': optimization_time,
            'all_parameters': self.parameters_history,
            'generation_stats': self.generation_stats,
            'parameter_space': self.parameter_space.get_parameters_dict(),
            'objective_names': self.objective_names,
            'objective_directions': [d.value for d in self.objective_directions],
            'weights': self.weights
        }


if __name__ == "__main__":
    # Example usage
    def example_objective(params):
        # Multi-objective test problem: minimize one, maximize the other
        x = params['x']
        y = params['y']
        
        # First objective: minimize distance from (0,0)
        obj1 = -(x**2 + y**2)  # Negated for maximization
        
        # Second objective: maximize product
        obj2 = x * y
        
        return [obj1, obj2]
    
    # Create parameter space
    param_space = ParameterSpace()
    param_space.add_real_parameter('x', -5.0, 5.0)
    param_space.add_real_parameter('y', -5.0, 5.0)
    
    # Create optimizer
    optimizer = MultiObjectiveOptimizer(
        parameter_space=param_space,
        objective_names=["distance", "product"],
        objective_directions=["maximize", "maximize"],
        weights=[0.5, 0.5],
        population_size=50,
        algorithm=MultiObjectiveAlgorithm.NSGA_II
    )
    
    # Run optimization
    results = optimizer.optimize(example_objective, n_generations=20)
    
    # Print results
    print(f"Optimization time: {results['optimization_time']:.2f} seconds")
    print(f"Pareto front size: {len(results['pareto_front'])}")
    print("\nBest compromise solution:")
    compromise = results['best_compromise']
    print(f"Parameters: {compromise['parameters']}")
    print(f"Objectives: {compromise['objectives']}")
