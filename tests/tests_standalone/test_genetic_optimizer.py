#!/usr/bin/env python3
"""
Standalone Tests for Genetic Algorithm Optimizer

This file contains dependency-free tests for the Genetic Algorithm optimizer.
The tests verify that the optimizer can converge to known optimal solutions for
common benchmark functions.
"""

import unittest
import random
import math
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from copy import deepcopy

# Define minimal versions of the classes we need for testing
class ParameterType(str, Enum):
    """Parameter types for the search space."""
    REAL = "real"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


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


class ParameterSpace:
    """Simplified parameter space for testing."""
    
    def __init__(self):
        """Initialize an empty parameter space."""
        self.parameters = []
        self.param_names = []
        self.param_types = {}
        self.bounds = {}
        self.defaults = {}
        self.categories = {}
    
    def add_real_parameter(self, name, lower_bound, upper_bound, default=None):
        """Add a real-valued parameter."""
        if default is None:
            default = (lower_bound + upper_bound) / 2.0
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.REAL
        self.bounds[name] = (lower_bound, upper_bound)
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.REAL.value,
            'bounds': (lower_bound, upper_bound),
            'default': default
        })
        
        return self
    
    def add_integer_parameter(self, name, lower_bound, upper_bound, default=None):
        """Add an integer parameter."""
        if default is None:
            default = (lower_bound + upper_bound) // 2
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.INTEGER
        self.bounds[name] = (lower_bound, upper_bound)
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.INTEGER.value,
            'bounds': (lower_bound, upper_bound),
            'default': default
        })
        
        return self
    
    def add_categorical_parameter(self, name, categories, default=None):
        """Add a categorical parameter."""
        if default is None and categories:
            default = categories[0]
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.CATEGORICAL
        self.categories[name] = categories
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.CATEGORICAL.value,
            'categories': categories,
            'default': default
        })
        
        return self
    
    def add_boolean_parameter(self, name, default=False):
        """Add a boolean parameter."""
        self.param_names.append(name)
        self.param_types[name] = ParameterType.BOOLEAN
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.BOOLEAN.value,
            'default': default
        })
        
        return self
    
    def get_default_parameters(self):
        """Get default parameters as a dictionary."""
        return {name: self.defaults[name] for name in self.param_names}
    
    def get_random_parameters(self, count=1):
        """Generate random parameter combinations."""
        result = []
        
        for _ in range(count):
            params = {}
            
            for param in self.parameters:
                name = param['name']
                param_type = param['type']
                
                if param_type == ParameterType.REAL.value:
                    lower, upper = param['bounds']
                    params[name] = lower + random.random() * (upper - lower)
                    
                elif param_type == ParameterType.INTEGER.value:
                    lower, upper = param['bounds']
                    params[name] = random.randint(lower, upper)
                    
                elif param_type == ParameterType.CATEGORICAL.value:
                    categories = param['categories']
                    params[name] = random.choice(categories)
                    
                elif param_type == ParameterType.BOOLEAN.value:
                    params[name] = random.choice([True, False])
            
            result.append(params)
        
        return result
    
    def __len__(self):
        """Get number of parameters."""
        return len(self.param_names)


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for testing.
    This is a simplified version of the GeneticOptimizer class for testing.
    """
    
    def __init__(
        self, 
        parameter_space,
        population_size=50,
        elite_size=5,
        tournament_size=3,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.UNIFORM,
        minimize=False,
        adaptive_mutation=True
    ):
        """Initialize the genetic optimizer."""
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
    
    def evaluate_population(self, objective_function):
        """Evaluate fitness for all individuals in the population."""
        for i, individual in enumerate(self.population):
            if self.fitness_values[i] is None:  # Only evaluate if not already evaluated
                try:
                    fitness = objective_function(individual)
                except Exception as e:
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
                        'value': fitness
                    })
    
    def _select_parents(self):
        """Select parents for reproduction."""
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection()
        elif self.selection_method == SelectionMethod.RANK:
            return self._rank_selection()
        else:
            return self._tournament_selection()  # Default
    
    def _tournament_selection(self):
        """Tournament selection method."""
        parents = []
        
        # Keep elite individuals
        if self.elite_size > 0:
            # Sort population by fitness
            sorted_indices = sorted(range(len(self.fitness_values)), 
                                    key=lambda i: self.fitness_values[i] if not self.minimize else -self.fitness_values[i],
                                    reverse=True)
            
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
    
    def _roulette_selection(self):
        """Roulette wheel selection method."""
        # Implementation simplified for testing
        return self._tournament_selection()
    
    def _rank_selection(self):
        """Rank-based selection method."""
        # Implementation simplified for testing
        return self._tournament_selection()
    
    def _crossover(self, parents):
        """Perform crossover to create offspring."""
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
    
    def _uniform_crossover(self, parent1, parent2):
        """Uniform crossover."""
        child = {}
        
        for param_name in self.param_names:
            # For each parameter, randomly choose from either parent
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _single_point_crossover(self, parent1, parent2):
        """Single-point crossover."""
        # Implementation simplified for testing
        return self._uniform_crossover(parent1, parent2)
    
    def _two_point_crossover(self, parent1, parent2):
        """Two-point crossover."""
        # Implementation simplified for testing
        return self._uniform_crossover(parent1, parent2)
    
    def _mutate(self, offspring):
        """Mutate offspring."""
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
                        sigma = max(1, int((upper - lower) * 0.1))  # 10% of range
                        offset = random.randint(-sigma, sigma)
                        new_value = individual[param_name] + offset
                        individual[param_name] = max(lower, min(upper, new_value))
                        
                    elif param_type == ParameterType.CATEGORICAL:
                        # Mutate categorical parameter
                        categories = self.categories[param_name]
                        if len(categories) > 1:
                            # Select a different category
                            current_category = individual[param_name]
                            available = [c for c in categories if c != current_category]
                            if available:
                                individual[param_name] = random.choice(available)
                            
                    elif param_type == ParameterType.BOOLEAN:
                        # Flip boolean parameter
                        individual[param_name] = not individual[param_name]
        
        return offspring
    
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
    
    def get_best_parameters(self):
        """Get the best parameters found so far."""
        if not self.parameters_history:
            # No evaluations yet, return default parameters
            return self.parameter_space.get_default_parameters(), 0.0
        
        return self.best_individual, self.best_fitness
    
    def optimize(self, objective_function, n_generations=20, callback=None):
        """Run the optimization process."""
        start_time = time.time()
        
        for generation in range(n_generations):
            # Evaluate current population
            self.evaluate_population(objective_function)
            
            # Log progress
            print(f"Generation {generation+1}/{n_generations}, Best: {self.best_fitness:.4f}")
            
            # Call callback if provided
            if callback:
                callback(self.best_individual, self.best_fitness, generation)
            
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
            'all_parameters': self.parameters_history
        }


# Benchmark functions for testing optimization algorithms
class BenchmarkFunctions:
    """Common benchmark functions for testing optimizers."""
    
    @staticmethod
    def sphere(params):
        """Sphere function (minimization) - optimal at (0,0)."""
        x = params['x']
        y = params['y']
        return -(x**2 + y**2)  # Negated for maximization
    
    @staticmethod
    def rosenbrock(params):
        """Rosenbrock function (minimization) - optimal at (1,1)."""
        x = params['x']
        y = params['y']
        return -((1 - x)**2 + 100 * (y - x**2)**2)  # Negated for maximization
    
    @staticmethod
    def rastrigin(params):
        """Rastrigin function (minimization) - optimal at (0,0)."""
        x = params['x']
        y = params['y']
        n = 2  # dimension
        A = 10
        return -(A * n + (x**2 - A * math.cos(2 * math.pi * x)) + (y**2 - A * math.cos(2 * math.pi * y)))
    
    @staticmethod
    def ackley(params):
        """Ackley function (minimization) - optimal at (0,0)."""
        x = params['x']
        y = params['y']
        term1 = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2)))
        term2 = -math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
        return -(term1 + term2 + 20 + math.e)  # Negated for maximization
    
    @staticmethod
    def mixed_integer(params):
        """Test function with mixed parameter types."""
        x = params['x']
        y = params['y']
        window = params['window']
        method = params['method']
        
        # Basic test function
        base = -(x**2 + y**2)
        
        # Adjust based on integer and categorical params
        if method == 'sma':
            method_factor = 1.0
        elif method == 'ema':
            method_factor = 1.2
        else:  # wma
            method_factor = 0.9
        
        window_factor = 1.0 + 0.01 * window
        
        return base * method_factor * window_factor


class TestGeneticOptimizer(unittest.TestCase):
    """Test the Genetic Algorithm optimizer."""
    
    def test_sphere_function(self):
        """Test optimization of the sphere function."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            parameter_space=param_space,
            population_size=30,
            n_generations=15,
            minimize=False  # Maximize negative sphere
        )
        
        # Run optimization
        results = optimizer.optimize(BenchmarkFunctions.sphere, n_generations=15)
        
        # Get best parameters
        best_params = results['best_parameters']
        
        # Check that it converged close to the optimum (0,0)
        self.assertAlmostEqual(best_params['x'], 0.0, delta=0.5)
        self.assertAlmostEqual(best_params['y'], 0.0, delta=0.5)
    
    def test_rosenbrock_function(self):
        """Test optimization of the Rosenbrock function."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            parameter_space=param_space,
            population_size=40,
            n_generations=20,
            minimize=False  # Maximize negative Rosenbrock
        )
        
        # Run optimization
        results = optimizer.optimize(BenchmarkFunctions.rosenbrock, n_generations=20)
        
        # Get best parameters
        best_params = results['best_parameters']
        
        # Check that it converged close to the optimum (1,1)
        self.assertAlmostEqual(best_params['x'], 1.0, delta=0.5)
        self.assertAlmostEqual(best_params['y'], 1.0, delta=0.5)
    
    def test_mixed_parameter_types(self):
        """Test optimization with mixed parameter types."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        param_space.add_integer_parameter('window', 5, 50)
        param_space.add_categorical_parameter('method', ['sma', 'ema', 'wma'])
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            parameter_space=param_space,
            population_size=30,
            n_generations=15,
            minimize=False  # Maximize
        )
        
        # Run optimization
        results = optimizer.optimize(BenchmarkFunctions.mixed_integer, n_generations=15)
        
        # Get best parameters
        best_params = results['best_parameters']
        
        # Check that real parameters converged close to the optimum (0,0)
        self.assertAlmostEqual(best_params['x'], 0.0, delta=0.5)
        self.assertAlmostEqual(best_params['y'], 0.0, delta=0.5)
        
        # Check that integer and categorical parameters are valid
        self.assertTrue(5 <= best_params['window'] <= 50)
        self.assertIn(best_params['method'], ['sma', 'ema', 'wma'])
    
    def test_selection_methods(self):
        """Test different selection methods."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        
        for method in [SelectionMethod.TOURNAMENT, SelectionMethod.ROULETTE, SelectionMethod.RANK]:
            # Create optimizer with different selection method
            optimizer = GeneticOptimizer(
                parameter_space=param_space,
                population_size=20,
                n_generations=10,
                selection_method=method,
                minimize=False
            )
            
            # Run optimization
            results = optimizer.optimize(BenchmarkFunctions.sphere, n_generations=10)
            
            # Get best parameters
            best_params = results['best_parameters']
            
            # Check that it found a reasonable solution
            self.assertLess(best_params['x']**2 + best_params['y']**2, 5.0)


if __name__ == "__main__":
    unittest.main()
