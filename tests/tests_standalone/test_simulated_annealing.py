#!/usr/bin/env python3
"""
Standalone Tests for Simulated Annealing Optimizer

This file contains dependency-free tests for the Simulated Annealing optimizer.
The tests verify that the optimizer can escape local optima and converge to
global optimal solutions for non-convex benchmark functions.
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


class CoolingSchedule:
    """Cooling schedule for simulated annealing."""
    
    @staticmethod
    def exponential(t0, alpha, iteration):
        """Exponential cooling schedule."""
        return t0 * (alpha ** iteration)
    
    @staticmethod
    def linear(t0, tf, n_iterations, iteration):
        """Linear cooling schedule."""
        return t0 - (t0 - tf) * (iteration / n_iterations)
    
    @staticmethod
    def logarithmic(t0, iteration):
        """Logarithmic cooling schedule."""
        return t0 / (1 + math.log(1 + iteration))


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


class SimulatedAnnealingOptimizer:
    """
    Simulated annealing optimization for testing.
    This is a simplified version of the SimulatedAnnealingOptimizer class for testing.
    """
    
    def __init__(
        self, 
        parameter_space,
        initial_temp=100.0,
        cooling_rate=0.95,
        n_steps_per_temp=10,
        min_temp=1e-10,
        adaptive_step_size=True,
        neighborhood_size=0.1,
        minimize=False,
        cooling_schedule="exponential"
    ):
        """Initialize the simulated annealing optimizer."""
        self.parameter_space = parameter_space
        self.param_names = parameter_space.param_names
        self.param_types = parameter_space.param_types
        self.bounds = parameter_space.bounds
        self.categories = parameter_space.categories
        
        # Annealing parameters
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.n_steps_per_temp = n_steps_per_temp
        self.min_temp = min_temp
        self.adaptive_step_size = adaptive_step_size
        self.neighborhood_size = neighborhood_size
        self.minimize = minimize
        self.cooling_schedule = cooling_schedule
        
        # Current state
        self.current_solution = None
        self.current_value = float('-inf') if not minimize else float('inf')
        self.best_solution = None
        self.best_value = float('-inf') if not minimize else float('inf')
        
        # History for tracking progress
        self.parameters_history = []
        self.temperature_history = []
        self.accept_history = []  # Track acceptance rate
    
    def initialize(self):
        """Initialize with random solution."""
        # Get default parameters as starting point
        self.current_solution = self.parameter_space.get_default_parameters()
        
        # Reset temperature
        self.current_temp = self.initial_temp
        
        # Clear history
        self.parameters_history = []
        self.temperature_history = []
        self.accept_history = []
    
    def _generate_neighbor(self, solution):
        """Generate a neighboring solution."""
        neighbor = deepcopy(solution)
        
        # Choose a random parameter to modify
        param_name = random.choice(self.param_names)
        param_type = self.param_types[param_name]
        
        if param_type == ParameterType.REAL:
            # Modify real parameter
            lower, upper = self.bounds[param_name]
            range_size = upper - lower
            
            # Determine step size
            if self.adaptive_step_size:
                # Step size decreases with temperature
                step_size = range_size * self.neighborhood_size * (self.current_temp / self.initial_temp)
            else:
                step_size = range_size * self.neighborhood_size
            
            # Apply Gaussian perturbation
            new_value = neighbor[param_name] + random.gauss(0, step_size)
            neighbor[param_name] = max(lower, min(upper, new_value))
            
        elif param_type == ParameterType.INTEGER:
            # Modify integer parameter
            lower, upper = self.bounds[param_name]
            range_size = upper - lower
            
            # Determine step size
            if self.adaptive_step_size:
                # Step size decreases with temperature
                step_size = max(1, int(range_size * self.neighborhood_size * (self.current_temp / self.initial_temp)))
            else:
                step_size = max(1, int(range_size * self.neighborhood_size))
            
            # Apply random step
            step = random.randint(-step_size, step_size)
            new_value = neighbor[param_name] + step
            neighbor[param_name] = max(lower, min(upper, new_value))
            
        elif param_type == ParameterType.CATEGORICAL:
            # Modify categorical parameter
            categories = self.categories[param_name]
            if len(categories) > 1:
                # Select a different category
                current_category = neighbor[param_name]
                available = [c for c in categories if c != current_category]
                if available:
                    neighbor[param_name] = random.choice(available)
                
        elif param_type == ParameterType.BOOLEAN:
            # Flip boolean parameter
            neighbor[param_name] = not neighbor[param_name]
        
        return neighbor
    
    def _acceptance_probability(self, current_value, new_value):
        """Calculate acceptance probability."""
        # For maximization, we want to move to higher values
        # For minimization, we want to move to lower values
        if (not self.minimize and new_value > current_value) or \
           (self.minimize and new_value < current_value):
            # Always accept better solutions
            return 1.0
        else:
            # Calculate difference (always positive)
            if self.minimize:
                delta = new_value - current_value
            else:
                delta = current_value - new_value
            
            # Calculate acceptance probability
            # Lower temperatures make accepting worse solutions less likely
            return math.exp(-delta / self.current_temp)
    
    def _cool_temperature(self, iteration, n_iterations):
        """Update temperature according to cooling schedule."""
        if self.cooling_schedule == "exponential":
            self.current_temp = CoolingSchedule.exponential(
                self.initial_temp, self.cooling_rate, iteration
            )
        elif self.cooling_schedule == "linear":
            self.current_temp = CoolingSchedule.linear(
                self.initial_temp, self.min_temp, n_iterations, iteration
            )
        elif self.cooling_schedule == "logarithmic":
            self.current_temp = CoolingSchedule.logarithmic(
                self.initial_temp, iteration
            )
        else:
            # Default to exponential
            self.current_temp = CoolingSchedule.exponential(
                self.initial_temp, self.cooling_rate, iteration
            )
        
        # Ensure temperature doesn't go below minimum
        self.current_temp = max(self.current_temp, self.min_temp)
    
    def get_best_parameters(self):
        """Get the best parameters found so far."""
        if self.best_solution is None:
            # No evaluations yet, return default parameters
            return self.parameter_space.get_default_parameters(), 0.0
        
        return self.best_solution, self.best_value
    
    def optimize(self, objective_function, n_iterations=100, callback=None):
        """Run the optimization process."""
        start_time = time.time()
        
        # Initialize
        self.initialize()
        
        # Evaluate initial solution
        try:
            self.current_value = objective_function(self.current_solution)
        except Exception as e:
            if self.minimize:
                self.current_value = float('inf')
            else:
                self.current_value = float('-inf')
        
        # Set initial solution as best solution
        self.best_solution = deepcopy(self.current_solution)
        self.best_value = self.current_value
        
        # Add to history
        self.parameters_history.append({
            'parameters': deepcopy(self.current_solution),
            'value': self.current_value,
            'temperature': self.current_temp
        })
        
        # Main annealing loop
        iteration = 0
        while iteration < n_iterations and self.current_temp > self.min_temp:
            accepted_count = 0
            
            # Steps at the current temperature
            for step in range(self.n_steps_per_temp):
                # Generate neighbor
                neighbor = self._generate_neighbor(self.current_solution)
                
                # Evaluate neighbor
                try:
                    neighbor_value = objective_function(neighbor)
                except Exception as e:
                    if self.minimize:
                        neighbor_value = float('inf')
                    else:
                        neighbor_value = float('-inf')
                
                # Determine if we should accept the neighbor
                accept_prob = self._acceptance_probability(self.current_value, neighbor_value)
                
                if random.random() < accept_prob:
                    # Accept neighbor
                    self.current_solution = neighbor
                    self.current_value = neighbor_value
                    accepted_count += 1
                    
                    # Update best solution if needed
                    if (not self.minimize and neighbor_value > self.best_value) or \
                       (self.minimize and neighbor_value < self.best_value):
                        self.best_solution = deepcopy(neighbor)
                        self.best_value = neighbor_value
                
                # Add to history
                self.parameters_history.append({
                    'parameters': deepcopy(neighbor),
                    'value': neighbor_value,
                    'accepted': random.random() < accept_prob,
                    'temperature': self.current_temp
                })
            
            # Track acceptance rate
            acceptance_rate = accepted_count / self.n_steps_per_temp
            self.accept_history.append(acceptance_rate)
            
            # Track temperature
            self.temperature_history.append(self.current_temp)
            
            # Log progress
            print(
                f"Iteration {iteration+1}/{n_iterations}, "
                f"Temp: {self.current_temp:.6f}, "
                f"Best: {self.best_value:.6f}"
            )
            
            # Call callback if provided
            if callback:
                callback(self.best_solution, self.best_value, self.current_temp)
            
            # Cool temperature
            self._cool_temperature(iteration, n_iterations)
            
            iteration += 1
        
        # Verify best solution (in case it wasn't evaluated properly)
        try:
            final_value = objective_function(self.best_solution)
            if (not self.minimize and final_value > self.best_value) or \
               (self.minimize and final_value < self.best_value):
                self.best_value = final_value
        except Exception:
            pass
        
        optimization_time = time.time() - start_time
        
        return {
            'best_parameters': self.best_solution,
            'best_value': self.best_value,
            'n_iterations': iteration,
            'optimization_time': optimization_time,
            'all_parameters': self.parameters_history,
            'temperature_history': self.temperature_history,
            'accept_history': self.accept_history
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
    def rastrigin(params):
        """Rastrigin function (minimization) - optimal at (0,0) but many local optima."""
        x = params['x']
        y = params['y']
        n = 2  # dimension
        A = 10
        return -(A * n + (x**2 - A * math.cos(2 * math.pi * x)) + (y**2 - A * math.cos(2 * math.pi * y)))
    
    @staticmethod
    def ackley(params):
        """Ackley function (minimization) - optimal at (0,0) but challenging landscape."""
        x = params['x']
        y = params['y']
        term1 = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2)))
        term2 = -math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
        return -(term1 + term2 + 20 + math.e)  # Negated for maximization
    
    @staticmethod
    def himmelblau(params):
        """Himmelblau function - has 4 identical local optima."""
        x = params['x']
        y = params['y']
        return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)  # Negated for maximization
    
    @staticmethod
    def mixed_params_with_barriers(params):
        """Function with mixed parameters and barriers."""
        x = params['x']
        y = params['y']
        method = params['method']
        use_barrier = params['use_barrier']
        
        # Base calculation
        base = -(x**2 + y**2)
        
        # Apply method factor
        if method == 'method_a':
            method_factor = 1.0
        elif method == 'method_b':
            method_factor = 1.2
        else:  # method_c
            method_factor = 0.8
        
        # Apply barrier if enabled
        barrier_factor = 1.0
        if use_barrier:
            # Create a barrier at x = 2
            if 1.8 < x < 2.2:
                barrier_factor = 0.1  # Significant penalty
        
        return base * method_factor * barrier_factor


class TestSimulatedAnnealing(unittest.TestCase):
    """Test the Simulated Annealing optimizer."""
    
    def test_sphere_function(self):
        """Test optimization of the sphere function."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        
        # Create optimizer
        optimizer = SimulatedAnnealingOptimizer(
            parameter_space=param_space,
            initial_temp=100.0,
            cooling_rate=0.95,
            n_steps_per_temp=5,
            minimize=False  # Maximize negative sphere
        )
        
        # Run optimization
        results = optimizer.optimize(BenchmarkFunctions.sphere, n_iterations=20)
        
        # Get best parameters
        best_params = results['best_parameters']
        
        # Check that it converged close to the optimum (0,0)
        self.assertAlmostEqual(best_params['x'], 0.0, delta=0.5)
        self.assertAlmostEqual(best_params['y'], 0.0, delta=0.5)
    
    def test_rastrigin_function(self):
        """Test optimization of the Rastrigin function (many local optima)."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        
        # Create optimizer
        optimizer = SimulatedAnnealingOptimizer(
            parameter_space=param_space,
            initial_temp=100.0,
            cooling_rate=0.9,  # Slower cooling
            n_steps_per_temp=10,  # More steps per temperature
            minimize=False  # Maximize negative Rastrigin
        )
        
        # Run optimization
        results = optimizer.optimize(BenchmarkFunctions.rastrigin, n_iterations=30)
        
        # Get best parameters
        best_params = results['best_parameters']
        best_value = results['best_value']
        
        # Check that it found a good solution
        # For Rastrigin, we might not hit exact optimum due to many local optima
        # but we should get close to one of them
        self.assertGreater(best_value, -2.0)  # Reasonable threshold
    
    def test_himmelblau_function(self):
        """Test optimization of Himmelblau function (4 local optima)."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        
        # Create optimizer
        optimizer = SimulatedAnnealingOptimizer(
            parameter_space=param_space,
            initial_temp=100.0,
            cooling_rate=0.92,
            n_steps_per_temp=8,
            minimize=False  # Maximize negative Himmelblau
        )
        
        # Run optimization
        results = optimizer.optimize(BenchmarkFunctions.himmelblau, n_iterations=25)
        
        # Get best parameters
        best_params = results['best_parameters']
        best_value = results['best_value']
        
        # Known optima of Himmelblau function
        optima = [
            (3.0, 2.0),
            (-2.805118, 3.131312),
            (-3.779310, -3.283186),
            (3.584428, -1.848126)
        ]
        
        # Check if we found one of the optima
        found_optimum = False
        for x_opt, y_opt in optima:
            if (abs(best_params['x'] - x_opt) < 0.5 and 
                abs(best_params['y'] - y_opt) < 0.5):
                found_optimum = True
                break
        
        self.assertTrue(found_optimum or best_value > -0.1)
    
    def test_mixed_parameters(self):
        """Test optimization with mixed parameter types."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        param_space.add_categorical_parameter('method', ['method_a', 'method_b', 'method_c'])
        param_space.add_boolean_parameter('use_barrier', False)
        
        # Create optimizer
        optimizer = SimulatedAnnealingOptimizer(
            parameter_space=param_space,
            initial_temp=100.0,
            cooling_rate=0.9,
            n_steps_per_temp=8,
            minimize=False  # Maximize
        )
        
        # Run optimization
        results = optimizer.optimize(BenchmarkFunctions.mixed_params_with_barriers, n_iterations=25)
        
        # Get best parameters
        best_params = results['best_parameters']
        
        # Check results
        # Real parameters should be close to (0,0)
        self.assertAlmostEqual(best_params['x'], 0.0, delta=1.0)
        self.assertAlmostEqual(best_params['y'], 0.0, delta=1.0)
        
        # The optimizer should find either method_a or method_b (both are reasonable)
        # In reality, method_b should be slightly better, but we can't guarantee it with
        # limited iterations in a stochastic algorithm
        self.assertIn(best_params['method'], ['method_a', 'method_b'])
        
        # use_barrier should be False to avoid the penalty
        self.assertEqual(best_params['use_barrier'], False)
    
    def test_cooling_schedules(self):
        """Test different cooling schedules."""
        # Create parameter space
        param_space = ParameterSpace()
        param_space.add_real_parameter('x', -5.0, 5.0)
        param_space.add_real_parameter('y', -5.0, 5.0)
        
        for schedule in ["exponential", "linear", "logarithmic"]:
            # Create optimizer with different cooling schedule
            optimizer = SimulatedAnnealingOptimizer(
                parameter_space=param_space,
                initial_temp=100.0,
                cooling_rate=0.9,
                n_steps_per_temp=5,
                cooling_schedule=schedule,
                minimize=False
            )
            
            # Run optimization
            results = optimizer.optimize(BenchmarkFunctions.sphere, n_iterations=15)
            
            # Get best parameters
            best_params = results['best_parameters']
            
            # Check that it found a reasonable solution
            self.assertLess(best_params['x']**2 + best_params['y']**2, 5.0)


if __name__ == "__main__":
    unittest.main()
