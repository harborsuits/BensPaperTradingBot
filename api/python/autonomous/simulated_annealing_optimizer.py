#!/usr/bin/env python3
"""
Simulated Annealing Optimizer

This module implements simulated annealing for strategy parameter optimization,
building directly on our existing parameter space representation. Simulated annealing
is effective at avoiding local optima in complex parameter landscapes.

Key features:
1. Temperature-based acceptance of worse solutions to escape local optima
2. Adaptive neighborhood sizing based on parameter types
3. Exponential cooling schedule with configurable parameters
4. Integration with our existing parameter space representation
5. Ability to handle mixed parameter types (continuous, integer, categorical)
"""

import os
import json
import logging
import time
import random
import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime
from copy import deepcopy

# Import parameter space from existing optimizer
from trading_bot.autonomous.bayesian_optimizer import (
    ParameterSpace, ParameterType, OptimizationDirection
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoolingSchedule:
    """Cooling schedule for simulated annealing."""
    
    @staticmethod
    def exponential(t0, alpha, iteration):
        """
        Exponential cooling schedule.
        
        Args:
            t0: Initial temperature
            alpha: Cooling rate (0-1)
            iteration: Current iteration
            
        Returns:
            Current temperature
        """
        return t0 * (alpha ** iteration)
    
    @staticmethod
    def linear(t0, tf, n_iterations, iteration):
        """
        Linear cooling schedule.
        
        Args:
            t0: Initial temperature
            tf: Final temperature
            n_iterations: Total iterations
            iteration: Current iteration
            
        Returns:
            Current temperature
        """
        return t0 - (t0 - tf) * (iteration / n_iterations)
    
    @staticmethod
    def logarithmic(t0, iteration):
        """
        Logarithmic cooling schedule.
        
        Args:
            t0: Initial temperature
            iteration: Current iteration
            
        Returns:
            Current temperature
        """
        return t0 / (1 + math.log(1 + iteration))


class SimulatedAnnealingOptimizer:
    """
    Simulated annealing optimization for strategy parameters.
    
    This class implements simulated annealing, a probabilistic technique for
    approximating the global optimum of a function. It is particularly useful
    for parameter spaces where multiple local optima exist.
    """
    
    def __init__(
        self, 
        parameter_space: ParameterSpace,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        n_steps_per_temp: int = 10,
        min_temp: float = 1e-10,
        adaptive_step_size: bool = True,
        neighborhood_size: float = 0.1,
        minimize: bool = False,
        cooling_schedule: str = "exponential"
    ):
        """
        Initialize the simulated annealing optimizer.
        
        Args:
            parameter_space: Parameter space to optimize
            initial_temp: Initial temperature
            cooling_rate: Cooling rate (0-1)
            n_steps_per_temp: Steps to perform at each temperature
            min_temp: Minimum temperature (termination condition)
            adaptive_step_size: Whether to adapt step sizes
            neighborhood_size: Size of neighborhood (as fraction of range)
            minimize: Whether to minimize (True) or maximize (False)
            cooling_schedule: Type of cooling schedule
        """
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
    
    def _generate_neighbor(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a neighboring solution.
        
        Args:
            solution: Current solution
            
        Returns:
            Neighbor solution
        """
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
                current_value = neighbor[param_name]
                new_value = current_value
                
                # Higher temperature = more likely to choose any category
                # Lower temperature = more likely to stay close to current category
                if random.random() < (self.current_temp / self.initial_temp):
                    # Choose completely randomly
                    while new_value == current_value:
                        new_value = random.choice(categories)
                else:
                    # Stay close to current category (if possible)
                    current_idx = categories.index(current_value)
                    new_idx = (current_idx + random.choice([-1, 1])) % len(categories)
                    new_value = categories[new_idx]
                
                neighbor[param_name] = new_value
                
        elif param_type == ParameterType.BOOLEAN:
            # Flip boolean parameter
            neighbor[param_name] = not neighbor[param_name]
        
        return neighbor
    
    def _acceptance_probability(self, current_value: float, new_value: float) -> float:
        """
        Calculate acceptance probability.
        
        Args:
            current_value: Current solution value
            new_value: New solution value
            
        Returns:
            Acceptance probability (0-1)
        """
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
    
    def _cool_temperature(self, iteration: int, n_iterations: int):
        """
        Update temperature according to cooling schedule.
        
        Args:
            iteration: Current iteration
            n_iterations: Total iterations
        """
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
    
    def get_best_parameters(self) -> Tuple[Dict[str, Any], float]:
        """
        Get the best parameters found so far.
        
        Returns:
            Tuple of (best_parameters, best_value)
        """
        if self.best_solution is None:
            # No evaluations yet, return default parameters
            return self.parameter_space.get_default_parameters(), 0.0
        
        return self.best_solution, self.best_value
    
    def optimize(
        self, 
        objective_function: Callable[[Dict[str, Any]], float],
        n_iterations: int = 100,
        callback: Optional[Callable[[Dict[str, Any], float, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the optimization process for a given objective function.
        
        Args:
            objective_function: Function that evaluates parameters
            n_iterations: Maximum number of temperature iterations
            callback: Optional callback after each iteration
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Initialize
        self.initialize()
        
        # Evaluate initial solution
        try:
            self.current_value = objective_function(self.current_solution)
        except Exception as e:
            logger.error(f"Error evaluating initial solution: {str(e)}")
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
            'temperature': self.current_temp,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Initial solution: value={self.current_value}, temp={self.current_temp}")
        
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
                    logger.error(f"Error evaluating neighbor: {str(e)}")
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
                    'temperature': self.current_temp,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Track acceptance rate
            acceptance_rate = accepted_count / self.n_steps_per_temp
            self.accept_history.append(acceptance_rate)
            
            # Track temperature
            self.temperature_history.append(self.current_temp)
            
            # Log progress
            logger.info(
                f"Iteration {iteration+1}/{n_iterations}, "
                f"Temp: {self.current_temp:.6f}, "
                f"Best: {self.best_value:.6f}, "
                f"Current: {self.current_value:.6f}, "
                f"Acceptance Rate: {acceptance_rate:.2f}"
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
            'accept_history': self.accept_history,
            'parameter_space': self.parameter_space.get_parameters_dict()
        }


if __name__ == "__main__":
    # Example usage
    def example_objective(params):
        # Ackley function (minimization)
        x = params['x']
        y = params['y']
        term1 = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2)))
        term2 = -math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
        return -(term1 + term2 + 20 + math.e)  # Negate for maximization
    
    # Create parameter space
    param_space = ParameterSpace()
    param_space.add_real_parameter('x', -5.0, 5.0)
    param_space.add_real_parameter('y', -5.0, 5.0)
    
    # Create optimizer
    optimizer = SimulatedAnnealingOptimizer(
        parameter_space=param_space,
        initial_temp=100.0,
        cooling_rate=0.95,
        n_steps_per_temp=10,
        minimize=False  # Maximize negative Ackley
    )
    
    # Run optimization
    results = optimizer.optimize(example_objective, n_iterations=50)
    
    # Print results
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best value: {results['best_value']}")
    print(f"Optimization time: {results['optimization_time']:.2f} seconds")
