"""
Genetic Algorithm Optimizer Module

Provides genetic algorithm based optimization for trading strategies.
"""

import logging
import random
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from trading_bot.ml_pipeline.optimizer.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

class GeneticOptimizer(BaseOptimizer):
    """
    Genetic algorithm based optimizer for trading strategies
    
    Uses evolutionary principles to efficiently explore parameter space:
    - Population of parameter sets (individuals)
    - Selection based on fitness (performance metrics)
    - Crossover between high-performing individuals
    - Mutation to explore new areas of parameter space
    - Evolution over generations to find optimal parameters
    """
    
    def __init__(self, config=None):
        """
        Initialize the genetic optimizer
        
        Args:
            config: Configuration dictionary with parameters
        """
        super().__init__(config)
        
        # Genetic algorithm parameters
        self.population_size = self.config.get('population_size', 50)
        self.generations = self.config.get('generations', 10)
        self.crossover_rate = self.config.get('crossover_rate', 0.7)
        self.mutation_rate = self.config.get('mutation_rate', 0.2)
        self.elitism_rate = self.config.get('elitism_rate', 0.1)  # Percentage of top performers to keep unchanged
        
        logger.info(f"Genetic Optimizer initialized with population={self.population_size}, generations={self.generations}")
    
    def optimize(self, 
                strategy_class, 
                param_space: Dict[str, Union[List, Tuple]], 
                historical_data: Dict[str, pd.DataFrame],
                metric: str = 'total_profit',
                metric_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters using genetic algorithm
        
        Args:
            strategy_class: Class of the strategy to optimize
            param_space: Dictionary of parameter names and possible values
            historical_data: Dictionary of symbol -> DataFrame with historical data
            metric: Metric to optimize ('total_profit', 'sortino', 'sharpe', etc.)
            metric_function: Optional custom function to calculate metric
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting genetic optimization for {strategy_class.__name__}")
        
        start_time = datetime.now()
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        # Initialize population
        param_names = list(param_space.keys())
        population = self._initialize_population(param_space)
        
        # Track best solution across generations
        best_individual = None
        best_fitness = float('-inf')
        best_metrics = {}
        all_results = []
        
        # Evaluate fitness of initial population
        fitness_results = self._evaluate_population(population, param_names, strategy_class, historical_data, metric, metric_function)
        
        # Find best individual in initial population
        for individual, fitness_result in zip(population, fitness_results):
            individual_params = dict(zip(param_names, individual))
            fitness = fitness_result.get(metric, float('-inf'))
            
            all_results.append({
                'params': individual_params.copy(),
                'metrics': fitness_result.copy(),
                'generation': 0
            })
            
            if fitness > best_fitness and 'error' not in fitness_result:
                best_fitness = fitness
                best_individual = individual.copy()
                best_metrics = fitness_result.copy()
        
        # Evolution over generations
        for generation in range(1, self.generations + 1):
            logger.info(f"Generation {generation}/{self.generations}")
            
            # Create next generation
            next_population = self._create_next_generation(population, fitness_results, param_space, metric)
            population = next_population
            
            # Evaluate new population
            fitness_results = self._evaluate_population(population, param_names, strategy_class, historical_data, metric, metric_function)
            
            # Update best individual
            for individual, fitness_result in zip(population, fitness_results):
                individual_params = dict(zip(param_names, individual))
                fitness = fitness_result.get(metric, float('-inf'))
                
                all_results.append({
                    'params': individual_params.copy(),
                    'metrics': fitness_result.copy(),
                    'generation': generation
                })
                
                if fitness > best_fitness and 'error' not in fitness_result:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    best_metrics = fitness_result.copy()
            
            # Log progress
            logger.info(f"Generation {generation} - Best {metric}: {best_fitness:.4f}")
        
        # Calculate elapsed time
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        # Convert best individual to parameters
        best_params = dict(zip(param_names, best_individual)) if best_individual else {}
        
        # Prepare final results
        final_results = {
            'strategy': strategy_class.__name__,
            'optimization_method': 'genetic',
            'best_params': best_params,
            'best_metrics': best_metrics,
            'generations': self.generations,
            'population_size': self.population_size,
            'elapsed_time': elapsed_time,
            'timestamp': end_time.isoformat(),
            'all_evaluations': all_results
        }
        
        # Store full results
        self.optimization_results.append(final_results)
        
        # Save results to disk
        self._save_results(final_results)
        
        logger.info(f"Genetic optimization completed in {elapsed_time:.1f} seconds")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {metric}: {best_fitness:.4f}")
        
        return final_results
    
    def _initialize_population(self, param_space: Dict[str, Union[List, Tuple]]) -> List[List]:
        """
        Initialize random population
        
        Args:
            param_space: Dictionary of parameter names and possible values
            
        Returns:
            List of individuals, each a list of parameter values
        """
        population = []
        param_values = list(param_space.values())
        
        for _ in range(self.population_size):
            # Generate a random individual
            individual = [np.random.choice(values) for values in param_values]
            population.append(individual)
        
        return population
    
    def _evaluate_population(self, 
                           population: List[List], 
                           param_names: List[str],
                           strategy_class,
                           historical_data: Dict[str, pd.DataFrame],
                           metric: str,
                           metric_function: Optional[Callable]) -> List[Dict[str, float]]:
        """
        Evaluate fitness of each individual in the population
        
        Args:
            population: List of individuals
            param_names: List of parameter names
            strategy_class: Strategy class to evaluate
            historical_data: Historical price data
            metric: Metric to optimize
            metric_function: Custom metric function
            
        Returns:
            List of fitness results
        """
        # Prepare combinations to evaluate
        combinations_to_test = []
        for individual in population:
            params = dict(zip(param_names, individual))
            combinations_to_test.append((strategy_class, params, historical_data, metric, metric_function))
        
        # Run evaluations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self._evaluate_strategy, *combo): combo[1]
                for combo in combinations_to_test
            }
            
            # Process results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating parameters: {e}")
                    results.append({"error": str(e)})
        
        return results
    
    def _create_next_generation(self, 
                               population: List[List], 
                               fitness_results: List[Dict[str, float]],
                               param_space: Dict[str, Union[List, Tuple]],
                               metric: str) -> List[List]:
        """
        Create next generation using selection, crossover and mutation
        
        Args:
            population: Current population
            fitness_results: Fitness results for current population
            param_space: Parameter space definition
            metric: Metric used for fitness
            
        Returns:
            New population for next generation
        """
        next_population = []
        param_values = list(param_space.values())
        
        # Extract fitness values
        fitness_values = [result.get(metric, float('-inf')) for result in fitness_results]
        
        # Handle case where all fitness values are invalid
        if all(f == float('-inf') for f in fitness_values):
            logger.warning("All individuals have invalid fitness, creating new random population")
            return self._initialize_population(param_space)
        
        # Apply elitism - keep best performers
        elites_count = max(1, int(self.population_size * self.elitism_rate))
        elite_indices = np.argsort(fitness_values)[-elites_count:]
        
        for idx in elite_indices:
            next_population.append(population[idx].copy())
        
        # Fill rest of population with crossover and mutation
        while len(next_population) < self.population_size:
            # Selection - tournament selection
            parent1 = self._tournament_selection(population, fitness_values)
            parent2 = self._tournament_selection(population, fitness_values)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1, param_values)
            child2 = self._mutate(child2, param_values)
            
            # Add to next generation
            next_population.append(child1)
            if len(next_population) < self.population_size:
                next_population.append(child2)
        
        return next_population
    
    def _tournament_selection(self, population: List[List], fitness_values: List[float]) -> List:
        """
        Select individual using tournament selection
        
        Args:
            population: Current population
            fitness_values: Fitness values for population
            
        Returns:
            Selected individual
        """
        # Select random individuals for tournament
        tournament_size = 3
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        
        # Find best individual in tournament
        best_idx = indices[0]
        for idx in indices[1:]:
            if fitness_values[idx] > fitness_values[best_idx]:
                best_idx = idx
        
        return population[best_idx]
    
    def _crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two children
        """
        # Single point crossover
        point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutate(self, individual: List, param_values: List[List]) -> List:
        """
        Mutate individual
        
        Args:
            individual: Individual to mutate
            param_values: Possible values for each parameter
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            # Apply mutation with some probability
            if random.random() < self.mutation_rate:
                # Replace with random value from parameter space
                mutated[i] = np.random.choice(param_values[i])
        
        return mutated
