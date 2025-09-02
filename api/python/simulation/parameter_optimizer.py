#!/usr/bin/env python3
"""
Parameter Optimizer for Trading Simulator.

This module provides optimization capabilities for trading strategies
and risk parameters using grid search and genetic algorithms.
"""

import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from trading_bot.simulation.trading_simulator import TradingSimulator, SimulationConfig
from trading_bot.data_providers.base_provider import DataProvider
from trading_bot.risk_manager import RiskManager

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ParameterSpace:
    """Definition of parameter ranges for optimization"""
    name: str
    values: List[Any]
    description: str = ""
    
@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization"""
    parameter_spaces: List[ParameterSpace]
    target_metric: str = "sharpe_ratio"  # Metric to optimize
    higher_is_better: bool = True  # Whether higher values are better
    max_parallel_jobs: int = 4  # Maximum parallel simulations
    save_all_results: bool = True  # Whether to save all results or just the best
    early_stopping: bool = False  # Whether to stop early if no improvement
    early_stopping_rounds: int = 5  # Number of rounds with no improvement before stopping
    output_dir: str = "optimization_results"
    resample_method: Optional[str] = None  # None, "bootstrap", "timeseries"
    resample_count: int = 5  # Number of resamples for robust optimization
    
    # Advanced settings - for genetic algorithm
    population_size: int = 20
    generations: int = 10
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    
    # Additional settings
    random_seed: Optional[int] = None
    
@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_parameters: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    optimization_time: float = 0.0
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    
class ParameterOptimizer:
    """
    Parameter optimizer for trading strategies and risk management.
    
    This class provides methods for optimizing strategy parameters
    using grid search or genetic algorithms.
    """
    
    def __init__(
        self,
        base_simulation_config: SimulationConfig,
        optimization_config: OptimizationConfig,
        data_provider: DataProvider,
        strategy_factory: Callable,
        risk_manager_factory: Optional[Callable] = None,
        custom_objective: Optional[Callable] = None
    ):
        """
        Initialize the parameter optimizer.
        
        Args:
            base_simulation_config: Base configuration for simulations
            optimization_config: Configuration for the optimization process
            data_provider: Data provider for market data
            strategy_factory: Factory function to create strategy instances with parameters
            risk_manager_factory: Optional factory function to create risk manager instances
            custom_objective: Optional custom objective function for evaluation
        """
        self.base_simulation_config = base_simulation_config
        self.optimization_config = optimization_config
        self.data_provider = data_provider
        self.strategy_factory = strategy_factory
        self.risk_manager_factory = risk_manager_factory
        self.custom_objective = custom_objective
        
        # Set random seed if provided
        if optimization_config.random_seed is not None:
            np.random.seed(optimization_config.random_seed)
        
        # Ensure output directory exists
        os.makedirs(optimization_config.output_dir, exist_ok=True)
        
        # Store evaluation results
        self.evaluation_history = []
        
        logger.info("Initialized ParameterOptimizer")
        logger.info(f"Parameter spaces: {len(optimization_config.parameter_spaces)}")
        logger.info(f"Target metric: {optimization_config.target_metric}")
        
    def run_grid_search(self) -> OptimizationResult:
        """
        Run grid search optimization across parameter spaces.
        
        Returns:
            OptimizationResult: The optimization results
        """
        logger.info("Starting grid search optimization")
        start_time = time.time()
        
        # Generate all parameter combinations
        param_names = [space.name for space in self.optimization_config.parameter_spaces]
        param_values = [space.values for space in self.optimization_config.parameter_spaces]
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Run simulations for each combination
        results = []
        best_score = float('-inf') if self.optimization_config.higher_is_better else float('inf')
        best_params = None
        
        with ProcessPoolExecutor(max_workers=self.optimization_config.max_parallel_jobs) as executor:
            # Submit all parameter combinations for evaluation
            future_to_params = {}
            
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                future = executor.submit(self._evaluate_parameters, param_dict)
                future_to_params[future] = param_dict
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_params)):
                param_dict = future_to_params[future]
                try:
                    score, metrics = future.result()
                    
                    # Combine parameters and metrics
                    result_entry = param_dict.copy()
                    result_entry.update(metrics)
                    result_entry["score"] = score
                    
                    results.append(result_entry)
                    
                    # Check if this is the best result
                    is_better = (self.optimization_config.higher_is_better and score > best_score) or \
                                (not self.optimization_config.higher_is_better and score < best_score)
                    
                    if is_better:
                        best_score = score
                        best_params = param_dict.copy()
                    
                    # Log progress
                    logger.info(f"Completed {i+1}/{len(param_combinations)} evaluations. " 
                               f"Current best: {best_score:.4f} with params: {best_params}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating parameters {param_dict}: {str(e)}")
                    continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate parameter importance
        param_importance = self._calculate_parameter_importance(results_df)
        
        # Create optimization result
        result = OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            all_results=results_df,
            optimization_time=time.time() - start_time,
            parameter_importance=param_importance
        )
        
        # Save results
        self._save_optimization_results(result)
        
        logger.info(f"Grid search completed in {result.optimization_time:.2f} seconds")
        logger.info(f"Best score: {result.best_score}")
        logger.info(f"Best parameters: {result.best_parameters}")
        
        return result
    
    def run_genetic_algorithm(self) -> OptimizationResult:
        """
        Run genetic algorithm optimization across parameter spaces.
        
        Returns:
            OptimizationResult: The optimization results
        """
        logger.info("Starting genetic algorithm optimization")
        start_time = time.time()
        
        # Get parameter spaces
        param_spaces = self.optimization_config.parameter_spaces
        param_names = [space.name for space in param_spaces]
        
        # Initialize random population
        population = self._initialize_population()
        best_individual = None
        best_score = float('-inf') if self.optimization_config.higher_is_better else float('inf')
        
        # Store all evaluation results
        all_results = []
        
        # Run for specified number of generations
        for generation in range(self.optimization_config.generations):
            logger.info(f"Starting generation {generation+1}/{self.optimization_config.generations}")
            
            # Evaluate all individuals in the population
            fitness_scores = []
            
            with ProcessPoolExecutor(max_workers=self.optimization_config.max_parallel_jobs) as executor:
                future_to_individual = {}
                
                for individual in population:
                    param_dict = dict(zip(param_names, individual))
                    future = executor.submit(self._evaluate_parameters, param_dict)
                    future_to_individual[future] = (individual, param_dict)
                
                # Process results as they complete
                for future in as_completed(future_to_individual):
                    individual, param_dict = future_to_individual[future]
                    try:
                        score, metrics = future.result()
                        
                        # Store result
                        result_entry = param_dict.copy()
                        result_entry.update(metrics)
                        result_entry["score"] = score
                        result_entry["generation"] = generation + 1
                        all_results.append(result_entry)
                        
                        fitness_scores.append(score)
                        
                        # Check if this is the best result
                        is_better = (self.optimization_config.higher_is_better and score > best_score) or \
                                    (not self.optimization_config.higher_is_better and score < best_score)
                        
                        if is_better:
                            best_score = score
                            best_individual = individual
                            best_params = param_dict.copy()
                            
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {param_dict}: {str(e)}")
                        # Assign a very poor fitness score
                        fitness_scores.append(float('-inf') if self.optimization_config.higher_is_better else float('inf'))
            
            # Create next generation
            population = self._create_next_generation(population, fitness_scores)
            
            # Log progress
            logger.info(f"Generation {generation+1} completed. Best score: {best_score:.4f}")
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Calculate parameter importance
        param_importance = self._calculate_parameter_importance(results_df)
        
        # Create optimization result
        result = OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            all_results=results_df,
            optimization_time=time.time() - start_time,
            parameter_importance=param_importance
        )
        
        # Save results
        self._save_optimization_results(result)
        
        logger.info(f"Genetic algorithm completed in {result.optimization_time:.2f} seconds")
        logger.info(f"Best score: {result.best_score}")
        logger.info(f"Best parameters: {result.best_parameters}")
        
        return result
    
    def _initialize_population(self) -> List[List[Any]]:
        """Initialize random population for genetic algorithm"""
        population = []
        param_spaces = self.optimization_config.parameter_spaces
        population_size = self.optimization_config.population_size
        
        for _ in range(population_size):
            individual = []
            for space in param_spaces:
                # Randomly select a value from the parameter space
                value = np.random.choice(space.values)
                individual.append(value)
            population.append(individual)
            
        return population
    
    def _create_next_generation(self, population, fitness_scores):
        """Create next generation using selection, crossover and mutation"""
        next_generation = []
        
        # Elitism: Keep best individual
        if self.optimization_config.higher_is_better:
            best_idx = np.argmax(fitness_scores)
        else:
            best_idx = np.argmin(fitness_scores)
        next_generation.append(population[best_idx])
        
        # Create remaining individuals through selection, crossover, mutation
        while len(next_generation) < len(population):
            # Selection (tournament selection)
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if np.random.random() < self.optimization_config.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            next_generation.append(child1)
            if len(next_generation) < len(population):
                next_generation.append(child2)
                
        return next_generation
    
    def _tournament_selection(self, population, fitness_scores):
        """Tournament selection for genetic algorithm"""
        tournament_size = min(self.optimization_config.tournament_size, len(population))
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        if self.optimization_config.higher_is_better:
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        else:
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def _mutate(self, individual):
        """Mutate an individual"""
        param_spaces = self.optimization_config.parameter_spaces
        
        for i in range(len(individual)):
            if np.random.random() < self.optimization_config.mutation_prob:
                # Replace with a random value from the parameter space
                individual[i] = np.random.choice(param_spaces[i].values)
                
        return individual
    
    def _evaluate_parameters(self, param_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a set of parameters using simulations.
        
        Args:
            param_dict: Dictionary of parameter values to evaluate
            
        Returns:
            Tuple[float, Dict[str, Any]]: Score and metrics dictionary
        """
        logger.debug(f"Evaluating parameters: {param_dict}")
        
        # Create strategy with parameters
        strategy_params = {k: v for k, v in param_dict.items() if k.startswith('strategy_')}
        
        # Create risk manager with parameters if factory provided
        risk_manager = None
        if self.risk_manager_factory is not None:
            risk_params = {k: v for k, v in param_dict.items() if k.startswith('risk_')}
            risk_manager = self.risk_manager_factory(**risk_params)
        
        # Run simulation with resampling if configured
        if self.optimization_config.resample_method:
            scores = []
            all_metrics = []
            
            for _ in range(self.optimization_config.resample_count):
                sim_config = self._create_resampled_config()
                
                # Create simulator
                simulator = TradingSimulator(
                    config=sim_config,
                    data_provider=self.data_provider,
                    risk_manager=risk_manager,
                    strategy_factory=lambda symbol, data_provider, risk_manager: self.strategy_factory(
                        symbol=symbol, data_provider=data_provider, risk_manager=risk_manager, **strategy_params
                    )
                )
                
                # Run simulation
                results = simulator.run_simulation()
                
                # Get performance metrics
                metrics = results["performance_metrics"]
                all_metrics.append(metrics)
                
                # Get target metric
                metric_value = metrics.get(self.optimization_config.target_metric, 0)
                scores.append(metric_value)
            
            # Average scores and metrics
            avg_score = np.mean(scores)
            avg_metrics = {k: np.mean([m.get(k, 0) for m in all_metrics]) for k in all_metrics[0].keys()}
            
            if self.custom_objective:
                # Use custom objective function if provided
                final_score = self.custom_objective(avg_score, avg_metrics, param_dict)
            else:
                final_score = avg_score
                
            return final_score, avg_metrics
            
        else:
            # Create simulator
            simulator = TradingSimulator(
                config=self.base_simulation_config,
                data_provider=self.data_provider,
                risk_manager=risk_manager,
                strategy_factory=lambda symbol, data_provider, risk_manager: self.strategy_factory(
                    symbol=symbol, data_provider=data_provider, risk_manager=risk_manager, **strategy_params
                )
            )
            
            # Run simulation
            results = simulator.run_simulation()
            
            # Get performance metrics
            metrics = results["performance_metrics"]
            
            # Get target metric
            metric_value = metrics.get(self.optimization_config.target_metric, 0)
            
            if self.custom_objective:
                # Use custom objective function if provided
                final_score = self.custom_objective(metric_value, metrics, param_dict)
            else:
                final_score = metric_value
                
            return final_score, metrics
    
    def _create_resampled_config(self) -> SimulationConfig:
        """Create a resampled configuration for robustness testing"""
        config = self.base_simulation_config
        
        if self.optimization_config.resample_method == "bootstrap":
            # Bootstrap resampling - randomly select data points with replacement
            # This would be implemented by modifying the data provider or using a wrapped provider
            # For simplicity, we'll just return the original config here
            return config
            
        elif self.optimization_config.resample_method == "timeseries":
            # Time series resampling - randomly select a continuous time window
            orig_duration = (config.end_date - config.start_date).days
            # Select a random start date within the original range, ensuring same duration
            max_start = config.end_date - datetime.timedelta(days=int(orig_duration * 0.8))
            new_start = config.start_date + datetime.timedelta(
                days=np.random.randint(0, (max_start - config.start_date).days)
            )
            new_end = new_start + datetime.timedelta(days=int(orig_duration * 0.8))
            
            # Create a new config with the resampled dates
            new_config = SimulationConfig(
                mode=config.mode,
                start_date=new_start,
                end_date=new_end,
                initial_capital=config.initial_capital,
                symbols=config.symbols,
                market_scenario=config.market_scenario,
                data_frequency=config.data_frequency,
                slippage_model=config.slippage_model,
                slippage_value=config.slippage_value,
                commission_model=config.commission_model,
                commission_value=config.commission_value,
                enable_fractional_shares=config.enable_fractional_shares,
                random_seed=np.random.randint(0, 10000)  # New random seed for each resample
            )
            
            return new_config
            
        # Default - return original config
        return config
    
    def _calculate_parameter_importance(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate parameter importance based on optimization results.
        
        Uses a simple approach of calculating correlation between parameters and scores.
        More sophisticated methods could be implemented (e.g., random forest feature importance).
        
        Args:
            results_df: DataFrame with all optimization results
            
        Returns:
            Dict[str, float]: Dictionary mapping parameter names to importance scores
        """
        if results_df.empty:
            return {}
            
        param_spaces = self.optimization_config.parameter_spaces
        param_names = [space.name for space in param_spaces]
        
        # Calculate correlations
        correlations = {}
        
        for param in param_names:
            if param in results_df.columns and results_df[param].nunique() > 1:
                # For numeric parameters, use correlation
                if pd.api.types.is_numeric_dtype(results_df[param]):
                    corr = results_df[param].corr(results_df["score"])
                    correlations[param] = abs(corr)  # Use absolute correlation as importance
                else:
                    # For categorical parameters, use ANOVA-like approach
                    # Calculate variance between groups / total variance
                    group_means = results_df.groupby(param)["score"].mean()
                    overall_mean = results_df["score"].mean()
                    
                    # Between-group sum of squares
                    between_ss = sum(len(group) * (mean - overall_mean) ** 2 
                                  for group, mean in zip(results_df.groupby(param).groups.values(), group_means))
                    
                    # Total sum of squares
                    total_ss = sum((score - overall_mean) ** 2 for score in results_df["score"])
                    
                    # R-squared as importance
                    if total_ss > 0:
                        correlations[param] = between_ss / total_ss
                    else:
                        correlations[param] = 0.0
        
        # Normalize importance scores
        total_importance = sum(correlations.values())
        if total_importance > 0:
            normalized_importance = {param: importance / total_importance 
                                    for param, importance in correlations.items()}
        else:
            normalized_importance = {param: 1.0 / len(correlations) for param in correlations}
            
        return normalized_importance
    
    def _save_optimization_results(self, result: OptimizationResult):
        """Save optimization results to files"""
        output_dir = self.optimization_config.output_dir
        timestamp = int(time.time())
        
        # Save best parameters
        best_params_file = os.path.join(output_dir, f"best_params_{timestamp}.json")
        with open(best_params_file, "w") as f:
            json.dump({
                "best_parameters": result.best_parameters,
                "best_score": result.best_score,
                "parameter_importance": result.parameter_importance,
                "optimization_time": result.optimization_time
            }, f, indent=2)
        
        # Save all results if configured
        if self.optimization_config.save_all_results and not result.all_results.empty:
            all_results_file = os.path.join(output_dir, f"all_results_{timestamp}.csv")
            result.all_results.to_csv(all_results_file, index=False)
            
        # Generate and save visualization
        self._generate_optimization_plots(result, output_dir, timestamp)
        
        logger.info(f"Saved optimization results to {output_dir}")
    
    def _generate_optimization_plots(self, result: OptimizationResult, output_dir: str, timestamp: int):
        """Generate and save optimization visualization plots"""
        if result.all_results.empty:
            return
            
        # Parameter importance plot
        if result.parameter_importance:
            fig, ax = plt.subplots(figsize=(10, 6))
            params = list(result.parameter_importance.keys())
            importance = list(result.parameter_importance.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importance)
            params = [params[i] for i in sorted_idx]
            importance = [importance[i] for i in sorted_idx]
            
            ax.barh(params, importance)
            ax.set_xlabel("Relative Importance")
            ax.set_title("Parameter Importance")
            plt.tight_layout()
            
            importance_plot_file = os.path.join(output_dir, f"parameter_importance_{timestamp}.png")
            plt.savefig(importance_plot_file)
            plt.close(fig)
        
        # Scatter plot matrix for parameters vs. score
        param_spaces = self.optimization_config.parameter_spaces
        param_names = [space.name for space in param_spaces]
        
        # Select only numeric parameters for scatter plots
        numeric_params = [p for p in param_names if pd.api.types.is_numeric_dtype(result.all_results[p])]
        
        if numeric_params:
            # Create scatter plots for each numeric parameter vs. score
            fig, axes = plt.subplots(len(numeric_params), 1, figsize=(10, 5 * len(numeric_params)))
            
            if len(numeric_params) == 1:
                axes = [axes]
                
            for i, param in enumerate(numeric_params):
                ax = axes[i]
                ax.scatter(result.all_results[param], result.all_results["score"], alpha=0.7)
                ax.set_xlabel(param)
                ax.set_ylabel("Score")
                ax.set_title(f"{param} vs. Score")
                ax.grid(True)
                
            plt.tight_layout()
            
            scatter_plot_file = os.path.join(output_dir, f"parameter_scatter_{timestamp}.png")
            plt.savefig(scatter_plot_file)
            plt.close(fig)
            
        # Convergence plot for genetic algorithm
        if "generation" in result.all_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by generation and get best score
            if self.optimization_config.higher_is_better:
                gen_best = result.all_results.groupby("generation")["score"].max()
            else:
                gen_best = result.all_results.groupby("generation")["score"].min()
                
            ax.plot(gen_best.index, gen_best.values, marker='o')
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best Score")
            ax.set_title("Optimization Convergence")
            ax.grid(True)
            
            convergence_plot_file = os.path.join(output_dir, f"convergence_{timestamp}.png")
            plt.savefig(convergence_plot_file)
            plt.close(fig) 