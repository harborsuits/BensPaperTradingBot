#!/usr/bin/env python3
"""
Evolve Funded Strategies - Specialized evolution targeting funded account requirements

This module evolves trading strategies specifically optimized to pass funded account
evaluation criteria: 5% max drawdown, 8-10% profit target, and 3% max daily loss.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
import logging
from collections import defaultdict

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import our modules
from advanced_strategies import (
    MeanReversionStrategy, 
    MomentumStrategy, 
    VolumeProfileStrategy, 
    VolatilityBreakoutStrategy,
    backtest_strategy
)
from strategy_registry import StrategyRegistry
from synthetic_market_generator import SyntheticMarketGenerator
from funded_account_evaluator import FundedAccountEvaluator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/evolve_funded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger('evolve_funded_strategies')


# Advanced strategy factory
ADVANCED_STRATEGY_TYPES = {
    "MeanReversion": MeanReversionStrategy,
    "Momentum": MomentumStrategy,
    "VolumeProfile": VolumeProfileStrategy,
    "VolatilityBreakout": VolatilityBreakoutStrategy
}


def create_advanced_strategy(strategy_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create an instance of an advanced strategy.
    
    Args:
        strategy_type: Type of strategy to create
        params: Parameters for the strategy
        
    Returns:
        Strategy instance
    """
    if strategy_type not in ADVANCED_STRATEGY_TYPES:
        valid_types = ", ".join(ADVANCED_STRATEGY_TYPES.keys())
        raise ValueError(f"Invalid strategy type: {strategy_type}. Valid types: {valid_types}")
    
    strategy_class = ADVANCED_STRATEGY_TYPES[strategy_type]
    
    if params:
        return strategy_class(**params)
    else:
        return strategy_class()


def get_parameter_bounds(strategy_type: str) -> Dict[str, Tuple[Any, Any]]:
    """
    Get parameter bounds for a given strategy type.
    
    Args:
        strategy_type: Type of strategy
        
    Returns:
        Dictionary of parameter bounds
    """
    bounds = {}
    
    if strategy_type == "MeanReversion":
        bounds = {
            "lookback_period": (10, 50),
            "entry_std": (1.0, 3.0),
            "exit_std": (0.1, 1.0),
            "smoothing": (1, 5)
        }
    elif strategy_type == "Momentum":
        bounds = {
            "short_period": (5, 30),
            "medium_period": (20, 60),
            "long_period": (50, 120),
            "threshold": (0.01, 0.05),
            "smoothing": (1, 5)
        }
    elif strategy_type == "VolumeProfile":
        bounds = {
            "lookback_period": (10, 50),
            "volume_threshold": (1.0, 3.0),
            "price_levels": (10, 50),
            "smoothing": (1, 5)
        }
    elif strategy_type == "VolatilityBreakout":
        bounds = {
            "atr_period": (5, 30),
            "breakout_multiple": (1.0, 3.0),
            "lookback_period": (3, 10),
            "filter_threshold": (0.05, 0.3)
        }
    
    return bounds


def random_params_for_strategy(strategy_type: str) -> Dict[str, Any]:
    """
    Generate random parameters for a given strategy type.
    
    Args:
        strategy_type: Type of strategy
        
    Returns:
        Dictionary of random parameters
    """
    bounds = get_parameter_bounds(strategy_type)
    params = {}
    
    for param, (min_val, max_val) in bounds.items():
        if isinstance(min_val, int) and isinstance(max_val, int):
            params[param] = np.random.randint(min_val, max_val + 1)
        else:
            params[param] = min_val + np.random.random() * (max_val - min_val)
    
    return params


def mutate_strategy_params(strategy_type: str, params: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
    """
    Mutate parameters of a strategy.
    
    Args:
        strategy_type: Type of strategy
        params: Original parameters
        mutation_rate: Probability of mutating each parameter
        
    Returns:
        Mutated parameters
    """
    bounds = get_parameter_bounds(strategy_type)
    mutated = params.copy()
    
    for param, (min_val, max_val) in bounds.items():
        # Only mutate with probability mutation_rate
        if np.random.random() < mutation_rate:
            if isinstance(min_val, int) and isinstance(max_val, int):
                # For integer parameters, use integer arithmetic
                change = np.random.randint(-3, 4)  # -3 to +3
                new_val = mutated[param] + change
                mutated[param] = max(min_val, min(max_val, new_val))
            else:
                # For float parameters, scale change by parameter range
                scale = (max_val - min_val) * 0.2
                change = (np.random.random() - 0.5) * scale
                new_val = mutated[param] + change
                mutated[param] = max(min_val, min(max_val, new_val))
    
    return mutated


def calculate_funded_fitness(evaluation: Dict[str, Any]) -> float:
    """
    Calculate fitness score for a strategy, based on funded account evaluation.
    
    Args:
        evaluation: Funded account evaluation results
        
    Returns:
        Fitness score (higher is better)
    """
    # Use the evaluator score as base
    fitness = evaluation["score"]
    
    # Huge bonus for passing the evaluation
    if evaluation["passes_evaluation"]:
        fitness += 50
    
    # Add penalties for specific threshold failures
    if not evaluation["threshold_results"]["meets_max_drawdown"]:
        fitness -= 30  # Severe penalty for exceeding drawdown limit
    
    if not evaluation["threshold_results"]["meets_daily_loss"]:
        fitness -= 20  # Heavy penalty for exceeding daily loss limit
    
    if not evaluation["threshold_results"]["meets_profit_target"]:
        fitness -= 15  # Penalty for not meeting profit target
    
    # Bonus for exceeding profit target
    if evaluation["threshold_results"].get("exceeds_profit_target", False):
        fitness += 15
    
    # Penalty for not meeting minimum trading days
    if not evaluation["threshold_results"]["meets_min_trading_days"]:
        fitness -= 10
    
    return fitness


def tournament_selection(population: List[Dict[str, Any]], tournament_size: int) -> Dict[str, Any]:
    """
    Select an individual using tournament selection.
    
    Args:
        population: Population to select from
        tournament_size: Size of tournament
        
    Returns:
        Selected individual
    """
    tournament = np.random.choice(len(population), tournament_size, replace=False)
    tournament_individuals = [population[i] for i in tournament]
    return max(tournament_individuals, key=lambda x: x["fitness"])


def crossover_params(strategy_type: str, params_a: Dict[str, Any], params_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new parameter set by crossing over two parameter sets.
    
    Args:
        strategy_type: Type of strategy
        params_a: First parameter set
        params_b: Second parameter set
        
    Returns:
        New parameter set
    """
    # Get common parameters between both sets
    bounds = get_parameter_bounds(strategy_type)
    common_params = set(bounds.keys()).intersection(params_a.keys()).intersection(params_b.keys())
    
    # Initialize new parameters with params_a
    new_params = params_a.copy()
    
    # For each common parameter, randomly select from either parent
    for param in common_params:
        if np.random.random() < 0.5:
            new_params[param] = params_b[param]
    
    return new_params


def evolve_funded_strategies(
    evaluator: FundedAccountEvaluator,
    registry: StrategyRegistry,
    market_type: str = "mixed",
    period_days: int = 21,
    population_size: int = 50,
    generations: int = 20,
    strategy_types: Optional[List[str]] = None,
    elite_percentage: float = 0.2,
    crossover_rate: float = 0.3,
    mutation_rate: float = 0.3,
    tournament_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Evolve a population of strategies optimized for funded accounts.
    
    Args:
        evaluator: Funded account evaluator
        registry: Strategy registry
        market_type: Type of market to evolve on
        period_days: Number of trading days in evaluation period
        population_size: Size of the population
        generations: Number of generations to evolve
        strategy_types: Types of strategies to include, defaults to all
        elite_percentage: Percentage of population to keep as elite
        crossover_rate: Rate of crossover for new strategies
        mutation_rate: Rate of mutation for parameters
        tournament_size: Size of tournament for selection
        
    Returns:
        List of top strategies
    """
    logger.info(f"Starting funded strategy evolution for {market_type} market")
    
    # Generate market data for consistency in evaluation
    market_data = evaluator.generate_evaluation_period(period_days, market_type)
    
    if strategy_types is None:
        strategy_types = list(ADVANCED_STRATEGY_TYPES.keys())
    
    # Validate strategy types
    for strategy_type in strategy_types:
        if strategy_type not in ADVANCED_STRATEGY_TYPES:
            valid_types = ", ".join(ADVANCED_STRATEGY_TYPES.keys())
            raise ValueError(f"Invalid strategy type: {strategy_type}. Valid types: {valid_types}")
    
    # Create output directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"funded_evolution_{market_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize population
    population = []
    
    for _ in range(population_size):
        # Select random strategy type
        strategy_type = np.random.choice(strategy_types)
        
        # Generate random parameters
        params = random_params_for_strategy(strategy_type)
        
        # Create strategy
        strategy = create_advanced_strategy(strategy_type, params)
        
        # Evaluate against funded account criteria
        evaluation = evaluator.evaluate_strategy(strategy, market_data)
        
        # Calculate fitness
        fitness = calculate_funded_fitness(evaluation)
        
        # Add to population
        population.append({
            "strategy_type": strategy_type,
            "parameters": params,
            "evaluation": evaluation,
            "fitness": fitness
        })
    
    # Sort population by fitness
    population.sort(key=lambda x: x["fitness"], reverse=True)
    
    # Register initial population
    for i, individual in enumerate(population):
        registry.register_strategy(
            strategy_type=individual["strategy_type"],
            parameters=individual["parameters"],
            performance=individual["evaluation"]["metrics"],
            scenario=market_type,
            generation=0,
            fitness=individual["fitness"]
        )
    
    logger.info(f"Initial population fitness: {[round(ind['fitness'], 2) for ind in population[:5]]}")
    
    # Track best fitness per generation
    best_fitness_history = [population[0]['fitness']]
    avg_fitness_history = [np.mean([ind['fitness'] for ind in population])]
    
    # Track passing percentage
    passing_history = [
        sum(1 for ind in population if ind['evaluation']['passes_evaluation']) / len(population)
    ]
    
    # Set counters
    generations_without_improvement = 0
    max_generations_without_improvement = 10  # Stop if no improvement after this many generations
    best_fitness_so_far = population[0]['fitness']
    
    # Evolve for specified generations
    for generation in range(1, generations + 1):
        logger.info(f"Starting generation {generation}")
        
        # Calculate elite count
        elite_count = max(1, int(population_size * elite_percentage))
        
        # Keep elite individuals
        new_population = population[:elite_count]
        
        # Create new individuals until population is full
        while len(new_population) < population_size:
            # Selection
            if np.random.random() < crossover_rate:
                # Crossover
                parent_a = tournament_selection(population, tournament_size)
                parent_b = tournament_selection(population, tournament_size)
                
                # Ensure different parents
                while parent_b["strategy_type"] == parent_a["strategy_type"] and len(population) > 1:
                    parent_b = tournament_selection(population, tournament_size)
                
                # Create child through crossover (same strategy type as parent_a)
                child_type = parent_a["strategy_type"]
                
                # Crossover parameters
                child_params = crossover_params(
                    child_type, 
                    parent_a["parameters"],
                    parent_b["parameters"]
                )
                
                # Apply mutation
                child_params = mutate_strategy_params(
                    child_type, 
                    child_params, 
                    mutation_rate
                )
                
                # Create strategy with new parameters
                child_strategy = create_advanced_strategy(child_type, child_params)
            else:
                # Mutation only
                parent = tournament_selection(population, tournament_size)
                
                # Create child through mutation
                child_type = parent["strategy_type"]
                child_params = mutate_strategy_params(
                    child_type, 
                    parent["parameters"], 
                    mutation_rate
                )
                
                # Create strategy with new parameters
                child_strategy = create_advanced_strategy(child_type, child_params)
            
            # Evaluate against funded account criteria
            child_evaluation = evaluator.evaluate_strategy(child_strategy, market_data)
            
            # Calculate fitness
            child_fitness = calculate_funded_fitness(child_evaluation)
            
            # Add to new population
            new_population.append({
                "strategy_type": child_type,
                "parameters": child_params,
                "evaluation": child_evaluation,
                "fitness": child_fitness
            })
        
        # Sort new population by fitness
        new_population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Register new population
        for individual in new_population:
            registry.register_strategy(
                strategy_type=individual["strategy_type"],
                parameters=individual["parameters"],
                performance=individual["evaluation"]["metrics"],
                scenario=market_type,
                generation=generation,
                fitness=individual["fitness"]
            )
        
        # Update population
        population = new_population
        
        # Update history
        best_fitness_history.append(population[0]['fitness'])
        avg_fitness_history.append(np.mean([ind['fitness'] for ind in population]))
        
        passing_history.append(
            sum(1 for ind in population if ind['evaluation']['passes_evaluation']) / len(population)
        )
        
        # Log progress
        logger.info(f"Generation {generation} top fitness: {population[0]['fitness']:.2f}")
        logger.info(f"Passing percentage: {passing_history[-1]:.1%}")
        
        # Check if we have strategies that pass funded evaluation
        if passing_history[-1] > 0:
            passing_strategies = [ind for ind in population if ind['evaluation']['passes_evaluation']]
            logger.info(f"Found {len(passing_strategies)} strategies that pass funded evaluation")
            
            # If we have a reasonable number of passing strategies, we can stop
            if len(passing_strategies) >= 5:
                logger.info(f"Success! Found {len(passing_strategies)} strategies that pass funded evaluation")
                break
        
        # Check for improvement
        if population[0]['fitness'] > best_fitness_so_far:
            best_fitness_so_far = population[0]['fitness']
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # Early stopping if no improvement
        if generations_without_improvement >= max_generations_without_improvement:
            logger.info(f"No improvement for {max_generations_without_improvement} generations. Stopping early.")
            break
    
    logger.info("Evolution complete")
    
    # Save evolution history
    history = {
        "best_fitness": best_fitness_history,
        "avg_fitness": avg_fitness_history,
        "passing_percentage": passing_history,
        "generations": len(best_fitness_history)
    }
    
    # Save history to file
    with open(os.path.join(output_dir, "evolution_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # Plot evolution progress
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(best_fitness_history, label="Best Fitness")
    plt.plot(avg_fitness_history, label="Average Fitness")
    plt.title(f"Fitness History - {market_type}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(passing_history, label="Passing Percentage")
    plt.title("Population Meeting Funded Account Criteria")
    plt.xlabel("Generation")
    plt.ylabel("Percentage")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "evolution_history.png"))
    plt.close()
    
    # Save top strategies
    top_strategies = population[:10]  # Top 10 strategies
    
    with open(os.path.join(output_dir, "top_strategies.json"), "w") as f:
        strategies_data = []
        
        for strategy in top_strategies:
            strategy_data = {
                "strategy_type": strategy["strategy_type"],
                "parameters": strategy["parameters"],
                "fitness": strategy["fitness"],
                "passes_evaluation": strategy["evaluation"]["passes_evaluation"],
                "score": strategy["evaluation"]["score"],
                "metrics": strategy["evaluation"]["metrics"]
            }
            
            strategies_data.append(strategy_data)
        
        json.dump(strategies_data, f, indent=2)
    
    return population


def multi_environment_evolution(
    registry_path: str = "./strategy_registry",
    output_dir: str = "./funded_evolution_results",
    population_size: int = 50,
    generations: int = 20,
    strategy_types: Optional[List[str]] = None
):
    """
    Run funded account evolution across multiple market environments.
    
    Args:
        registry_path: Path to strategy registry
        output_dir: Directory for output
        population_size: Size of population for each scenario
        generations: Number of generations to evolve
        strategy_types: List of strategy types to include
    """
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"multi_environment_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize registry
    registry = StrategyRegistry(registry_path)
    
    # Initialize funded account evaluator
    evaluator = FundedAccountEvaluator()
    
    # Market environments to test
    environments = ["bull", "bear", "volatile", "mixed"]
    
    # Run evolution on each environment
    all_top_strategies = {}
    
    for env in environments:
        logger.info(f"Starting evolution for environment: {env}")
        
        # Evolve strategies for this environment
        population = evolve_funded_strategies(
            evaluator=evaluator,
            registry=registry,
            market_type=env,
            population_size=population_size,
            generations=generations,
            strategy_types=strategy_types
        )
        
        # Extract passing strategies
        passing_strategies = [ind for ind in population if ind['evaluation']['passes_evaluation']]
        
        # Sort by fitness
        passing_strategies.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Store top strategies
        all_top_strategies[env] = passing_strategies[:5]  # Top 5 passing strategies
        
        logger.info(f"Completed evolution for environment: {env}")
    
    # Find robust strategies across environments
    # A robust strategy is one that passes in multiple environments
    robust_strategies = []
    
    # Collect all strategies across environments
    strategy_performance = defaultdict(dict)
    
    for env, strategies in all_top_strategies.items():
        for strategy in strategies:
            # Create a strategy identifier
            strategy_id = f"{strategy['strategy_type']}-{hash(str(strategy['parameters']))}"
            
            # Record performance in this environment
            strategy_performance[strategy_id][env] = {
                "fitness": strategy["fitness"],
                "passes": strategy["evaluation"]["passes_evaluation"],
                "score": strategy["evaluation"]["score"]
            }
            
            # Add strategy data if not already present
            if strategy_id not in [s["id"] for s in robust_strategies]:
                robust_strategies.append({
                    "id": strategy_id,
                    "strategy_type": strategy["strategy_type"],
                    "parameters": strategy["parameters"],
                    "environments": [],
                    "passing_count": 0,
                    "average_score": 0
                })
    
    # Analyze performance across environments
    for strategy in robust_strategies:
        strategy_id = strategy["id"]
        
        # Count environments where strategy passes
        passing_envs = []
        total_score = 0
        env_count = 0
        
        for env, performance in strategy_performance[strategy_id].items():
            if performance["passes"]:
                passing_envs.append(env)
            
            total_score += performance["score"]
            env_count += 1
        
        strategy["environments"] = passing_envs
        strategy["passing_count"] = len(passing_envs)
        strategy["average_score"] = total_score / env_count if env_count > 0 else 0
    
    # Sort by number of environments passed, then by average score
    robust_strategies.sort(
        key=lambda x: (x["passing_count"], x["average_score"]), 
        reverse=True
    )
    
    # Save robust strategies
    with open(os.path.join(run_dir, "robust_strategies.json"), "w") as f:
        json.dump(robust_strategies, f, indent=2)
    
    # Generate summary report
    report = []
    report.append("=========================================")
    report.append("FUNDED ACCOUNT MULTI-ENVIRONMENT SUMMARY")
    report.append("=========================================")
    report.append("")
    
    # Report on each environment
    for env in environments:
        passing_count = len(all_top_strategies[env])
        report.append(f"{env.upper()} ENVIRONMENT: {passing_count} passing strategies")
        
        if passing_count > 0:
            top_strategy = all_top_strategies[env][0]
            report.append(f"  Top Strategy: {top_strategy['strategy_type']}")
            report.append(f"  Fitness: {top_strategy['fitness']:.2f}")
            report.append(f"  Score: {top_strategy['evaluation']['score']:.2f}")
            report.append(f"  Return: {top_strategy['evaluation']['metrics']['total_return_pct']:.2f}%")
            report.append(f"  Max Drawdown: {top_strategy['evaluation']['metrics']['max_drawdown']:.2f}%")
            report.append(f"  Worst Daily Loss: {top_strategy['evaluation']['metrics']['worst_daily_loss']:.2f}%")
        
        report.append("")
    
    # Report on robust strategies
    report.append("ROBUST STRATEGIES (Passing in Multiple Environments):")
    
    for i, strategy in enumerate(robust_strategies[:5]):  # Top 5 robust strategies
        if strategy["passing_count"] > 1:
            report.append(f"{i+1}. {strategy['strategy_type']}")
            report.append(f"   Passes in {strategy['passing_count']} environments: {', '.join(strategy['environments'])}")
            report.append(f"   Average Score: {strategy['average_score']:.2f}")
            report.append("")
    
    report.append("=========================================")
    
    # Save summary report
    with open(os.path.join(run_dir, "summary_report.txt"), "w") as f:
        f.write("\n".join(report))
    
    # Print summary
    print("\n".join(report))
    
    logger.info(f"Multi-environment evolution complete. Results saved to {run_dir}")
    
    return robust_strategies


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evolve strategies for funded accounts")
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default="./strategy_registry",
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./funded_evolution_results",
        help="Output directory"
    )
    
    parser.add_argument(
        "--population", 
        type=int, 
        default=100,  # Larger population for funded account evolution
        help="Population size"
    )
    
    parser.add_argument(
        "--generations", 
        type=int, 
        default=30,  # More generations to find strategies that pass strict criteria
        help="Number of generations"
    )
    
    parser.add_argument(
        "--environment", 
        type=str, 
        default=None,
        choices=["bull", "bear", "volatile", "sideways", "mixed"],
        help="Specific environment to evolve in (if not specified, will run multi-environment evolution)"
    )
    
    parser.add_argument(
        "--strategies", 
        type=str, 
        nargs="+",
        default=None,
        help="Strategy types to include (MeanReversion, Momentum, VolumeProfile, VolatilityBreakout)"
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    if args.environment:
        # Run evolution for specific environment
        registry = StrategyRegistry(args.registry)
        evaluator = FundedAccountEvaluator()
        
        evolve_funded_strategies(
            evaluator=evaluator,
            registry=registry,
            market_type=args.environment,
            population_size=args.population,
            generations=args.generations,
            strategy_types=args.strategies
        )
    else:
        # Run multi-environment evolution
        multi_environment_evolution(
            registry_path=args.registry,
            output_dir=args.output,
            population_size=args.population,
            generations=args.generations,
            strategy_types=args.strategies
        )
