#!/usr/bin/env python3
"""
Enhanced Evolution With Thresholds - Combines advanced strategies, strict evaluation, and BenBot data recording

This module enhances the evolutionary process with strict performance thresholds
and records detailed performance data for the BenBot system.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import argparse
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
from strategy_evaluator import StrategyEvaluator
from benbot_data_formatter import BenBotDataFormatter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/enhanced_evolution_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger('enhanced_evolution')


# Performance thresholds (aggressive targets based on user requirements)
PERFORMANCE_THRESHOLDS = {
    "min_return_pct": 10.0,           # Minimum return percentage
    "max_drawdown_pct": 15.0,         # Maximum drawdown percentage
    "min_win_rate_pct": 70.0,         # Minimum win rate percentage (aiming for 80%)
    "min_trade_count": 20,            # Minimum number of trades
    "min_profit_factor": 1.5,         # Minimum profit factor
    "min_sharpe_ratio": 1.0,          # Minimum Sharpe ratio
    "max_consistency_score": 0.7,     # Maximum equity curve consistency score (lower is better)
}


# Advanced strategy factory for creating strategy instances
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


def calculate_fitness(strategy_results: Dict[str, Any], evaluator: StrategyEvaluator) -> float:
    """
    Calculate fitness score for a strategy, using the evaluator for scoring.
    
    Args:
        strategy_results: Backtest results
        evaluator: Strategy evaluator
        
    Returns:
        Fitness score
    """
    # Use evaluator to get a standardized score
    evaluation = evaluator.evaluate_strategy(strategy_results)
    
    # Get score from evaluation 
    score = evaluation["score"]
    
    # Add bonus for meeting thresholds
    if evaluation["meets_thresholds"]:
        score += 20  # Significant bonus for meeting all thresholds
    
    # Add penalties for specific threshold failures
    if not evaluation["threshold_results"]["meets_win_rate"]:
        score -= 15  # Heavy penalty for low win rate
    
    if not evaluation["threshold_results"]["meets_return"]:
        score -= 10  # Penalty for low returns
    
    if not evaluation["threshold_results"]["meets_drawdown"]:
        score -= 10  # Penalty for high drawdown
    
    return score


def evolve_with_thresholds(
    market_data: pd.DataFrame,
    registry: StrategyRegistry,
    scenario_name: str,
    population_size: int = 50,
    generations: int = 20,
    strategy_types: Optional[List[str]] = None,
    elite_percentage: float = 0.2,
    crossover_rate: float = 0.3,
    mutation_rate: float = 0.3,
    tournament_size: int = 3,
    evaluator: Optional[StrategyEvaluator] = None,
    benbot_formatter: Optional[BenBotDataFormatter] = None,
    min_score: float = 70.0
) -> List[Dict[str, Any]]:
    """
    Evolve a population of advanced strategies with threshold-based evaluation.
    
    Args:
        market_data: Market data for backtesting
        registry: Strategy registry for storing strategies
        scenario_name: Name of the market scenario
        population_size: Size of the population
        generations: Number of generations to evolve
        strategy_types: Types of strategies to include, defaults to all
        elite_percentage: Percentage of population to keep as elite
        crossover_rate: Rate of crossover for new strategies
        mutation_rate: Rate of mutation for parameters
        tournament_size: Size of tournament for selection
        evaluator: Strategy evaluator instance
        benbot_formatter: BenBot data formatter instance
        min_score: Minimum score threshold
        
    Returns:
        List of top strategies
    """
    logger.info(f"Starting threshold-based evolution for {scenario_name}")
    logger.info(f"Market data shape: {market_data.shape}")
    
    # Create evaluator if not provided
    if evaluator is None:
        evaluator = StrategyEvaluator(thresholds=PERFORMANCE_THRESHOLDS)
    
    # Create BenBot formatter if not provided
    if benbot_formatter is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        benbot_dir = f"benbot_data/evolution_{timestamp}"
        benbot_formatter = BenBotDataFormatter(benbot_dir)
    
    if strategy_types is None:
        strategy_types = list(ADVANCED_STRATEGY_TYPES.keys())
    
    # Validate strategy types
    for strategy_type in strategy_types:
        if strategy_type not in ADVANCED_STRATEGY_TYPES:
            valid_types = ", ".join(ADVANCED_STRATEGY_TYPES.keys())
            raise ValueError(f"Invalid strategy type: {strategy_type}. Valid types: {valid_types}")
    
    # Initialize population
    population = []
    
    for _ in range(population_size):
        # Select random strategy type
        strategy_type = np.random.choice(strategy_types)
        
        # Generate random parameters
        params = random_params_for_strategy(strategy_type)
        
        # Create strategy
        strategy = create_advanced_strategy(strategy_type, params)
        
        # Backtest strategy
        results = backtest_strategy(strategy, market_data)
        
        # Evaluate strategy
        evaluation = evaluator.evaluate_strategy(results)
        
        # Calculate fitness
        fitness = calculate_fitness(results, evaluator)
        
        # Add to population
        population.append({
            "strategy_type": strategy_type,
            "parameters": params,
            "results": results,
            "evaluation": evaluation,
            "fitness": fitness
        })
        
        # Record in BenBot format
        strategy_id = f"{strategy_type}-{hash(str(params)) % 100000}"
        
        benbot_record = benbot_formatter.format_strategy_record(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            parameters=params,
            performance_metrics=evaluation["metrics"],
            generation=0
        )
        
        benbot_formatter.add_strategy_record(benbot_record)
    
    # Sort population by fitness
    population.sort(key=lambda x: x["fitness"], reverse=True)
    
    # Register initial population
    for individual in population:
        registry.register_strategy(
            strategy_type=individual["strategy_type"],
            parameters=individual["parameters"],
            performance=individual["results"],
            scenario=scenario_name,
            generation=0
        )
    
    logger.info(f"Initial population fitness: {[round(ind['fitness'], 2) for ind in population[:5]]}")
    logger.info(f"Initial top score: {population[0]['evaluation']['score']}")
    
    # Track best fitness per generation
    best_fitness_history = [population[0]['fitness']]
    best_score_history = [population[0]['evaluation']['score']]
    avg_fitness_history = [np.mean([ind['fitness'] for ind in population])]
    
    # Track threshold achievement
    passing_threshold_history = [
        sum(1 for ind in population if ind['evaluation']['meets_thresholds']) / len(population)
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
                
                # Create child through crossover
                child_type = parent_a["strategy_type"]  # Use parent_a's type
                
                # Crossover parameters from both parents
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
                
                # Record parent IDs for lineage tracking
                parent_ids = [
                    f"{parent_a['strategy_type']}-{hash(str(parent_a['parameters'])) % 100000}",
                    f"{parent_b['strategy_type']}-{hash(str(parent_b['parameters'])) % 100000}"
                ]
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
                
                # Record parent ID for lineage tracking
                parent_ids = [
                    f"{parent['strategy_type']}-{hash(str(parent['parameters'])) % 100000}"
                ]
            
            # Backtest child
            child_results = backtest_strategy(child_strategy, market_data)
            
            # Evaluate child
            child_evaluation = evaluator.evaluate_strategy(child_results)
            
            # Calculate fitness
            child_fitness = calculate_fitness(child_results, evaluator)
            
            # Add to new population
            new_population.append({
                "strategy_type": child_type,
                "parameters": child_params,
                "results": child_results,
                "evaluation": child_evaluation,
                "fitness": child_fitness
            })
            
            # Record in BenBot format
            child_id = f"{child_type}-{hash(str(child_params)) % 100000}"
            
            benbot_record = benbot_formatter.format_strategy_record(
                strategy_id=child_id,
                strategy_type=child_type,
                parameters=child_params,
                performance_metrics=child_evaluation["metrics"],
                generation=generation,
                parent_ids=parent_ids
            )
            
            benbot_formatter.add_strategy_record(benbot_record)
        
        # Sort new population by fitness
        new_population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Register new population
        for individual in new_population:
            registry.register_strategy(
                strategy_type=individual["strategy_type"],
                parameters=individual["parameters"],
                performance=individual["results"],
                scenario=scenario_name,
                generation=generation
            )
        
        # Update population
        population = new_population
        
        # Update history
        best_fitness_history.append(population[0]['fitness'])
        best_score_history.append(population[0]['evaluation']['score'])
        avg_fitness_history.append(np.mean([ind['fitness'] for ind in population]))
        
        passing_threshold_history.append(
            sum(1 for ind in population if ind['evaluation']['meets_thresholds']) / len(population)
        )
        
        # Log progress
        logger.info(f"Generation {generation} top fitness: {population[0]['fitness']:.2f}")
        logger.info(f"Generation {generation} top score: {population[0]['evaluation']['score']:.2f}")
        logger.info(f"Threshold achievement: {passing_threshold_history[-1]:.1%}")
        
        # Check if we've reached our aggressive thresholds
        if passing_threshold_history[-1] > 0.2:  # At least 20% of population meets all thresholds
            logger.info(f"Success! {passing_threshold_history[-1]:.1%} of population meets all thresholds")
            logger.info("Evolution completed successfully")
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
        "best_score": best_score_history,
        "avg_fitness": avg_fitness_history,
        "passing_threshold": passing_threshold_history,
        "generations": len(best_fitness_history)
    }
    
    # Save history to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = f"evolution_history"
    os.makedirs(history_dir, exist_ok=True)
    
    with open(os.path.join(history_dir, f"{scenario_name}_{timestamp}.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # Save BenBot records
    benbot_formatter.save_all_records()
    
    # Plot evolution progress
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(best_fitness_history, label="Best Fitness")
    plt.plot(avg_fitness_history, label="Average Fitness")
    plt.title(f"Fitness History - {scenario_name}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(passing_threshold_history, label="Threshold Achievement")
    plt.title("Population Meeting Thresholds")
    plt.xlabel("Generation")
    plt.ylabel("Percentage")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(history_dir, f"{scenario_name}_{timestamp}.png"))
    plt.close()
    
    return population


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


def run_multi_scenario_evolution_with_thresholds(
    registry_path: str = "./strategy_registry",
    output_dir: str = "./threshold_evolution_results",
    population_size: int = 50,
    generations: int = 20,
    scenarios: Optional[List[str]] = None,
    strategy_types: Optional[List[str]] = None
):
    """
    Run evolution with thresholds on multiple market scenarios.
    
    Args:
        registry_path: Path to strategy registry
        output_dir: Directory for output
        population_size: Size of population for each scenario
        generations: Number of generations to evolve
        scenarios: List of scenarios to run on
        strategy_types: List of strategy types to include
    """
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"multi_scenario_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize registry
    registry = StrategyRegistry(registry_path)
    
    # Initialize market generator
    generator = SyntheticMarketGenerator()
    
    # Initialize evaluator
    evaluator = StrategyEvaluator(thresholds=PERFORMANCE_THRESHOLDS)
    
    # Initialize BenBot formatter
    benbot_formatter = BenBotDataFormatter(os.path.join(run_dir, "benbot_data"))
    
    # Default scenarios if none provided
    if scenarios is None:
        scenarios = [
            "bull_market", "bear_market", "sideways_market",
            "volatile_market", "flash_crash", "sector_rotation"
        ]
    
    # Run evolution on each scenario
    all_top_strategies = {}
    
    for scenario in scenarios:
        logger.info(f"Starting evolution for scenario: {scenario}")
        
        # Generate synthetic market data
        if scenario == "bull_market":
            market_data = generator.generate_bull_market()
        elif scenario == "bear_market":
            market_data = generator.generate_bear_market()
        elif scenario == "sideways_market":
            market_data = generator.generate_sideways_market()
        elif scenario == "volatile_market":
            market_data = generator.generate_volatile_market()
        elif scenario == "flash_crash":
            market_data = generator.generate_flash_crash()
        elif scenario == "sector_rotation":
            market_data = generator.generate_sector_rotation()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Create scenario directory
        scenario_dir = os.path.join(run_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Save market data
        market_data.to_csv(os.path.join(scenario_dir, "market_data.csv"))
        
        # Evolve strategies on this scenario
        top_strategies = evolve_with_thresholds(
            market_data=market_data,
            registry=registry,
            scenario_name=scenario,
            population_size=population_size,
            generations=generations,
            strategy_types=strategy_types,
            evaluator=evaluator,
            benbot_formatter=benbot_formatter
        )
        
        # Save top strategies
        all_top_strategies[scenario] = top_strategies[:5]  # Top 5 strategies
        
        # Save top strategies
        with open(os.path.join(scenario_dir, "top_strategies.json"), "w") as f:
            top_data = [
                {
                    "strategy_type": s["strategy_type"],
                    "parameters": s["parameters"],
                    "fitness": s["fitness"],
                    "score": s["evaluation"]["score"],
                    "meets_thresholds": s["evaluation"]["meets_thresholds"],
                    "metrics": s["evaluation"]["metrics"]
                }
                for s in top_strategies[:5]
            ]
            json.dump(top_data, f, indent=2)
        
        logger.info(f"Completed evolution for scenario: {scenario}")
    
    # Generate cross-scenario evaluation
    cross_eval_dir = os.path.join(run_dir, "cross_evaluation")
    os.makedirs(cross_eval_dir, exist_ok=True)
    
    # Find robust strategies across scenarios
    scenario_ranks = {}
    
    for scenario, strategies in all_top_strategies.items():
        for i, strategy in enumerate(strategies):
            strategy_id = f"{strategy['strategy_type']}-{hash(str(strategy['parameters']))}"
            
            if strategy_id not in scenario_ranks:
                scenario_ranks[strategy_id] = {
                    "strategy_type": strategy["strategy_type"],
                    "parameters": strategy["parameters"],
                    "scenario_results": {},
                    "total_score": 0,
                    "threshold_passes": 0
                }
            
            # Add rank score (5 for 1st place, 4 for 2nd, etc.)
            rank_score = 5 - i
            scenario_ranks[strategy_id]["scenario_results"][scenario] = {
                "fitness": strategy["fitness"],
                "score": strategy["evaluation"]["score"],
                "meets_thresholds": strategy["evaluation"]["meets_thresholds"],
                "rank": i + 1,
                "rank_score": rank_score
            }
            scenario_ranks[strategy_id]["total_score"] += rank_score
            
            # Count threshold passes
            if strategy["evaluation"]["meets_thresholds"]:
                scenario_ranks[strategy_id]["threshold_passes"] += 1
    
    # Sort by threshold passes first, then total score
    robust_strategies = sorted(
        scenario_ranks.values(), 
        key=lambda x: (x["threshold_passes"], x["total_score"]), 
        reverse=True
    )
    
    # Save robust strategies
    with open(os.path.join(cross_eval_dir, "robust_strategies.json"), "w") as f:
        json.dump(robust_strategies[:10], f, indent=2)  # Top 10 robust strategies
    
    logger.info(f"Multi-scenario evolution complete. Results saved to {run_dir}")
    
    if robust_strategies:
        top_strategy = robust_strategies[0]
        logger.info(f"Top robust strategy: {top_strategy['strategy_type']} with score {top_strategy['total_score']}")
        logger.info(f"Passes thresholds in {top_strategy['threshold_passes']} scenarios")
    else:
        logger.info("No robust strategies found")
    
    # Generate BenBot summary report
    benbot_formatter.generate_summary_report()
    
    return robust_strategies


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Run evolution with thresholds")
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default="./strategy_registry",
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./threshold_evolution_results",
        help="Output directory"
    )
    
    parser.add_argument(
        "--population", 
        type=int, 
        default=100,  # Increased from 50 to 100
        help="Population size"
    )
    
    parser.add_argument(
        "--generations", 
        type=int, 
        default=50,  # Increased from 20 to 50
        help="Number of generations"
    )
    
    parser.add_argument(
        "--scenarios", 
        type=str, 
        nargs="+",
        default=None,
        help="Scenarios to run (bull_market, bear_market, sideways_market, volatile_market, flash_crash, sector_rotation)"
    )
    
    parser.add_argument(
        "--strategies", 
        type=str, 
        nargs="+",
        default=None,
        help="Strategy types to include (MeanReversion, Momentum, VolumeProfile, VolatilityBreakout)"
    )
    
    args = parser.parse_args()
    
    run_multi_scenario_evolution_with_thresholds(
        registry_path=args.registry,
        output_dir=args.output,
        population_size=args.population,
        generations=args.generations,
        scenarios=args.scenarios,
        strategy_types=args.strategies
    )
