#!/usr/bin/env python3
"""
Advanced Strategy Integration - Integrates advanced strategies with the evolutionary system
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Type
import argparse
import datetime
import logging

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import our modules
from advanced_strategies import (
    MeanReversionStrategy, 
    MomentumStrategy, 
    VolumeProfileStrategy, 
    VolatilityBreakoutStrategy
)
from strategy_registry import StrategyRegistry
from synthetic_market_generator import SyntheticMarketGenerator
from enhanced_evolution import run_evolution_on_scenario


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/advanced_integration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger('advanced_integration')


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


def evolve_advanced_strategies(
    market_data: pd.DataFrame,
    registry: StrategyRegistry,
    scenario_name: str,
    population_size: int = 20,
    generations: int = 10,
    strategy_types: Optional[List[str]] = None,
    elite_percentage: float = 0.2,
    crossover_rate: float = 0.3,
    mutation_rate: float = 0.3,
    tournament_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Evolve a population of advanced strategies.
    
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
        
    Returns:
        List of top strategies
    """
    logger.info(f"Starting advanced strategy evolution for {scenario_name}")
    logger.info(f"Market data shape: {market_data.shape}")
    
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
        from advanced_strategies import backtest_strategy
        results = backtest_strategy(strategy, market_data)
        
        # Add to population
        population.append({
            "strategy_type": strategy_type,
            "parameters": params,
            "results": results,
            "fitness": calculate_fitness(results)
        })
    
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
                child_params = crossover_params(
                    parent_a["strategy_type"], 
                    parent_a["parameters"],
                    parent_b["parameters"]
                )
                
                # Apply mutation
                child_params = mutate_strategy_params(
                    parent_a["strategy_type"], 
                    child_params, 
                    mutation_rate
                )
                
                child_strategy = create_advanced_strategy(parent_a["strategy_type"], child_params)
            else:
                # Mutation only
                parent = tournament_selection(population, tournament_size)
                
                # Create child through mutation
                child_params = mutate_strategy_params(
                    parent["strategy_type"], 
                    parent["parameters"], 
                    mutation_rate
                )
                
                child_strategy = create_advanced_strategy(parent["strategy_type"], child_params)
            
            # Backtest child
            from advanced_strategies import backtest_strategy
            child_results = backtest_strategy(child_strategy, market_data)
            
            # Add to new population
            new_population.append({
                "strategy_type": child_strategy.strategy_name,
                "parameters": child_params,
                "results": child_results,
                "fitness": calculate_fitness(child_results)
            })
        
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
        
        logger.info(f"Generation {generation} top fitness: {[round(ind['fitness'], 2) for ind in population[:5]]}")
    
    logger.info("Evolution complete")
    
    return population


def calculate_fitness(results: Dict[str, Any]) -> float:
    """
    Calculate fitness score for a strategy.
    
    Args:
        results: Backtest results
        
    Returns:
        Fitness score
    """
    if results["trade_count"] < 5:
        return -100  # Penalize strategies with too few trades
    
    # Reward return, penalize drawdown
    fitness = results["total_return_pct"]
    
    # Add win rate bonus
    fitness += results["win_rate"] * 0.2
    
    # Penalize drawdown (heavily)
    fitness -= results["max_drawdown"] * 1.5
    
    # Reward profit factor
    if results["profit_factor"] > 1:
        fitness += np.log(results["profit_factor"]) * 10
    else:
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


def run_multi_scenario_advanced_evolution(
    registry_path: str = "./strategy_registry",
    output_dir: str = "./advanced_evolution_results",
    population_size: int = 20,
    generations: int = 10,
    scenarios: Optional[List[str]] = None,
    strategy_types: Optional[List[str]] = None
):
    """
    Run evolution of advanced strategies on multiple market scenarios.
    
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
        
        # Evolve strategies on this scenario
        top_strategies = evolve_advanced_strategies(
            market_data=market_data,
            registry=registry,
            scenario_name=scenario,
            population_size=population_size,
            generations=generations,
            strategy_types=strategy_types
        )
        
        # Save top strategies
        all_top_strategies[scenario] = top_strategies[:5]  # Top 5 strategies
        
        # Save scenario results
        scenario_dir = os.path.join(run_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Save market data
        market_data.to_csv(os.path.join(scenario_dir, "market_data.csv"))
        
        # Save top strategies
        with open(os.path.join(scenario_dir, "top_strategies.json"), "w") as f:
            top_data = [
                {
                    "strategy_type": s["strategy_type"],
                    "parameters": s["parameters"],
                    "fitness": s["fitness"],
                    "results": {
                        k: v for k, v in s["results"].items() 
                        if k not in ["equity_curve", "trades"]  # Exclude large arrays
                    }
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
                    "total_score": 0
                }
            
            # Add rank score (5 for 1st place, 4 for 2nd, etc.)
            rank_score = 5 - i
            scenario_ranks[strategy_id]["scenario_results"][scenario] = {
                "fitness": strategy["fitness"],
                "rank": i + 1,
                "rank_score": rank_score
            }
            scenario_ranks[strategy_id]["total_score"] += rank_score
    
    # Sort by total score
    robust_strategies = sorted(
        scenario_ranks.values(), 
        key=lambda x: x["total_score"], 
        reverse=True
    )
    
    # Save robust strategies
    with open(os.path.join(cross_eval_dir, "robust_strategies.json"), "w") as f:
        json.dump(robust_strategies[:10], f, indent=2)  # Top 10 robust strategies
    
    logger.info(f"Multi-scenario evolution complete. Results saved to {run_dir}")
    logger.info(f"Top robust strategy: {robust_strategies[0]['strategy_type']} with score {robust_strategies[0]['total_score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate advanced strategies with evolution")
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default="./strategy_registry",
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./advanced_evolution_results",
        help="Output directory"
    )
    
    parser.add_argument(
        "--population", 
        type=int, 
        default=20,
        help="Population size"
    )
    
    parser.add_argument(
        "--generations", 
        type=int, 
        default=10,
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
    
    run_multi_scenario_advanced_evolution(
        registry_path=args.registry,
        output_dir=args.output,
        population_size=args.population,
        generations=args.generations,
        scenarios=args.scenarios,
        strategy_types=args.strategies
    )
