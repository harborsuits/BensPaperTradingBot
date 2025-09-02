#!/usr/bin/env python3
"""
EvoTrader - Evolutionary Trading Bot System

This script runs a simulation of the EvoTrader system, demonstrating
the evolutionary process of trading bots competing and adapting their
strategies over multiple generations.
"""

import os
import sys
import time
import argparse
import logging
import uuid
import random
from typing import List, Dict, Any

from evotrader.core.challenge_bot import ChallengeBot
from evotrader.core.challenge_manager import ChallengeManager
from evotrader.core.strategy import Strategy
from evotrader.strategies.trend_following import MovingAverageCrossover, RSIStrategy
from evotrader.strategies.mean_reversion import BollingerBandsStrategy, PriceDeviationStrategy
from evotrader.strategies.breakout import ChannelBreakoutStrategy, SupportResistanceStrategy
from evotrader.strategies.volatility import ATRPositionSizing, VolatilityBreakoutStrategy
from evotrader.strategies.pattern import HeadAndShouldersStrategy
from evotrader.strategies.pattern_double import DoubleTopBottomStrategy
from evotrader.strategies.timeframe import SwingTradingStrategy
from evotrader.strategies.timeframe_day import DayTradingStrategy
from evotrader.data.market_data_provider import RandomWalkDataProvider
from evotrader.utils.logging import setup_logging
from evotrader.utils.visualization import visualize_results, plot_equity_curves
from evotrader.utils.visualization_fix import generate_performance_report


def create_initial_population(
    population_size: int = 100,
    strategy_classes: List[type] = None,
    initial_balance: float = 1.0
) -> List[ChallengeBot]:
    """
    Create the initial bot population with randomized strategies.
    
    Args:
        population_size: Number of bots to create
        strategy_classes: List of strategy classes to use
        initial_balance: Starting balance for each bot
        
    Returns:
        List of configured bots
    """
    if strategy_classes is None:
        strategy_classes = [
            # Trend following strategies
            MovingAverageCrossover,
            RSIStrategy,
            # Mean reversion strategies
            BollingerBandsStrategy,
            PriceDeviationStrategy,
            # Breakout strategies
            ChannelBreakoutStrategy,
            SupportResistanceStrategy,
            # Volatility strategies
            ATRPositionSizing,
            VolatilityBreakoutStrategy,
            # Pattern detection strategies
            HeadAndShouldersStrategy,
            DoubleTopBottomStrategy,
            # Timeframe strategies
            SwingTradingStrategy,
            DayTradingStrategy
        ]
        
    bots = []
    
    # Distribution of strategies - equally distributed initially
    strategies_per_type = population_size // len(strategy_classes)
    
    for i, strategy_class in enumerate(strategy_classes):
        # Create bots with this strategy class
        for j in range(strategies_per_type):
            # Create bot with unique ID
            bot_id = f"bot_{i}_{j}_{str(uuid.uuid4())[:8]}"
            
            # Create strategy with random parameter variations
            strategy = create_randomized_strategy(strategy_class)
            
            # Create and add bot
            bot = ChallengeBot(bot_id=bot_id, strategy=strategy, initial_balance=initial_balance)
            bots.append(bot)
            
    # Add remaining bots with random strategies if needed
    remaining = population_size - len(bots)
    for i in range(remaining):
        bot_id = f"bot_r_{i}_{str(uuid.uuid4())[:8]}"
        strategy_class = random.choice(strategy_classes)
        strategy = create_randomized_strategy(strategy_class)
        bot = ChallengeBot(bot_id=bot_id, strategy=strategy, initial_balance=initial_balance)
        bots.append(bot)
        
    return bots


def create_randomized_strategy(strategy_class: type) -> Strategy:
    """
    Create a strategy instance with randomized parameters.
    
    Args:
        strategy_class: Strategy class to instantiate
        
    Returns:
        Instantiated strategy with randomized parameters
    """
    # Get parameter definitions
    param_defs = strategy_class.get_parameters()
    
    # Create randomized parameters within constraints
    params = {}
    for param in param_defs:
        if isinstance(param.default_value, (int, float)):
            # Numeric parameter
            if param.min_value is not None and param.max_value is not None:
                # Use constraints
                if isinstance(param.default_value, int):
                    # Integer parameter
                    params[param.name] = random.randint(int(param.min_value), int(param.max_value))
                else:
                    # Float parameter
                    params[param.name] = random.uniform(param.min_value, param.max_value)
            else:
                # No constraints, use default with some randomness
                variation = param.default_value * 0.3  # 30% variation
                if isinstance(param.default_value, int):
                    params[param.name] = max(1, int(param.default_value + random.uniform(-variation, variation)))
                else:
                    params[param.name] = param.default_value + random.uniform(-variation, variation)
        elif isinstance(param.default_value, bool):
            # Boolean parameter
            params[param.name] = random.choice([True, False])
        else:
            # Other types (strings, etc.) - use default for now
            params[param.name] = param.default_value
            
    # Create strategy with ID and parameters
    strategy_id = f"{strategy_class.__name__}_{str(uuid.uuid4())[:8]}"
    return strategy_class(strategy_id=strategy_id, parameters=params)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EvoTrader - Evolutionary Trading Bot System")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--generations", 
        type=int, 
        default=5,
        help="Number of generations to simulate"
    )
    parser.add_argument(
        "--population", 
        type=int, 
        default=100,
        help="Population size"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of results"
    )
    parser.add_argument(
        "--strategy-groups",
        type=str,
        default="all",
        help="Strategy groups to use (comma-separated): trend,mean_reversion,breakout,volatility,pattern,timeframe,all"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "evotrader.log")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.log_level, log_file)
    
    # Log start
    logger.info(f"Starting EvoTrader simulation with {args.population} bots, {args.generations} generations")
    
    # Initialize challenge manager
    manager = ChallengeManager(args.config)
    
    # Override config values from command line
    manager.config_manager.set("simulation.bots", args.population)
    manager.config_manager.set("simulation.seed", args.seed if args.seed else random.randint(1, 10000))
    manager.config_manager.set("output.directory", args.output_dir)
    
    # Create market data provider
    data_provider = RandomWalkDataProvider(
        seed=manager.config.get("simulation", {}).get("seed", 42)
    )
    manager.register_market_data_provider(data_provider)
    
    # Filter strategy classes based on command line argument
    selected_strategies = []
    strategy_categories = {
        "trend": [MovingAverageCrossover, RSIStrategy],
        "mean_reversion": [BollingerBandsStrategy, PriceDeviationStrategy],
        "breakout": [ChannelBreakoutStrategy, SupportResistanceStrategy],
        "volatility": [ATRPositionSizing, VolatilityBreakoutStrategy],
        "pattern": [HeadAndShouldersStrategy, DoubleTopBottomStrategy],
        "timeframe": [SwingTradingStrategy, DayTradingStrategy]
    }
    
    if args.strategy_groups.lower() == "all":
        # Use all strategies
        selected_strategies = None
    else:
        # Use only selected groups
        groups = [g.strip() for g in args.strategy_groups.split(",")]
        for group in groups:
            if group in strategy_categories:
                selected_strategies.extend(strategy_categories[group])
        
        if not selected_strategies:
            logger.warning(f"No valid strategy groups specified. Using all strategies.")
            selected_strategies = None
    
    # Create initial population
    initial_bots = create_initial_population(
        population_size=args.population,
        strategy_classes=selected_strategies,
        initial_balance=manager.config.get("simulation", {}).get("initial_balance", 1.0)
    )
    
    # Register bots
    for bot in initial_bots:
        manager.register_bot(bot)
    
    # Run the simulation
    start_time = time.time()
    results = manager.run(max_generations=args.generations)
    duration = time.time() - start_time
    
    # Print summary
    logger.info(f"Simulation completed in {duration:.2f} seconds")
    logger.info(f"Results saved to {args.output_dir}")
    
    final_gen = max(results.keys())
    final_stats = results[final_gen].to_dict()
    
    print("\n=== EvoTrader Simulation Results ===")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Duration: {duration:.2f} seconds")
    print("\nFinal Generation Stats:")
    print(f"  Average equity: ${final_stats['avg_equity']:.2f}")
    print(f"  Max equity: ${final_stats['max_equity']:.2f}")
    print(f"  Win rate: {final_stats['win_rate']*100:.1f}%")
    print(f"  Strategy distribution: {final_stats['strategy_distribution']}")
    print(f"\nResults saved to {args.output_dir}")
    
    # Export detailed results
    results_path = os.path.join(args.output_dir, "detailed_results")
    manager.export_results(results_path)
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualization of results...")
        
        # Create visualization directory
        vis_path = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_path, exist_ok=True)
        
        # Generate equity curve plots
        equity_plot_path = os.path.join(vis_path, "equity_curves.png")
        plot_equity_curves(results, equity_plot_path)
        
        # Generate strategy distribution chart
        strategy_dist_path = os.path.join(vis_path, "strategy_distribution.png")
        visualize_results(results, strategy_dist_path)
        
        # Generate detailed performance report
        report_path = os.path.join(vis_path, "performance_report.html")
        generate_performance_report(results, results_path, report_path)
        
        logger.info(f"Visualizations saved to {vis_path}")
        print(f"Visualizations saved to {vis_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
