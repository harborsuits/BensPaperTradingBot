"""
Test script for running the evolutionary trading system with a small population.

This script creates a set of test strategies, runs them through multiple generations
of evolution, and analyzes the performance improvements.
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import json
import time
import logging
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from test_strategies import create_test_strategies
from strategy_adapter import BensBotStrategyAdapter
from main import EvoTraderBridge
from testing_framework import SimulationEnvironment
from ab_testing import ABTest


def setup_test_environment(
    strategy_count: int = 15,
    test_id: str = None,
    config: Dict[str, Any] = None
) -> Tuple[EvoTraderBridge, List[BensBotStrategyAdapter]]:
    """
    Set up the test environment with a population of diverse strategies.
    
    Args:
        strategy_count: Number of strategies to create
        test_id: Optional identifier for this test
        config: Optional configuration overrides
        
    Returns:
        Tuple of (bridge, adapted_strategies)
    """
    if test_id is None:
        test_id = f"test_{int(time.time())}"
        
    # Create output directory
    output_dir = f"test_results/{test_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test configuration
    test_config = {
        "output_dir": output_dir,
        "evolution": {
            "output_dir": f"{output_dir}/evolution",
            "selection_percentage": 0.3,
            "mutation_rate": 0.2,
            "crossover_rate": 0.3,
            "population_size": strategy_count,
            "max_generations": 10
        },
        "simulation": {
            "output_dir": f"{output_dir}/simulation",
            "data_source": "synthetic",  # Use synthetic data for testing
            "symbols": ["BTC/USD", "ETH/USD"],
            "timeframe": "1h",
            "initial_balance": 10000,
            "fee_rate": 0.001
        },
        "ab_testing": {
            "output_dir": f"{output_dir}/ab_testing",
            "test_count": 5,
            "bootstrap_samples": 50,
            "significance_level": 0.05
        }
    }
    
    # Override with custom config if provided
    if config:
        for section, values in config.items():
            if section in test_config and isinstance(values, dict):
                test_config[section].update(values)
            else:
                test_config[section] = values
    
    # Save configuration
    with open(f"{output_dir}/test_config.json", 'w') as f:
        json.dump(test_config, f, indent=2)
    
    # Initialize bridge with test configuration
    bridge = EvoTraderBridge(config=test_config)
    
    # Create test strategies
    benbot_strategies = create_test_strategies(strategy_count)
    
    # Adapt strategies to EvoTrader framework
    adapted_strategies = []
    for benbot_strategy in benbot_strategies:
        adapter = bridge.adapt_benbot_strategy(benbot_strategy)
        bridge.register_strategy(adapter)
        adapted_strategies.append(adapter)
    
    return bridge, adapted_strategies


def run_evolution(
    bridge: EvoTraderBridge,
    generations: int = 5,
    evaluate_every_generation: bool = True
) -> Dict[str, Any]:
    """
    Run the evolutionary process for multiple generations.
    
    Args:
        bridge: EvoTraderBridge instance
        generations: Number of generations to evolve
        evaluate_every_generation: Whether to evaluate all strategies after each generation
        
    Returns:
        Evolution statistics
    """
    print(f"Starting evolution process for {generations} generations...")
    
    # Initialize statistics tracking
    evolution_stats = {
        "generations": [],
        "avg_fitness": [],
        "max_fitness": [],
        "strategy_counts": {}
    }
    
    # Run evolution for specified number of generations
    for i in range(generations):
        gen_start_time = time.time()
        print(f"\nEvolving generation {i+1}/{generations}...")
        
        # Evolve next generation
        bridge.evolution_manager.evolve_next_generation()
        
        # Get current generation number
        current_gen = bridge.evolution_manager.current_generation
        
        # Evaluate strategies if requested
        if evaluate_every_generation:
            print(f"Evaluating generation {current_gen} strategies...")
            strategies = bridge.evolution_manager.strategies
            
            for strategy_id, strategy in strategies.items():
                try:
                    # Evaluate strategy
                    test_id = f"gen_{current_gen}_eval"
                    bridge.evaluate_strategy(strategy, test_id)
                except Exception as e:
                    print(f"Error evaluating strategy {strategy_id}: {str(e)}")
        
        # Get generation stats
        gen_stats = bridge.evolution_manager.get_generation_stats(current_gen)
        
        # Track statistics
        evolution_stats["generations"].append(current_gen)
        evolution_stats["avg_fitness"].append(gen_stats.get("avg_fitness", 0))
        evolution_stats["max_fitness"].append(gen_stats.get("max_fitness", 0))
        
        # Track strategy type distribution
        strategy_distribution = gen_stats.get("strategy_distribution", {})
        for strategy_type, count in strategy_distribution.items():
            if strategy_type not in evolution_stats["strategy_counts"]:
                evolution_stats["strategy_counts"][strategy_type] = []
            
            # Extend list if needed
            while len(evolution_stats["strategy_counts"][strategy_type]) < current_gen:
                evolution_stats["strategy_counts"][strategy_type].append(0)
            
            evolution_stats["strategy_counts"][strategy_type].append(count)
        
        # Print generation summary
        gen_duration = time.time() - gen_start_time
        print(f"Generation {current_gen} complete in {gen_duration:.2f} seconds")
        print(f"Average fitness: {gen_stats.get('avg_fitness', 0):.4f}")
        print(f"Max fitness: {gen_stats.get('max_fitness', 0):.4f}")
        
    return evolution_stats


def main():
    """Run the test evolution process."""
    parser = argparse.ArgumentParser(description='Test EvoTrader Evolution')
    parser.add_argument('--strategies', type=int, default=15, help='Number of strategies to create')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations to evolve')
    parser.add_argument('--test-id', type=str, help='Identifier for this test run')
    
    args = parser.parse_args()
    
    print(f"Setting up test environment with {args.strategies} strategies...")
    bridge, strategies = setup_test_environment(
        strategy_count=args.strategies,
        test_id=args.test_id
    )
    
    print(f"Initial population: {len(strategies)} strategies")
    for i, strategy in enumerate(strategies):
        strategy_type = "Unknown"
        if hasattr(strategy, "benbot_strategy") and strategy.benbot_strategy:
            strategy_type = strategy.benbot_strategy.__class__.__name__
        print(f"  Strategy {i+1}: {strategy.strategy_id} (Type: {strategy_type})")
    
    # Evaluate initial population
    print("\nEvaluating initial population...")
    for strategy in strategies:
        bridge.evaluate_strategy(strategy, "initial_evaluation")
    
    # Run evolution
    evolution_stats = run_evolution(
        bridge,
        generations=args.generations,
        evaluate_every_generation=True
    )
    
    # Print summary
    print("\nEvolution complete!")
    print(f"Total generations: {bridge.evolution_manager.current_generation}")
    
    # Get best strategies
    best_strategies = bridge.get_best_strategies(count=3)
    print("\nTop 3 evolved strategies:")
    for i, strategy in enumerate(best_strategies):
        strategy_type = "Unknown"
        if hasattr(strategy, "benbot_strategy") and strategy.benbot_strategy:
            strategy_type = strategy.benbot_strategy.__class__.__name__
        print(f"  {i+1}. {strategy.strategy_id} (Type: {strategy_type})")
    
    # Compare best evolved strategy with its original ancestor
    if best_strategies:
        best = best_strategies[0]
        parent_ids = best.metadata.get("parent_ids", [])
        
        if parent_ids and parent_ids[0] in bridge.evolution_manager.strategies:
            original = bridge.evolution_manager.strategies[parent_ids[0]]
            
            print("\nComparing best evolved strategy with original ancestor...")
            comparison = bridge.compare_strategies(original, best)
            
            overall = comparison.get("overall", {})
            if overall.get("is_improvement", False):
                print("  ✅ Evolved strategy shows improvement")
                if overall.get("is_significant_improvement", False):
                    print("  ✅ Improvement is statistically significant")
            else:
                print("  ❌ No significant improvement detected")
    
    # Export results
    bridge.evolution_manager.export_results()
    
    print("\nTest completed successfully!")
    
    return evolution_stats


if __name__ == "__main__":
    main()
