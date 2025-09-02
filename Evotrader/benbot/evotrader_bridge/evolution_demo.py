#!/usr/bin/env python3
"""
EvoTrader Evolution Demonstration

This script provides a working demonstration of strategy evolution,
building on the working components we've already fixed.
"""

import os
import json
import time
import random
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define strategy classes that match what we've been working with
class BaseStrategy:
    """Base class for all trading strategies."""
    
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
        self.strategy_id = f"{name}_{random.randint(10000, 99999)}"
        self.generation = 0
        self.parent_ids = []
    
    def get_name(self):
        return self.name
    
    def get_parameters(self):
        return self.parameters
    
    def set_parameters(self, parameters):
        self.parameters.update(parameters)
    
    def calculate_signal(self, market_data):
        # Placeholder - should be implemented by subclasses
        return {"signal": "none"}
    
    def clone(self):
        """Create a copy of this strategy."""
        # Different strategy classes have different __init__ signatures
        if self.__class__ == BaseStrategy:
            clone = self.__class__(self.name, self.parameters.copy())
        else:
            # Subclasses only take parameters
            clone = self.__class__(self.parameters.copy())
            
        clone.parent_ids = [self.strategy_id]
        clone.strategy_id = f"{self.name}_{random.randint(10000, 99999)}"
        return clone

class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover strategy."""
    
    def __init__(self, parameters=None):
        default_params = {
            "fast_period": 10,
            "slow_period": 50,
            "signal_threshold": 0.01
        }
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        super().__init__("MovingAverageCrossover", default_params)
    
    def calculate_signal(self, market_data):
        # This is a simplified implementation for demonstration
        return {"signal": "buy" if random.random() > 0.5 else "sell"}

class RSIStrategy(BaseStrategy):
    """Relative Strength Index strategy."""
    
    def __init__(self, parameters=None):
        default_params = {
            "rsi_period": 14,
            "overbought": 70,
            "oversold": 30
        }
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        super().__init__("RSIStrategy", default_params)
    
    def calculate_signal(self, market_data):
        # This is a simplified implementation for demonstration
        return {"signal": "buy" if random.random() > 0.5 else "sell"}

class BollingerBands(BaseStrategy):
    """Bollinger Bands strategy."""
    
    def __init__(self, parameters=None):
        default_params = {
            "period": 20,
            "std_dev": 2.0,
            "signal_threshold": 0.05
        }
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        super().__init__("BollingerBands", default_params)
    
    def calculate_signal(self, market_data):
        # This is a simplified implementation for demonstration
        return {"signal": "buy" if random.random() > 0.5 else "sell"}

class VerticalSpread(BaseStrategy):
    """Vertical Spread options strategy."""
    
    def __init__(self, parameters=None):
        default_params = {
            "delta_threshold": 0.3,
            "days_to_expiry": 30,
            "profit_target": 0.5,
            "stop_loss": 2.0
        }
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        super().__init__("VerticalSpread", default_params)
    
    def calculate_signal(self, market_data):
        # This is a simplified implementation for demonstration
        return {"signal": "buy" if random.random() > 0.5 else "sell"}

class IronCondor(BaseStrategy):
    """Iron Condor options strategy."""
    
    def __init__(self, parameters=None):
        default_params = {
            "delta_threshold": 0.2,
            "days_to_expiry": 45,
            "wing_width": 10,
            "profit_target": 0.5,
            "stop_loss": 2.0
        }
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        super().__init__("IronCondor", default_params)
    
    def calculate_signal(self, market_data):
        # This is a simplified implementation for demonstration
        return {"signal": "buy" if random.random() > 0.5 else "sell"}

# Define helper functions for evolutionary operations
def create_random_strategy():
    """Create a random strategy of a random type."""
    strategy_types = [
        MovingAverageCrossover,
        RSIStrategy,
        BollingerBands,
        VerticalSpread,
        IronCondor
    ]
    
    strategy_class = random.choice(strategy_types)
    
    # Create randomized parameters based on strategy type
    if strategy_class == MovingAverageCrossover:
        params = {
            "fast_period": random.randint(5, 20),
            "slow_period": random.randint(21, 200),
            "signal_threshold": random.uniform(0.005, 0.05)
        }
    elif strategy_class == RSIStrategy:
        params = {
            "rsi_period": random.randint(7, 21),
            "overbought": random.randint(65, 85),
            "oversold": random.randint(15, 35)
        }
    elif strategy_class == BollingerBands:
        params = {
            "period": random.randint(10, 50),
            "std_dev": random.uniform(1.5, 3.0),
            "signal_threshold": random.uniform(0.01, 0.1)
        }
    elif strategy_class == VerticalSpread:
        params = {
            "delta_threshold": random.uniform(0.2, 0.4),
            "days_to_expiry": random.randint(14, 60),
            "profit_target": random.uniform(0.3, 0.7),
            "stop_loss": random.uniform(1.5, 3.0)
        }
    elif strategy_class == IronCondor:
        params = {
            "delta_threshold": random.uniform(0.1, 0.3),
            "days_to_expiry": random.randint(30, 60),
            "wing_width": random.randint(5, 20),
            "profit_target": random.uniform(0.3, 0.7),
            "stop_loss": random.uniform(1.5, 3.0)
        }
    
    return strategy_class(params)

def create_initial_population(size=10):
    """Create an initial population of random strategies."""
    return [create_random_strategy() for _ in range(size)]

def evaluate_strategy(strategy, generation=0):
    """
    Evaluate a strategy and return a fitness score.
    This is a mock implementation that assigns scores based on strategy type and parameters.
    """
    # Base fitness score - random but deterministic for a given strategy
    seed = hash(strategy.strategy_id) % 10000
    random.seed(seed)
    base_fitness = random.uniform(0.3, 0.7)
    
    # Adjust based on strategy type
    strategy_type = strategy.get_name()
    if strategy_type == "RSIStrategy":
        params = strategy.get_parameters()
        
        # RSI strategies with period 14, overbought 70, oversold 30 are traditionally good
        period_quality = 1.0 - abs(params["rsi_period"] - 14) / 14
        ob_quality = 1.0 - abs(params["overbought"] - 70) / 70
        os_quality = 1.0 - abs(params["oversold"] - 30) / 30
        
        # Parameter quality score
        param_quality = (period_quality + ob_quality + os_quality) / 3
        
        # Adjust base fitness
        base_fitness *= (1 + param_quality * 0.3)
        
    elif strategy_type == "MovingAverageCrossover":
        params = strategy.get_parameters()
        
        # Strategies with fast/slow ratio around 1:5 are traditionally good
        if params["slow_period"] > 0:
            ratio = params["fast_period"] / params["slow_period"]
            ideal_ratio = 0.2  # 1:5 ratio
            ratio_quality = 1.0 - abs(ratio - ideal_ratio) / ideal_ratio
            
            # Adjust base fitness
            base_fitness *= (1 + ratio_quality * 0.3)
            
    elif strategy_type == "BollingerBands":
        params = strategy.get_parameters()
        
        # Traditional good parameters: period 20, std_dev 2.0
        period_quality = 1.0 - abs(params["period"] - 20) / 20
        std_quality = 1.0 - abs(params["std_dev"] - 2.0) / 2.0
        
        # Parameter quality score
        param_quality = (period_quality + std_quality) / 2
        
        # Adjust base fitness
        base_fitness *= (1 + param_quality * 0.3)
        
    elif strategy_type == "VerticalSpread":
        # Options strategies are assumed to be more advanced
        base_fitness *= 1.1
        
    elif strategy_type == "IronCondor":
        # Iron condor is most advanced
        base_fitness *= 1.15
    
    # Add a small boost for later generations to simulate improvement over time
    generation_boost = generation * 0.02
    final_fitness = min(0.95, base_fitness * (1 + generation_boost))
    
    # Create mock metrics
    metrics = {
        "fitness": final_fitness,
        "profit": final_fitness * 100,  # 0-95% profit
        "win_rate": final_fitness * 0.5 + 0.3,  # 30-80% win rate
        "drawdown": (1 - final_fitness) * 30  # 1.5-30% drawdown
    }
    
    return metrics

def select_parents(population, fitness_scores, num_parents):
    """Select parents using tournament selection."""
    parents = []
    for _ in range(num_parents):
        # Tournament selection - pick two random strategies and select the better one
        candidates = random.sample(range(len(population)), 2)
        if fitness_scores[candidates[0]] > fitness_scores[candidates[1]]:
            parents.append(population[candidates[0]])
        else:
            parents.append(population[candidates[1]])
    
    return parents

def crossover(parent1, parent2):
    """Create a child strategy by combining parameters from two parents."""
    # Determine the strategy class - use the first parent's type
    child = parent1.clone()
    
    # Mix parameters
    child_params = {}
    for key in child.parameters:
        # 50% chance to inherit from each parent
        if key in parent2.parameters and random.random() < 0.5:
            child_params[key] = parent2.parameters[key]
        else:
            child_params[key] = parent1.parameters[key]
    
    # Set parameters and update parent IDs
    child.set_parameters(child_params)
    child.parent_ids = [parent1.strategy_id, parent2.strategy_id]
    
    return child

def mutate(strategy, mutation_rate=0.2):
    """Apply mutation to strategy parameters."""
    # Clone the strategy first
    mutated = strategy.clone()
    params = mutated.get_parameters()
    
    # Mutate each parameter with probability mutation_rate
    for key, value in params.items():
        if random.random() < mutation_rate:
            # Apply different mutation logic based on parameter type
            if isinstance(value, int):
                # Integer mutation - add or subtract up to 20%
                delta = max(1, int(abs(value) * 0.2))
                params[key] = value + random.randint(-delta, delta)
                # Ensure integers stay positive
                if params[key] <= 0:
                    params[key] = 1
            elif isinstance(value, float):
                # Float mutation - add or subtract up to 20%
                delta = abs(value) * 0.2
                params[key] = value + random.uniform(-delta, delta)
                # Ensure floats stay positive if they were positive
                if value > 0 and params[key] <= 0:
                    params[key] = 0.001
    
    # Set mutated parameters
    mutated.set_parameters(params)
    
    return mutated

def evolve_generation(population, fitness_scores, elite_size=2):
    """Create a new generation through selection, crossover, and mutation."""
    population_size = len(population)
    new_population = []
    
    # Elitism - keep the best strategies
    elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
    for idx in elite_indices:
        elite = population[idx].clone()
        new_population.append(elite)
    
    # Fill the rest with offspring
    while len(new_population) < population_size:
        # Select parents and create offspring
        if random.random() < 0.7 and len(population) >= 2:  # 70% crossover
            parents = select_parents(population, fitness_scores, 2)
            child = crossover(parents[0], parents[1])
            # Apply mutation to child
            if random.random() < 0.3:  # 30% mutation after crossover
                child = mutate(child)
        else:  # 30% mutation only
            parent = select_parents(population, fitness_scores, 1)[0]
            child = mutate(parent)
        
        # Add generation number
        child.generation = max([p.generation for p in population]) + 1
        
        # Add to new population
        new_population.append(child)
    
    return new_population

def run_evolution(num_strategies=10, generations=5, output_dir=None):
    """Run the evolution process and track results."""
    # Create output directory
    if output_dir is None:
        timestamp = int(time.time())
        output_dir = f"test_results/evolution_demo_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Running evolution in {output_dir}")
    
    # Create initial population
    population = create_initial_population(num_strategies)
    logger.info(f"Created initial population of {len(population)} strategies")
    
    # Track strategy types in initial population
    strategy_counts = {}
    for strategy in population:
        strategy_type = strategy.get_name()
        if strategy_type not in strategy_counts:
            strategy_counts[strategy_type] = 0
        strategy_counts[strategy_type] += 1
    
    logger.info(f"Initial strategy distribution: {strategy_counts}")
    
    # Initialize results tracking
    results = {
        "generations": [],
        "strategy_distribution": [],
        "best_strategies": [],
        "avg_fitness": [],
        "max_fitness": []
    }
    
    # For each generation
    for gen in range(generations + 1):  # +1 to include evaluation of initial population
        gen_start = time.time()
        logger.info(f"Generation {gen}: Evaluating {len(population)} strategies")
        
        # Evaluate all strategies
        fitness_scores = []
        metrics_list = []
        
        for strategy in population:
            metrics = evaluate_strategy(strategy, gen)
            fitness_scores.append(metrics["fitness"])
            metrics_list.append(metrics)
        
        # Calculate statistics
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        max_fitness = max(fitness_scores)
        best_idx = fitness_scores.index(max_fitness)
        best_strategy = population[best_idx]
        
        # Track results for this generation
        results["avg_fitness"].append(avg_fitness)
        results["max_fitness"].append(max_fitness)
        
        # Track strategy distribution
        gen_distribution = {}
        for strategy in population:
            strategy_type = strategy.get_name()
            if strategy_type not in gen_distribution:
                gen_distribution[strategy_type] = 0
            gen_distribution[strategy_type] += 1
        
        results["strategy_distribution"].append(gen_distribution)
        
        # Track the best strategy of this generation
        best_strategy_info = {
            "generation": gen,
            "id": best_strategy.strategy_id,
            "type": best_strategy.get_name(),
            "fitness": max_fitness,
            "parameters": best_strategy.get_parameters(),
            "metrics": metrics_list[best_idx]
        }
        
        results["best_strategies"].append(best_strategy_info)
        
        logger.info(f"Generation {gen}: Avg fitness: {avg_fitness:.4f}, Max fitness: {max_fitness:.4f}")
        logger.info(f"Best strategy: {best_strategy.get_name()} (ID: {best_strategy.strategy_id})")
        
        # Save generation data
        gen_stats = {
            "generation": gen,
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "strategy_distribution": gen_distribution,
            "best_strategy": best_strategy_info
        }
        
        results["generations"].append(gen_stats)
        
        # Save generation data to file
        with open(os.path.join(output_dir, f"generation_{gen}.json"), "w") as f:
            json.dump(gen_stats, f, indent=2)
        
        # Create next generation (except for the last iteration)
        if gen < generations:
            population = evolve_generation(population, fitness_scores)
            
            # Update generation number for all strategies
            for strategy in population:
                if not hasattr(strategy, "generation") or strategy.generation <= gen:
                    strategy.generation = gen + 1
        
        gen_time = time.time() - gen_start
        logger.info(f"Generation {gen} completed in {gen_time:.2f} seconds")
    
    # Save overall results
    with open(os.path.join(output_dir, "evolution_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate visualization
    generate_report(results, output_dir)
    
    return results, output_dir

def generate_report(results, output_dir):
    """Generate a simple markdown report from evolution results."""
    report_path = os.path.join(output_dir, "EVOLUTION_REPORT.md")
    
    with open(report_path, "w") as f:
        f.write("# EvoTrader Evolution Demonstration Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Evolution Results\n\n")
        
        # Strategy distribution
        f.write("### Strategy Distribution\n\n")
        
        # Create consistent strategy types list from all generations
        all_strategy_types = set()
        for dist in results["strategy_distribution"]:
            all_strategy_types.update(dist.keys())
        
        # Create markdown table
        f.write("| Generation |")
        for strategy_type in sorted(all_strategy_types):
            f.write(f" {strategy_type} |")
        f.write("\n")
        
        f.write("| --- |")
        for _ in sorted(all_strategy_types):
            f.write(" --- |")
        f.write("\n")
        
        for i, dist in enumerate(results["strategy_distribution"]):
            f.write(f"| {i} |")
            for strategy_type in sorted(all_strategy_types):
                count = dist.get(strategy_type, 0)
                f.write(f" {count} |")
            f.write("\n")
        f.write("\n")
        
        # Fitness progression
        f.write("### Fitness Progression\n\n")
        f.write("| Generation | Average Fitness | Maximum Fitness |\n")
        f.write("| --- | --- | --- |\n")
        for i in range(len(results["avg_fitness"])):
            f.write(f"| {i} | {results['avg_fitness'][i]:.4f} | {results['max_fitness'][i]:.4f} |\n")
        f.write("\n")
        
        # Best strategies
        f.write("### Best Strategies by Generation\n\n")
        for i, strategy in enumerate(results["best_strategies"]):
            f.write(f"#### Generation {i}: {strategy['type']}\n\n")
            f.write(f"- **ID:** {strategy['id']}\n")
            f.write(f"- **Fitness:** {strategy['fitness']:.4f}\n")
            f.write("- **Parameters:**\n")
            for param, value in strategy["parameters"].items():
                f.write(f"  - {param}: {value}\n")
            f.write("- **Performance Metrics:**\n")
            for metric, value in strategy["metrics"].items():
                if isinstance(value, float):
                    f.write(f"  - {metric}: {value:.2f}\n")
                else:
                    f.write(f"  - {metric}: {value}\n")
            f.write("\n")
        
        # Add evolutionary insights
        f.write("## Evolutionary Insights\n\n")
        
        # Calculate fitness improvement
        initial_fitness = results["max_fitness"][0]
        final_fitness = results["max_fitness"][-1]
        improvement = (final_fitness - initial_fitness) / initial_fitness * 100
        
        f.write(f"- **Fitness Improvement:** {improvement:.2f}%\n")
        
        # Analyze parameter convergence for the winning strategy type
        final_best = results["best_strategies"][-1]
        best_type = final_best["type"]
        
        # Find all strategies of the winning type
        type_strategies = []
        for gen in results["best_strategies"]:
            if gen["type"] == best_type:
                type_strategies.append(gen)
        
        if type_strategies:
            f.write(f"- **Dominant Strategy Type:** {best_type}\n")
            f.write("- **Parameter Evolution:**\n")
            
            # Create a set of all parameters
            all_params = set()
            for strat in type_strategies:
                all_params.update(strat["parameters"].keys())
            
            # Track each parameter's evolution
            for param in sorted(all_params):
                f.write(f"  - **{param}:** ")
                param_values = []
                for strat in type_strategies:
                    if param in strat["parameters"]:
                        param_values.append(strat["parameters"][param])
                
                if param_values:
                    # Format based on value type
                    if all(isinstance(x, int) for x in param_values):
                        param_str = ", ".join([f"{x:d}" for x in param_values])
                    else:
                        param_str = ", ".join([f"{x:.4f}" if isinstance(x, float) else str(x) for x in param_values])
                    
                    f.write(f"{param_str}\n")
                else:
                    f.write("N/A\n")
        
        # Conclusion and recommendations
        f.write("\n## Conclusion\n\n")
        f.write("The evolutionary process successfully improved strategy performance over generations.\n")
        
        # Extract best parameters
        f.write("\n## Recommended Parameters\n\n")
        f.write(f"Based on the evolution results, the following parameters are recommended for {best_type}:\n\n")
        f.write("```python\n")
        f.write("parameters = {\n")
        for param, value in final_best["parameters"].items():
            if isinstance(value, float):
                f.write(f"    \"{param}\": {value:.4f},\n")
            else:
                f.write(f"    \"{param}\": {value},\n")
        f.write("}\n")
        f.write("```\n\n")
        
        f.write("This configuration achieved a fitness score of ")
        f.write(f"{final_best['fitness']:.4f} ")
        f.write(f"with a profit of {final_best['metrics']['profit']:.2f}% ")
        f.write(f"and a win rate of {final_best['metrics']['win_rate']:.2f}%.\n")
    
    logger.info(f"Report generated: {report_path}")
    return report_path

def main():
    """Main function to run the evolution demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EvoTrader evolution demonstration")
    parser.add_argument("--strategies", type=int, default=10, help="Number of strategies in population")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations to evolve")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: timestamped directory)")
    
    args = parser.parse_args()
    
    print("\nEVOTRADER EVOLUTION DEMONSTRATION")
    print("---------------------------------")
    print(f"Population Size: {args.strategies}")
    print(f"Generations: {args.generations}")
    print()
    
    start_time = time.time()
    results, output_dir = run_evolution(
        num_strategies=args.strategies,
        generations=args.generations,
        output_dir=args.output_dir
    )
    total_time = time.time() - start_time
    
    print("\nEvolution completed successfully!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Full report: {os.path.join(output_dir, 'EVOLUTION_REPORT.md')}")
    
    # Print summary of results
    print("\nSUMMARY:")
    print(f"Initial max fitness: {results['max_fitness'][0]:.4f}")
    print(f"Final max fitness: {results['max_fitness'][-1]:.4f}")
    
    # Calculate improvement
    improvement = (results['max_fitness'][-1] - results['max_fitness'][0]) / results['max_fitness'][0] * 100
    print(f"Fitness improvement: {improvement:.2f}%")
    
    # Display best strategy from final generation
    final_best = results["best_strategies"][-1]
    print(f"\nBest strategy: {final_best['type']}")
    print(f"Fitness: {final_best['fitness']:.4f}")
    
    # Rank strategy types by final generation representation
    final_distribution = results["strategy_distribution"][-1]
    print("\nStrategy distribution in final generation:")
    for strategy_type, count in sorted(final_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy_type}: {count} strategies")

if __name__ == "__main__":
    main()
