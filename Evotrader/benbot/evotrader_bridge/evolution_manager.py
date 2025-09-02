"""
Evolution Manager for BensBot-EvoTrader Integration

This module handles the evolutionary process for trading strategies, including
selection, mutation, crossover, and promotion based on performance metrics.
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import json
import time
import random
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Type, Union

# Import EvoTrader components
from evotrader.utils.evolution import (
    apply_mutation, 
    calculate_population_diversity,
    adaptive_crossover_rate
)

# Import our adapter
from benbot.evotrader_bridge.strategy_adapter import BensBotStrategyAdapter

logger = logging.getLogger(__name__)

class EvolutionStats:
    """Tracks statistics for a single generation of strategies."""
    
    def __init__(self, generation: int):
        self.generation = generation
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.strategy_count = 0
        self.surviving_strategies = 0
        self.extinct_strategies = 0
        self.avg_performance = 0.0
        self.max_performance = 0.0
        self.min_performance = float('inf')
        self.best_strategy_id: Optional[str] = None
        self.worst_strategy_id: Optional[str] = None
        self.avg_trades_per_strategy = 0.0
        self.total_trades = 0
        self.win_rate = 0.0
        self.avg_fitness = 0.0
        self.strategy_distribution: Dict[str, int] = {}  # Strategy type -> count
        
    def finalize(self, strategy_results: Dict[str, Dict[str, Any]]):
        """Calculate final statistics based on strategy results."""
        self.end_time = time.time()
        self.strategy_count = len(strategy_results)
        
        if self.strategy_count == 0:
            return
            
        # Calculate aggregate statistics
        total_performance = 0.0
        total_fitness = 0.0
        total_trades = 0
        total_wins = 0
        total_losses = 0
        max_performance = 0.0
        min_performance = float('inf')
        best_strategy_id = None
        worst_strategy_id = None
        
        for strategy_id, result in strategy_results.items():
            # Financial stats
            performance = result.get("performance", 0.0)
            total_performance += performance
            
            # Track best and worst strategies
            if performance > max_performance:
                max_performance = performance
                best_strategy_id = strategy_id
            if performance < min_performance and performance > 0:
                min_performance = performance
                worst_strategy_id = strategy_id
                
            # Trading stats
            total_trades += result.get("total_trades", 0)
            total_wins += result.get("win_count", 0)
            total_losses += result.get("loss_count", 0)
            
            # Fitness
            total_fitness += result.get("fitness_score", 0.0)
            
            # Track strategy distribution
            strategy_type = result.get("strategy_type", "unknown")
            if strategy_type not in self.strategy_distribution:
                self.strategy_distribution[strategy_type] = 0
            self.strategy_distribution[strategy_type] += 1
        
        # Set final statistics
        self.avg_performance = total_performance / self.strategy_count
        self.max_performance = max_performance
        self.min_performance = min_performance
        self.best_strategy_id = best_strategy_id
        self.worst_strategy_id = worst_strategy_id
        self.total_trades = total_trades
        self.avg_trades_per_strategy = total_trades / self.strategy_count if self.strategy_count > 0 else 0
        self.avg_fitness = total_fitness / self.strategy_count if self.strategy_count > 0 else 0
        
        # Calculate win rate
        total_closed_trades = total_wins + total_losses
        self.win_rate = total_wins / total_closed_trades if total_closed_trades > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a dictionary for serialization."""
        return {
            "generation": self.generation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.end_time - self.start_time if self.end_time else None,
            "strategy_count": self.strategy_count,
            "surviving_strategies": self.surviving_strategies,
            "extinct_strategies": self.extinct_strategies,
            "avg_performance": self.avg_performance,
            "max_performance": self.max_performance,
            "min_performance": self.min_performance,
            "best_strategy_id": self.best_strategy_id,
            "worst_strategy_id": self.worst_strategy_id,
            "avg_trades_per_strategy": self.avg_trades_per_strategy,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_fitness": self.avg_fitness,
            "strategy_distribution": self.strategy_distribution
        }


class EvolutionManager:
    """Manages evolutionary process for BensBot strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the evolution manager.
        
        Args:
            config: Configuration dictionary with evolution parameters
        """
        self.config = config or {
            "output_dir": "strategy_evolution",
            "selection_percentage": 0.3,  # Top 30% of strategies survive
            "mutation_rate": 0.2,
            "crossover_rate": 0.3,
            "population_size": 100,
            "max_generations": 20,
            "fitness_metrics": {
                "profit_weight": 0.5,
                "win_rate_weight": 0.3,
                "risk_weight": 0.2,
            }
        }
        
        # Create output directory
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize strategy tracking
        self.strategies: Dict[str, BensBotStrategyAdapter] = {}
        self.strategy_results: Dict[str, Dict[str, Any]] = {}
        self.generation_stats: Dict[int, EvolutionStats] = {}
        self.current_generation = 0
        self.start_time = time.time()
        
        # Set up logger
        self.logger = logging.getLogger(f"{__name__}.evolution_manager")
        self.setup_logger()
    
    def setup_logger(self):
        """Configure logging for the evolution manager."""
        log_file = os.path.join(self.output_dir, "evolution.log")
        
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Evolution manager initialized with output directory: {self.output_dir}")
    
    def register_strategy(self, strategy: BensBotStrategyAdapter) -> str:
        """
        Register a strategy to be included in the evolution process.
        
        Args:
            strategy: Strategy instance to register
            
        Returns:
            strategy_id: ID of the registered strategy
        """
        strategy_id = strategy.strategy_id
        self.strategies[strategy_id] = strategy
        
        self.logger.info(f"Registered strategy {strategy_id} for evolution")
        
        return strategy_id
    
    def record_strategy_performance(self, strategy_id: str, performance_data: Dict[str, Any]):
        """
        Record performance metrics for a strategy after testing/execution.
        
        Args:
            strategy_id: ID of the strategy
            performance_data: Dictionary of performance metrics
        """
        if strategy_id not in self.strategies:
            self.logger.warning(f"Cannot record performance for unknown strategy {strategy_id}")
            return
            
        # Store performance data
        self.strategy_results[strategy_id] = performance_data
        
        # Add performance to strategy metadata
        strategy = self.strategies[strategy_id]
        strategy.metadata["performance_history"].append({
            "generation": self.current_generation,
            "timestamp": time.time(),
            "performance": performance_data
        })
        
        self.logger.info(f"Recorded performance for strategy {strategy_id}: "
                        f"profit={performance_data.get('profit', 0):.2f}, "
                        f"win_rate={performance_data.get('win_rate', 0):.2f}")
    
    def calculate_fitness(self, performance_data: Dict[str, Any]) -> float:
        """
        Calculate fitness score from performance metrics.
        
        Args:
            performance_data: Dictionary of performance metrics
            
        Returns:
            fitness: Normalized fitness score (0-1)
        """
        # Extract metrics with defaults
        profit = performance_data.get("profit", 0)
        win_rate = performance_data.get("win_rate", 0)
        max_drawdown = performance_data.get("max_drawdown", 0)
        
        # Get weights from config
        weights = self.config["fitness_metrics"]
        profit_weight = weights.get("profit_weight", 0.5)
        win_rate_weight = weights.get("win_rate_weight", 0.3)
        risk_weight = weights.get("risk_weight", 0.2)
        
        # Normalize profit (this is simplistic, should be improved)
        # A better approach is to calculate profit relative to benchmark or population
        norm_profit = min(max(profit / 100, 0), 1)  # Assume 100% is max expected profit
        
        # Win rate is already normalized (0-1)
        norm_win_rate = min(max(win_rate, 0), 1)
        
        # Normalize max drawdown (lower is better)
        # Convert to risk score where 1 = no drawdown
        risk_score = max(1 - (max_drawdown / 100), 0)
        
        # Calculate weighted sum
        fitness = (
            profit_weight * norm_profit +
            win_rate_weight * norm_win_rate +
            risk_weight * risk_score
        )
        
        return fitness
    
    def evolve_next_generation(self):
        """
        Create the next generation of strategies through selection and evolution.
        """
        self.logger.info(f"Starting evolution of generation {self.current_generation + 1}")
        
        # Calculate fitness for all strategies
        for strategy_id, performance in self.strategy_results.items():
            if "fitness_score" not in performance:
                fitness = self.calculate_fitness(performance)
                self.strategy_results[strategy_id]["fitness_score"] = fitness
                
        # Sort strategies by fitness
        sorted_strategies = sorted(
            self.strategy_results.keys(), 
            key=lambda s: self.strategy_results[s].get("fitness_score", 0),
            reverse=True
        )
        
        # Calculate diversity of current population
        current_strategies = [self.strategies[s] for s in sorted_strategies if s in self.strategies]
        diversity = calculate_population_diversity(current_strategies)
        
        # Adjust crossover rate based on diversity
        crossover_rate = adaptive_crossover_rate(
            diversity,
            min_rate=0.1,
            max_rate=0.7
        )
        
        self.logger.info(f"Population diversity: {diversity:.4f}, adaptive crossover rate: {crossover_rate:.4f}")
        
        # Select top performers to survive
        selection_count = int(len(sorted_strategies) * self.config["selection_percentage"])
        selection_count = max(selection_count, 2)  # Ensure at least 2 strategies survive
        
        survivors = sorted_strategies[:selection_count]
        
        self.logger.info(f"Selected {len(survivors)} strategies to survive to next generation")
        
        # Create new population
        new_strategies = {}
        next_generation = self.current_generation + 1
        
        # Keep the survivors
        for strategy_id in survivors:
            if strategy_id not in self.strategies:
                continue
                
            strategy = self.strategies[strategy_id]
            new_strategies[strategy_id] = strategy
        
        # Fill population with new strategies (mutations and crossovers)
        population_size = self.config["population_size"]
        to_create = population_size - len(new_strategies)
        
        # Create offsprings through mutations and crossovers
        for i in range(to_create):
            # Decide whether to do mutation or crossover
            if random.random() < 0.7 and len(survivors) >= 2:  # 70% crossover if possible
                # Crossover (preferring better strategies but with some randomness)
                # Use weighted selection giving preference to better strategies
                weights = [max(0.1, self.strategy_results.get(s, {}).get("fitness_score", 0) * 10) 
                          for s in survivors]
                total_weight = sum(weights)
                
                # Handle edge case where total_weight is 0
                if total_weight <= 0 or len(survivors) < 2:
                    # If we have insufficient survivors or all have zero fitness, use random selection
                    self.logger.warning("Insufficient valid strategies for crossover or all have zero fitness. Using random selection.")
                    available_strategies = list(self.strategies.keys())
                    if len(available_strategies) >= 2:
                        # Random selection without weights
                        parents = random.sample(available_strategies, 2)
                        parent1_id = parents[0]
                        parent2_id = parents[1]
                    else:
                        # Not enough strategies for crossover, fall back to mutation
                        self.logger.warning("Not enough strategies available for crossover, falling back to mutation")
                        # Skip to next iteration which will likely do mutation
                        continue
                else:
                    # Normalize weights to probabilities
                    probs = [w / total_weight for w in weights]
                    
                    # Select two parents
                    parent_indices = random.choices(range(len(survivors)), weights=probs, k=2)
                    parent1_id = survivors[parent_indices[0]]
                    parent2_id = survivors[parent_indices[1]]
                
                parent1 = self.strategies[parent1_id]
                parent2 = self.strategies[parent2_id]
                
                # Create hybrid
                hybrid = parent1.create_hybrid(parent2, crossover_rate)
                hybrid.metadata["generation"] = next_generation
                
                # Add to new population
                new_strategies[hybrid.strategy_id] = hybrid
                
                self.logger.debug(f"Created hybrid strategy {hybrid.strategy_id} from parents "
                                f"{parent1_id} and {parent2_id}")
            else:
                # Mutation (select random parent, weighted by fitness)
                weights = [max(0.1, self.strategy_results.get(s, {}).get("fitness_score", 0) * 10) 
                          for s in survivors]
                
                # Handle case where all strategies have zero fitness
                if not survivors:
                    self.logger.warning("No surviving strategies found for selection. Using random selection.")
                    # If somehow we have no survivors, use any available strategy
                    available_strategies = list(self.strategies.keys())
                    if available_strategies:
                        parent_id = random.choice(available_strategies)
                    else:
                        self.logger.error("No strategies available for evolution")
                        continue
                else:
                    # Normal weighted selection
                    parent_id = random.choices(survivors, weights=weights, k=1)[0]
                parent = self.strategies[parent_id]
                
                # Create mutated clone
                mutation_rate = self.config["mutation_rate"]
                mutant = parent.clone_with_mutation(mutation_rate)
                mutant.metadata["generation"] = next_generation
                
                # Add to new population
                new_strategies[mutant.strategy_id] = mutant
                
                self.logger.debug(f"Created mutated strategy {mutant.strategy_id} from parent {parent_id}")
        
        # Update current generation
        self.current_generation = next_generation
        
        # Replace old population with new
        self.strategies = new_strategies
        self.strategy_results = {}  # Clear results for next generation
        
        # Create stats for this generation
        gen_stats = EvolutionStats(next_generation)
        gen_stats.finalize(self.strategy_results)  # This will be mostly empty until new results are recorded
        self.generation_stats[next_generation] = gen_stats
        
        self.logger.info(f"Evolved generation {next_generation} with {len(self.strategies)} strategies")
        
        # Save this generation's data
        self._save_generation_data(next_generation)
        
        return next_generation
    
    def _save_generation_data(self, generation: int):
        """Save generation data to disk."""
        gen_dir = os.path.join(self.output_dir, f"generation_{generation}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # Save generation stats
        stats_file = os.path.join(gen_dir, "stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.generation_stats.get(generation, EvolutionStats(generation)).to_dict(), f, indent=2)
        
        # Save all strategies
        strategies_dir = os.path.join(gen_dir, "strategies")
        os.makedirs(strategies_dir, exist_ok=True)
        
        for strategy_id, strategy in self.strategies.items():
            strategy_file = os.path.join(strategies_dir, f"{strategy_id}.json")
            strategy.save_to_file(strategy_file)
        
        self.logger.info(f"Saved generation {generation} data to {gen_dir}")
    
    def get_best_strategies(self, generation: int = None, count: int = 10) -> List[str]:
        """
        Get the IDs of the best performing strategies.
        
        Args:
            generation: Filter by generation (None for all generations)
            count: Number of strategies to return
            
        Returns:
            List of strategy IDs sorted by performance
        """
        if not self.strategy_results:
            self.logger.warning("No strategy results available for ranking. Returning random strategies.")
            # If no results available, return random strategies
            available = list(self.strategies.keys())
            if generation is not None:
                # Filter by generation if specified
                available = [s_id for s_id, strategy in self.strategies.items() 
                             if strategy.metadata.get("generation", 0) == generation]
            
            if not available:
                self.logger.warning(f"No strategies available{' for generation ' + str(generation) if generation is not None else ''}.")
                return []
                
            # Return random strategies up to the requested count
            count = min(count, len(available))
            return random.sample(available, count)
            
        # Get strategies for filtering
        strategies_to_consider = list(self.strategy_results.keys())
        
        # Filter by generation if specified
        if generation is not None:
            strategies_to_consider = [s_id for s_id in strategies_to_consider
                                     if s_id in self.strategies and 
                                     self.strategies[s_id].metadata.get("generation", 0) == generation]
        
        # Sort by fitness score
        sorted_strategies = sorted(
            strategies_to_consider, 
            key=lambda s: self.strategy_results[s].get("fitness_score", 0),
            reverse=True
        )
        
        # Handle case where no strategies have fitness scores
        if not sorted_strategies:
            self.logger.warning(f"No strategies with fitness scores found{' for generation ' + str(generation) if generation is not None else ''}. Returning random strategies.")
            available = list(self.strategies.keys())
            if generation is not None:
                available = [s_id for s_id, strategy in self.strategies.items() 
                            if strategy.metadata.get("generation", 0) == generation]
            
            count = min(count, len(available))
            return random.sample(available, count) if available else []
            
        return sorted_strategies[:count]
    
    def get_generation_stats(self, generation: int = None) -> Dict[str, Any]:
        """
        Get statistics for a specific generation.
        
        Args:
            generation: Generation number (None for current)
            
        Returns:
            Dictionary of generation statistics
        """
        if generation is None:
            generation = self.current_generation
            
        if generation in self.generation_stats:
            return self.generation_stats[generation].to_dict()
            
        return {}
    
    def get_evolutionary_history(self) -> Dict[str, Any]:
        """
        Get complete history of the evolutionary process.
        
        Returns:
            Dictionary with evolution statistics and trends
        """
        history = {
            "generations": self.current_generation,
            "total_strategies": len(self.strategies),
            "generation_stats": {gen: stats.to_dict() for gen, stats in self.generation_stats.items()},
            "performance_trends": self._calculate_performance_trends(),
            "diversity_trends": self._calculate_diversity_trends(),
            "current_generation_distribution": self._get_strategy_type_distribution()
        }
        
        return history
    
    def _calculate_performance_trends(self) -> Dict[str, List[float]]:
        """Calculate performance trends across generations."""
        trends = {
            "avg_performance": [],
            "max_performance": [],
            "avg_fitness": []
        }
        
        for gen in range(self.current_generation + 1):
            if gen in self.generation_stats:
                stats = self.generation_stats[gen]
                trends["avg_performance"].append(stats.avg_performance)
                trends["max_performance"].append(stats.max_performance)
                trends["avg_fitness"].append(stats.avg_fitness)
            else:
                # Fill with zeros if generation data is missing
                trends["avg_performance"].append(0)
                trends["max_performance"].append(0)
                trends["avg_fitness"].append(0)
        
        return trends
    
    def _calculate_diversity_trends(self) -> List[float]:
        """Calculate strategy diversity trends across generations."""
        diversity_trend = []
        
        for gen in range(self.current_generation + 1):
            gen_strategies = [s for s in self.strategies.values() 
                             if s.metadata.get("generation", 0) == gen]
            
            if gen_strategies:
                diversity = calculate_population_diversity(gen_strategies)
                diversity_trend.append(diversity)
            else:
                diversity_trend.append(0)
        
        return diversity_trend
    
    def _get_strategy_type_distribution(self) -> Dict[str, int]:
        """Get distribution of strategy types in current generation."""
        distribution = {}
        
        for strategy in self.strategies.values():
            strategy_type = "Unknown"
            if hasattr(strategy, "benbot_strategy") and strategy.benbot_strategy:
                strategy_type = strategy.benbot_strategy.__class__.__name__
            
            if strategy_type not in distribution:
                distribution[strategy_type] = 0
                
            distribution[strategy_type] += 1
        
        return distribution
    
    def export_results(self, output_dir: str = None) -> str:
        """
        Export all evolution results to files.
        
        Args:
            output_dir: Custom output directory (None for timestamp-based)
            
        Returns:
            Path to the output directory
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"export_{timestamp}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overall metadata
        metadata = {
            "generations": self.current_generation,
            "total_strategies": len(self.strategies),
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.start_time if self.start_time else 0
        }
        
        with open(os.path.join(output_dir, "evolution_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save generation stats
        gen_stats_dir = os.path.join(output_dir, "generation_stats")
        os.makedirs(gen_stats_dir, exist_ok=True)
        
        for gen, stats in self.generation_stats.items():
            with open(os.path.join(gen_stats_dir, f"generation_{gen}.json"), 'w') as f:
                json.dump(stats.to_dict(), f, indent=2)
        
        # Save best strategies of each generation
        best_dir = os.path.join(output_dir, "best_strategies")
        os.makedirs(best_dir, exist_ok=True)
        
        for gen in range(self.current_generation + 1):
            if gen in self.generation_stats:
                best_id = self.generation_stats[gen].best_strategy_id
                if best_id and best_id in self.strategies:
                    best_strategy = self.strategies[best_id]
                    best_strategy.save_to_file(os.path.join(best_dir, f"gen_{gen}_best.json"))
        
        # Save evolutionary history
        with open(os.path.join(output_dir, "evolutionary_history.json"), 'w') as f:
            json.dump(self.get_evolutionary_history(), f, indent=2)
            
        self.logger.info(f"Exported evolution results to {output_dir}")
        return output_dir
