"""Enhanced ChallengeManager for evolutionary trading bot simulations."""

import logging
import os
import time
import json
import random
import uuid
import copy
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime
from pathlib import Path

from .trading_bot import TradingBot
from .challenge_bot import ChallengeBot
from ..config import ConfigManager
from ..utils.evolution import create_hybrid_bot, calculate_population_diversity, adaptive_crossover_rate
from .strategy import Strategy
from ..utils.logging import get_bot_logger
from ..data.random_walk_provider import RandomWalkDataProvider
from ..data.sequential_data_provider import SequentialDataProvider

class EvolutionStats:
    """Tracks statistics for a single generation of bots."""
    
    def __init__(self, generation: int):
        self.generation = generation
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.bot_count = 0
        self.surviving_bots = 0
        self.extinct_bots = 0
        self.total_balance = 0.0
        self.total_equity = 0.0
        self.avg_balance = 0.0
        self.avg_equity = 0.0
        self.max_balance = 0.0
        self.max_equity = 0.0
        self.min_balance = float('inf')
        self.min_equity = float('inf')
        self.best_bot_id: Optional[str] = None
        self.worst_bot_id: Optional[str] = None
        self.avg_trades_per_bot = 0.0
        self.total_trades = 0
        self.win_rate = 0.0
        self.avg_fitness = 0.0
        self.strategy_distribution: Dict[str, int] = {}  # Strategy type -> count
        
    def finalize(self, bot_results: Dict[str, Dict[str, Any]]):
        """Calculate final statistics based on bot results."""
        self.end_time = time.time()
        self.bot_count = len(bot_results)
        
        if self.bot_count == 0:
            return
            
        # Calculate aggregate statistics
        total_balance = 0.0
        total_equity = 0.0
        total_fitness = 0.0
        total_trades = 0
        total_wins = 0
        total_losses = 0
        max_equity = 0.0
        min_equity = float('inf')
        best_bot_id = None
        worst_bot_id = None
        
        for bot_id, result in bot_results.items():
            # Financial stats
            balance = result.get("final_balance", 0.0)
            equity = result.get("final_equity", 0.0)
            total_balance += balance
            total_equity += equity
            
            # Track best and worst bots
            if equity > max_equity:
                max_equity = equity
                best_bot_id = bot_id
            if equity < min_equity and equity > 0:
                min_equity = equity
                worst_bot_id = bot_id
                
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
        self.total_balance = total_balance
        self.total_equity = total_equity
        self.avg_balance = total_balance / self.bot_count
        self.avg_equity = total_equity / self.bot_count
        self.max_equity = max_equity
        self.min_equity = min_equity
        self.best_bot_id = best_bot_id
        self.worst_bot_id = worst_bot_id
        self.total_trades = total_trades
        self.avg_trades_per_bot = total_trades / self.bot_count if self.bot_count > 0 else 0
        
        # Calculate win rate
        total_closed_trades = total_wins + total_losses
        self.win_rate = total_wins / total_closed_trades if total_closed_trades > 0 else 0
        
        # Average fitness
        self.avg_fitness = total_fitness / self.bot_count
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a dictionary for serialization."""
        return {
            "generation": self.generation,
            "duration_seconds": self.end_time - self.start_time if self.end_time else 0,
            "bot_count": self.bot_count,
            "surviving_bots": self.surviving_bots,
            "extinct_bots": self.extinct_bots,
            "avg_balance": self.avg_balance,
            "avg_equity": self.avg_equity,
            "max_equity": self.max_equity,
            "min_equity": self.min_equity,
            "best_bot_id": self.best_bot_id,
            "worst_bot_id": self.worst_bot_id,
            "total_trades": self.total_trades,
            "avg_trades_per_bot": self.avg_trades_per_bot,
            "win_rate": self.win_rate,
            "avg_fitness": self.avg_fitness,
            "strategy_distribution": self.strategy_distribution
        }


class ChallengeManager:
    """Manages evolutionary trading bot competitions.
    
    Features:
    - Multi-generation bot evolution
    - Natural selection of successful strategies
    - Mutation of strategy parameters
    - Comprehensive performance tracking
    - Configurable selection criteria
    """

    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.bots: Dict[str, ChallengeBot] = {}  # bot_id -> bot
        self.logger = logging.getLogger(__name__)
        
        # Bot results by generation
        self.bot_results: Dict[str, Dict[str, Any]] = {}  # bot_id -> results
        self.generation_stats: Dict[int, EvolutionStats] = {}  # generation -> stats
        self.current_generation = 0
        
        # Runtime state
        self.is_running = False
        self.start_time = None
        self.data_provider = None
        
        # Output directory for results
        self.output_dir = self.config.get("output", {}).get("directory", "results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize generation stats
        self.generation_stats[0] = EvolutionStats(0)

    def register_bot(self, bot: ChallengeBot) -> None:
        """Register a trading bot to the challenge.
        
        Args:
            bot: Bot instance to register
        """
        # Apply position tracking fix to the strategy if needed
        try:
            from ..strategies.position_tracking_fix import fix_moving_average_strategy, fix_rsi_strategy
            
            # Apply appropriate fix based on strategy type
            strategy_name = bot.strategy.__class__.__name__
            if strategy_name == "MovingAverageCrossover":
                fix_moving_average_strategy(bot.strategy)
            elif strategy_name == "RSIStrategy":
                fix_rsi_strategy(bot.strategy)
                
            self.logger.debug(f"Applied position tracking fix to {strategy_name} for bot {bot.bot_id}")
        except Exception as e:
            self.logger.warning(f"Could not apply position tracking fix: {str(e)}")
            
        # Apply enhanced debug logging to track the complete trading process
        if self.config.get("simulation", {}).get("debug_mode", False):
            try:
                from ..core.trading_bot_debug import enhance_bot_logging
                enhance_bot_logging(bot)
                self.logger.debug(f"Applied enhanced debug logging to bot {bot.bot_id}")
            except Exception as e:
                self.logger.warning(f"Could not apply enhanced debug logging: {str(e)}")
        
        # Register the bot
        self.bots[bot.bot_id] = bot
        self.logger.debug(f"Registered bot {bot.bot_id} with strategy {bot.strategy.__class__.__name__}")
        self.generation_stats[self.current_generation].bot_count += 1
        self.logger.debug(f"Registered bot: {bot.bot_id} (gen: {bot.generation})")

    def register_market_data_provider(self, provider):
        """Set the market data provider for the simulation.
        
        Args:
            provider: Object with a get_data(day) method
        """
        self.data_provider = provider
        self.logger.info(f"Registered market data provider: {provider.__class__.__name__}")

    def setup(self):
        # Initialize data provider based on configuration
        data_provider_type = self.config.get("simulation", {}).get("data_provider", "random_walk")
        data_provider_config = self.config.get("data_provider", {})
        
        # Create the appropriate data provider
        if data_provider_type == "sequential":
            # Use a base data provider for generating the data
            base_provider = RandomWalkDataProvider(
                seed=self.config.get("simulation", {}).get("seed", 42),
                initial_prices={symbol: 100.0 for symbol in self.config.get("simulation", {}).get("symbols", ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD"])},
                drift=0.0,
                volatility=0.02
            )
            
            # Wrap with sequential provider to prevent lookahead bias
            self.data_provider = SequentialDataProvider(
                data_source=base_provider,
                lookback_window=data_provider_config.get("lookback_window", 100),
                seed=self.config.get("simulation", {}).get("seed", 42),
                calculate_indicators=data_provider_config.get("calculate_indicators", True)
            )
            self.logger.info("Using SequentialDataProvider to prevent lookahead bias")
        else:  # Default to random walk
            self.data_provider = RandomWalkDataProvider(
                seed=self.config.get("simulation", {}).get("seed", 42),
                initial_prices={symbol: 100.0 for symbol in self.config.get("simulation", {}).get("symbols", ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD"])},
                drift=0.0,
                volatility=0.02
            )
            self.logger.info("Using RandomWalkDataProvider for market simulation")

        initial_balance = self.config.get("simulation", {}).get("initial_balance", 1.0)
        
        for bot_id, bot in self.bots.items():
            try:
                bot.initialize(initial_balance)
                self.logger.debug(f"Initialized bot: {bot_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize bot {bot_id}: {str(e)}")
                # Mark bot as inactive
                bot.is_active = False
                
        self.logger.info(f"Initialized {sum(1 for b in self.bots.values() if b.is_active)} active bots")

    def run(self, max_generations: int = 1):
        """Execute the complete evolutionary simulation.
        
        Args:
            max_generations: Maximum number of generations to simulate
        """
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info(f"Starting evolutionary simulation with {len(self.bots)} bots")
        self.logger.info(f"Target: {max_generations} generations")
        
        try:
            # Initial generation (0)
            self._run_generation()
            
            # Evolve additional generations
            for gen in range(1, max_generations):
                self.current_generation = gen
                self.logger.info(f"Evolving generation {gen}")
                
                # Create next generation through selection and mutation
                self._evolve_next_generation()
                
                # Run the generation
                self._run_generation()
                
                # Save generation stats
                self._save_generation_stats(gen)
                
        except KeyboardInterrupt:
            self.logger.warning("Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
            raise
        finally:
            self.is_running = False
            self.logger.info(f"Simulation completed in {time.time() - self.start_time:.2f} seconds")
            
        return self.generation_stats
    
    def _run_generation(self):
        """Run a single generation of the simulation."""
        duration = self.config.get("simulation", {}).get("duration_days", 30)
        self.logger.info(f"Running generation {self.current_generation} for {duration} days")
        
        # Ensure we have stats tracking for this generation
        if self.current_generation not in self.generation_stats:
            self.generation_stats[self.current_generation] = EvolutionStats(self.current_generation)

        # Setup all bots
        self.setup()

        # Run the simulation
        for day in range(duration):
            # Get market data for the day
            market_data = self._get_market_data(day)

            # Update all active bots
            for bot_id, bot in self.bots.items():
                if not bot.is_active:
                    continue

                    
                try:
                    bot.on_data(market_data)
                except Exception as e:
                    self.logger.error(f"Bot {bot_id} failed on day {day}: {str(e)}")
                    bot.is_active = False
            
            # Log progress periodically
            if day % 5 == 0 or day == duration - 1:
                active_bots = sum(1 for b in self.bots.values() if b.is_active)
                self.logger.info(f"Generation {self.current_generation} - Day {day}/{duration}: {active_bots} active bots")
        
        # Finalize and gather results
        self._finalize_generation()
    
    def _finalize_generation(self):
        """Finalize the current generation and collect results."""
        self.logger.info(f"Finalizing generation {self.current_generation}")
        
        # Collect results from all bots
        for bot_id, bot in self.bots.items():
            try:
                result = bot.finalize()
                self.bot_results[bot_id] = result
            except Exception as e:
                self.logger.error(f"Failed to finalize bot {bot_id}: {str(e)}")
        
        # Calculate generation statistics
        stats = self.generation_stats[self.current_generation]
        stats.finalize(self.bot_results)
        
        # Log generation summary
        self.logger.info(f"Generation {self.current_generation} summary:")
        self.logger.info(f"  Bots: {stats.bot_count}")
        self.logger.info(f"  Avg equity: ${stats.avg_equity:.2f}")
        self.logger.info(f"  Max equity: ${stats.max_equity:.2f}")
        self.logger.info(f"  Win rate: {stats.win_rate:.2%}")
        self.logger.info(f"  Avg fitness: {stats.avg_fitness:.4f}")
    
    def _evolve_next_generation(self):
        """Create the next generation of bots through selection, crossover, and mutation."""
        # Configuration for evolution
        config = self.config.get("evolution", {})
        population_size = config.get("population_size", 100)
        selection_method = config.get("selection_method", "tournament")
        mutation_rate = config.get("mutation_rate", 0.1)
        elite_percent = config.get("elite_percent", 0.1)
        tournament_size = config.get("tournament_size", 5)
        enable_crossover = config.get("enable_crossover", True)  # New: Enable strategy crossover
        crossover_rate = config.get("crossover_rate", 0.3)      # New: Rate of parameter crossover
        crossover_percent = config.get("crossover_percent", 0.3) # New: Percentage of population from crossover
        
        # Get previous generation's results
        prev_gen_bots = {bot_id: bot for bot_id, bot in self.bots.items() 
                       if bot.generation == self.current_generation - 1}
        
        if not prev_gen_bots:
            self.logger.error(f"No bots from generation {self.current_generation - 1} found")
            return
            
        self.logger.info(f"Evolving generation {self.current_generation} from {len(prev_gen_bots)} parent bots")
        
        # Clear current generation bots (but keep previous ones for reference)
        self.bots = {bot_id: bot for bot_id, bot in self.bots.items() 
                   if bot.generation < self.current_generation}
        
        # Create new generation stats
        self.generation_stats[self.current_generation] = EvolutionStats(self.current_generation)
        
        # Sort previous generation by fitness
        sorted_bots = sorted(prev_gen_bots.values(), 
                            key=lambda b: self.bot_results.get(b.bot_id, {}).get("fitness_score", 0), 
                            reverse=True)
        
        # Preserve elite bots (with minimal mutation)
        elite_count = max(1, int(population_size * elite_percent))
        self.logger.info(f"Preserving {elite_count} elite bots")
        
        new_bots = []
        
        # Add elite bots with minimal mutation
        for i in range(min(elite_count, len(sorted_bots))):
            elite_bot = sorted_bots[i]
            new_bot_id = f"elite_{self.current_generation}_{i}"
            
            try:
                # Clone with minimal mutation
                new_bot = elite_bot.clone_with_mutation(new_bot_id, mutation_rate * 0.2)
                new_bots.append(new_bot)
                self.logger.debug(f"Elite bot {elite_bot.bot_id} cloned to {new_bot_id}")
            except Exception as e:
                self.logger.error(f"Failed to clone elite bot {elite_bot.bot_id}: {str(e)}")
        
        # Evaluate population diversity to adapt crossover rate
        strategies = [bot.strategy for bot in sorted_bots]
        diversity = calculate_population_diversity(strategies)
        self.logger.info(f"Population diversity: {diversity:.4f}")
        
        # Adapt crossover rate based on diversity if enabled
        if enable_crossover:
            adaptive_rate = adaptive_crossover_rate(diversity)
            # Use the adaptive rate if it's higher than configured (encourages more crossover when needed)
            crossover_rate = max(crossover_rate, adaptive_rate)
            self.logger.info(f"Using crossover rate: {crossover_rate:.4f}")
        
        # Determine how many bots to create through each method
        crossover_count = int(population_size * crossover_percent) if enable_crossover else 0
        mutation_count = population_size - len(new_bots) - crossover_count
        
        self.logger.info(f"Creating {crossover_count} bots through crossover and {mutation_count} through mutation")
        
        # Create hybrid bots through crossover (if enabled)
        if enable_crossover and crossover_count > 0:
            # Create sufficient hybrids through crossover
            for i in range(crossover_count):
                try:
                    # Select two parents using tournament selection
                    tournament1 = random.sample(sorted_bots, min(tournament_size, len(sorted_bots)))
                    parent1 = max(tournament1, key=lambda b: self.bot_results.get(b.bot_id, {}).get("fitness_score", 0))
                    
                    # Exclude the first parent from second tournament
                    remaining_bots = [b for b in sorted_bots if b.bot_id != parent1.bot_id]
                    if not remaining_bots:  # Fallback if only one bot left
                        remaining_bots = sorted_bots
                        
                    tournament2 = random.sample(remaining_bots, min(tournament_size, len(remaining_bots)))
                    parent2 = max(tournament2, key=lambda b: self.bot_results.get(b.bot_id, {}).get("fitness_score", 0))
                    
                    # Create hybrid bot through crossover
                    new_bot_id = f"hybrid_{self.current_generation}_{i}"
                    new_bot = create_hybrid_bot(
                        parent1, parent2, new_bot_id, 
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate
                    )
                    new_bots.append(new_bot)
                    self.logger.debug(f"Created hybrid bot {new_bot_id} from parents {parent1.bot_id} and {parent2.bot_id}")
                except Exception as e:
                    self.logger.error(f"Failed to create hybrid bot: {str(e)}")
        
        # Create remaining bots through selection and mutation
        remaining_count = population_size - len(new_bots)
        
        if selection_method == "tournament":
            for i in range(remaining_count):
                # Tournament selection
                tournament = random.sample(sorted_bots, min(tournament_size, len(sorted_bots)))
                parent = max(tournament, key=lambda b: self.bot_results.get(b.bot_id, {}).get("fitness_score", 0))
                
                new_bot_id = f"bot_{self.current_generation}_{i}"
                try:
                    new_bot = parent.clone_with_mutation(new_bot_id, mutation_rate)
                    new_bots.append(new_bot)
                except Exception as e:
                    self.logger.error(f"Failed to clone bot {parent.bot_id}: {str(e)}")
        else:  # Default to fitness-proportional selection
            # Calculate selection probabilities based on fitness
            total_fitness = sum(self.bot_results.get(b.bot_id, {}).get("fitness_score", 0.01) for b in sorted_bots)
            if total_fitness <= 0:
                total_fitness = len(sorted_bots)  # Fallback to uniform selection
                
            selection_probs = [max(0.01, self.bot_results.get(b.bot_id, {}).get("fitness_score", 0.01)) / total_fitness 
                             for b in sorted_bots]
            
            for i in range(remaining_count):
                try:
                    # Select parent based on fitness
                    parent = random.choices(sorted_bots, weights=selection_probs, k=1)[0]
                    
                    # Create mutated child
                    new_bot_id = f"bot_{self.current_generation}_{i}"
                    new_bot = parent.clone_with_mutation(new_bot_id, mutation_rate)
                    new_bots.append(new_bot)
                except Exception as e:
                    self.logger.error(f"Failed to create new bot: {str(e)}")
        
        # Register new bots
        for bot in new_bots:
            self.register_bot(bot)
            
        self.logger.info(f"Created {len(new_bots)} new bots for generation {self.current_generation}")
    
    def _save_generation_stats(self, generation: int):
        """Save generation statistics to disk."""
        if generation not in self.generation_stats:
            return
            
        stats = self.generation_stats[generation]
        
        # Create output directory structure
        gen_dir = os.path.join(self.output_dir, f"generation_{generation}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # Save generation stats
        stats_file = os.path.join(gen_dir, "stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        # Save bot results
        results_file = os.path.join(gen_dir, "bot_results.json")
        with open(results_file, 'w') as f:
            # Only save results for this generation
            gen_results = {bot_id: result for bot_id, result in self.bot_results.items() 
                         if result.get("generation") == generation}
            json.dump(gen_results, f, indent=2)
            
        self.logger.info(f"Saved generation {generation} statistics to {gen_dir}")
        
        # Optionally save best bot's strategy
        if stats.best_bot_id and stats.best_bot_id in self.bots:
            best_bot = self.bots[stats.best_bot_id]

    def _get_market_data(self, day: int) -> Dict[str, Any]:
        """Retrieve or generate market data for a given day.
        
        Args:
            day: Simulation day (0-indexed)
            
        Returns:
            Market data snapshot for the day
        """
        if self.data_provider:
            try:
                return self.data_provider.get_data(day)
            except Exception as e:
                self.logger.error(f"Error getting market data for day {day}: {str(e)}")
            
        # Simple random price generator as fallback
        # Only used if no data provider is registered
        self.logger.debug(f"Generating synthetic market data for day {day}")
        
        seed = self.config.get("simulation", {}).get("seed", 42)
        random.seed(seed + day)  # Different seed each day but reproducible
        
        symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD"]
        prices = {
            "BTC/USD": 30000,
            "ETH/USD": 2000,
            "XRP/USD": 0.5,
            "LTC/USD": 100
        }
        
        # Generate random price movements (+/- 5%)
        market_data = {}
        for symbol in symbols:
            base_price = prices[symbol]
            movement = random.uniform(-0.05, 0.05)  # -5% to +5%
            current_price = base_price * (1 + movement)
            
            market_data[symbol] = {
                "price": current_price,
                "high": current_price * (1 + random.uniform(0, 0.02)),
                "low": current_price * (1 - random.uniform(0, 0.02)),
                "volume": random.uniform(1000, 10000),
                "day": day,
                "timestamp": time.time()
            }
            
        return market_data
    
    def get_best_bots(self, generation: int = None, count: int = 10) -> List[str]:
        """Get the IDs of the best performing bots.
        
        Args:
            generation: Generation to query (None for most recent)
            count: Number of bots to return
            
        Returns:
            List of bot IDs sorted by performance
        """
        if generation is None:
            generation = self.current_generation
            
        # Get bots from the specified generation
        gen_bots = {bot_id: bot for bot_id, bot in self.bots.items() 
                  if bot.generation == generation}
                  
        # Sort by fitness
        sorted_bots = sorted(gen_bots.keys(), 
                            key=lambda b: self.bot_results.get(b, {}).get("fitness_score", 0), 
                            reverse=True)
                            
        return sorted_bots[:count]
    
    def get_bot_result(self, bot_id: str) -> Dict[str, Any]:
        """Get the results for a specific bot.
        
        Args:
            bot_id: ID of the bot
            
        Returns:
            Bot results dictionary or empty dict if not found
        """
        return self.bot_results.get(bot_id, {})
    
    def get_generation_summary(self, generation: int = None) -> Dict[str, Any]:
        """Get summary statistics for a generation.
        
        Args:
            generation: Generation to query (None for most recent)
            
        Returns:
            Statistics dictionary
        """
        if generation is None:
            generation = self.current_generation
            
        if generation in self.generation_stats:
            return self.generation_stats[generation].to_dict()
            
        return {}
    
    def export_results(self, output_dir: str = None) -> str:
        """Export all simulation results to files.
        
        Args:
            output_dir: Directory to save results (default: timestamp-based)
            
        Returns:
            Path to the output directory
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"export_{timestamp}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overall simulation metadata
        metadata = {
            "generations": self.current_generation + 1,
            "total_bots": len(self.bots),
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.start_time if self.start_time else 0
        }
        
        with open(os.path.join(output_dir, "simulation_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save generation stats
        gen_stats_dir = os.path.join(output_dir, "generation_stats")
        os.makedirs(gen_stats_dir, exist_ok=True)
        
        for gen, stats in self.generation_stats.items():
            with open(os.path.join(gen_stats_dir, f"generation_{gen}.json"), 'w') as f:
                json.dump(stats.to_dict(), f, indent=2)
        
        # Save bot results
        results_dir = os.path.join(output_dir, "bot_results")
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, "all_results.json"), 'w') as f:
            json.dump(self.bot_results, f, indent=2)
            
        self.logger.info(f"Exported all simulation results to {output_dir}")
        return output_dir
