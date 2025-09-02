"""
Simplified EvoTrader Bridge

This is a minimal implementation that avoids dependencies on the broken testing framework
while still demonstrating the core functionality.
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import json
import time
import logging
import random
from typing import Dict, List, Any, Optional

# Import core components - avoid importing the broken testing_framework indirectly
from benbot.evotrader_bridge.strategy_adapter import BensBotStrategyAdapter
from benbot.evotrader_bridge.evolution_manager import EvolutionManager
from benbot.evotrader_bridge.performance_tracker import PerformanceTracker

# Import our minimal testing framework
from minimal_testing_framework import SimulationEnvironment

logger = logging.getLogger(__name__)

class SimpleBridge:
    """Simplified bridge between BensBot and EvoTrader."""
    
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize the bridge with configuration.
        
        Args:
            config_path: Path to configuration JSON file
            config: Configuration dictionary (takes precedence over config_path)
        """
        # Load configuration
        self.config = config or {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Merge configs, with passed config taking precedence
                for k, v in file_config.items():
                    if k not in self.config:
                        self.config[k] = v
        
        # Set defaults for missing config values
        self.config.setdefault("output_dir", "bridge_results")
        self.config.setdefault("population_size", 10)
        self.config.setdefault("mutation_rate", 0.2)
        self.config.setdefault("crossover_rate", 0.7)
        
        # Create output directory
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("evotrader_bridge")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.config["output_dir"], "bridge.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        
        # Initialize components
        simulation_dir = os.path.join(self.config["output_dir"], "simulations")
        os.makedirs(simulation_dir, exist_ok=True)
        
        # Initialize performance tracker
        tracker_db = os.path.join(self.config["output_dir"], "performance.db")
        self.performance_tracker = PerformanceTracker(db_path=tracker_db)
        
        # Initialize simulation environment with our minimal implementation
        sim_config = {
            "output_dir": simulation_dir,
            **{k: v for k, v in self.config.items() if k not in ["output_dir"]}
        }
        self.simulation_env = SimulationEnvironment(config=sim_config)
        
        # Initialize evolution manager
        evolution_dir = os.path.join(self.config["output_dir"], "evolution")
        os.makedirs(evolution_dir, exist_ok=True)
        
        evolution_config = {
            "output_dir": evolution_dir,
            "population_size": self.config["population_size"],
            "mutation_rate": self.config["mutation_rate"],
            "crossover_rate": self.config["crossover_rate"],
            **{k: v for k, v in self.config.items() if k not in ["output_dir"]}
        }
        
        self.evolution_manager = EvolutionManager(
            config=evolution_config,
            performance_tracker=self.performance_tracker
        )
        
        self.logger.info("SimpleBridge initialized successfully")
    
    def register_strategy(self, strategy: BensBotStrategyAdapter) -> str:
        """
        Register a strategy with the evolution manager.
        
        Args:
            strategy: Strategy to register
            
        Returns:
            Strategy ID
        """
        self.logger.info(f"Adapting BensBot strategy: {strategy.get_name()}")
        
        # Register with performance tracker
        self.performance_tracker.register_strategy(
            strategy.strategy_id,
            strategy_type=strategy.get_name(),
            generation=0
        )
        
        # Register with evolution manager
        self.evolution_manager.register_strategy(strategy)
        
        self.logger.info(f"Registered strategy {strategy.strategy_id} for evolution")
        
        return strategy.strategy_id
    
    def evaluate_strategy(self, strategy: BensBotStrategyAdapter, test_id: str = None) -> Dict[str, Any]:
        """
        Evaluate a strategy using the simulation environment.
        
        Args:
            strategy: Strategy to evaluate
            test_id: Optional identifier for this test
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Evaluating strategy {strategy.strategy_id}")
        
        # Run simulation using our minimal mock tester
        metrics = self.simulation_env.test_strategy(strategy, test_id)
        
        return metrics
    
    def get_best_strategies(self, generation: int = None, count: int = 10) -> List[BensBotStrategyAdapter]:
        """
        Get the best performing strategies from a generation.
        
        Args:
            generation: Generation number (None for current)
            count: Number of strategies to return
            
        Returns:
            List of best strategies
        """
        # Get best strategy IDs from evolution manager
        best_ids = self.evolution_manager.get_best_strategies(generation, count)
        
        # Get strategy instances
        best_strategies = [
            self.evolution_manager.strategies[strategy_id]
            for strategy_id in best_ids
            if strategy_id in self.evolution_manager.strategies
        ]
        
        return best_strategies
    
    def evolve_strategies(self, generations: int = 1) -> Dict[str, Any]:
        """
        Evolve registered strategies for a number of generations.
        
        Args:
            generations: Number of generations to evolve
            
        Returns:
            Evolution statistics
        """
        self.logger.info(f"Starting evolution process for {generations} generations")
        
        evolution_stats = {
            "initial_generation": self.evolution_manager.current_generation,
            "final_generation": 0,
            "generations_evolved": 0,
            "generation_stats": {}
        }
        
        # Run evolution for specified number of generations
        for i in range(generations):
            # Evolve next generation
            self.evolution_manager.evolve_next_generation()
            
            # Get current generation number
            gen_num = self.evolution_manager.current_generation
            
            # Evaluate all strategies in this generation
            for strategy_id, strategy in self.evolution_manager.strategies.items():
                if strategy.metadata.get("generation", 0) == gen_num:
                    test_id = f"gen_{gen_num}_eval"
                    self.evaluate_strategy(strategy, test_id)
            
            # Get statistics for this generation
            gen_stats = self.evolution_manager.get_generation_stats(gen_num)
            evolution_stats["generation_stats"][gen_num] = gen_stats
            
        evolution_stats["final_generation"] = self.evolution_manager.current_generation
        evolution_stats["generations_evolved"] = generations
        
        # Save evolution results
        self.evolution_manager.export_results()
        
        self.logger.info(f"Completed evolution of {generations} generations")
        
        return evolution_stats
