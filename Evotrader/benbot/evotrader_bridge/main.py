"""
Main implementation for BensBot-EvoTrader Integration

This module provides the main entry point for using the EvoTrader evolutionary
trading system within BensBot. It orchestrates the strategy adaptation, evolution,
and promotion processes.
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union

from benbot.evotrader_bridge.strategy_adapter import BensBotStrategyAdapter
from benbot.evotrader_bridge.evolution_manager import EvolutionManager
from benbot.evotrader_bridge.performance_tracker import PerformanceTracker
from benbot.evotrader_bridge.testing_framework import SimulationEnvironment
from benbot.evotrader_bridge.ab_testing import ABTest, ABTestBatch


class EvoTraderBridge:
    """Main bridge component connecting BensBot with EvoTrader's evolutionary capabilities."""
    
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize the EvoTrader bridge.
        
        Args:
            config_path: Path to configuration file (None for defaults)
            config: Direct configuration dictionary (overrides config_path if provided)
        """
        # Load configuration
        if config is not None:
            # Use provided config dictionary directly
            default_config = self._load_config(None)  # Get defaults
            self.config = default_config
            # Update defaults with provided config
            self._deep_update(self.config, config)
        else:
            # Load from file path
            self.config = self._load_config(config_path)
        
        # Create output directory
        self.output_dir = self.config.get("output_dir", "evotrader_bridge_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("evotrader_bridge")
        self._setup_logging()
        
        # Initialize components
        self.evolution_manager = EvolutionManager(self.config.get("evolution", {}))
        
        performance_db = os.path.join(self.output_dir, "performance.db")
        self.performance_tracker = PerformanceTracker(db_path=performance_db)
        
        sim_config = self.config.get("simulation", {})
        sim_config["output_dir"] = os.path.join(self.output_dir, "simulations")
        self.simulation_env = SimulationEnvironment(sim_config)
        
        self.logger.info("EvoTraderBridge initialized successfully")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "output_dir": "evotrader_bridge_output",
            "evolution": {
                "output_dir": "strategy_evolution",
                "selection_percentage": 0.3,
                "mutation_rate": 0.2,
                "crossover_rate": 0.3,
                "population_size": 100,
                "max_generations": 20
            },
            "simulation": {
                "data_source": "historical",
                "symbols": ["BTC/USD", "ETH/USD"],
                "timeframe": "1h",
                "start_date": "2022-01-01",
                "end_date": "2022-03-01",
                "initial_balance": 10000,
                "fee_rate": 0.001
            },
            "ab_testing": {
                "test_count": 10,
                "bootstrap_samples": 100,
                "significance_level": 0.05
            },
            "strategy_promotion": {
                "min_improvement": 5.0,  # Minimum percentage improvement required
                "required_significance": True,  # Whether statistical significance is required
                "approval_threshold": 0.7  # Percentage of metrics that must show improvement
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults (keeping custom values)
                    self._deep_update(default_config, loaded_config)
            except Exception as e:
                print(f"Error loading config from {config_path}: {str(e)}")
                print("Using default configuration")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update a dictionary with another dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_file = os.path.join(self.output_dir, "evotrader_bridge.log")
        
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO)
    
    def adapt_benbot_strategy(self, benbot_strategy: Any, strategy_id: str = None) -> BensBotStrategyAdapter:
        """
        Adapt a BensBot strategy to be compatible with the EvoTrader evolution system.
        
        Args:
            benbot_strategy: Original BensBot strategy instance
            strategy_id: Optional ID for the strategy (generated if None)
            
        Returns:
            Adapted strategy instance
        """
        self.logger.info(f"Adapting BensBot strategy: {benbot_strategy.__class__.__name__}")
        
        # Create adapter
        adapter = BensBotStrategyAdapter(
            strategy_id=strategy_id,
            benbot_strategy=benbot_strategy
        )
        
        return adapter
    
    def register_strategy(self, strategy: BensBotStrategyAdapter) -> str:
        """
        Register a strategy with the evolution and performance tracking systems.
        
        Args:
            strategy: Strategy to register
            
        Returns:
            Strategy ID
        """
        # Register with evolution manager
        self.evolution_manager.register_strategy(strategy)
        
        # Register with performance tracker
        self.performance_tracker.register_strategy(
            strategy.strategy_id,
            {
                "strategy_type": strategy.benbot_strategy.__class__.__name__ if strategy.benbot_strategy else "Unknown",
                "generation": strategy.metadata.get("generation", 0),
                "parent_ids": strategy.metadata.get("parent_ids", []),
                "creation_timestamp": time.time()
            }
        )
        
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
        
        # Run simulation
        metrics = self.simulation_env.test_strategy(strategy, test_id)
        
        # Record performance
        self.performance_tracker.record_performance(
            strategy.strategy_id,
            metrics,
            test_id=test_id,
            generation=strategy.metadata.get("generation", 0)
        )
        
        return metrics
    
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
        
        for i in range(generations):
            gen_num = self.evolution_manager.current_generation + 1
            self.logger.info(f"Evolving generation {gen_num}")
            
            # Evolve next generation
            self.evolution_manager.evolve_next_generation()
            
            # Get new generation strategies
            new_strategies = self.evolution_manager.strategies
            
            # Evaluate all strategies in the new generation
            test_id = f"gen_{gen_num}_evaluation"
            for strategy_id, strategy in new_strategies.items():
                try:
                    self.evaluate_strategy(strategy, test_id)
                except Exception as e:
                    self.logger.error(f"Error evaluating strategy {strategy_id}: {str(e)}")
            
            # Get stats for this generation
            gen_stats = self.evolution_manager.get_generation_stats(gen_num)
            evolution_stats["generation_stats"][gen_num] = gen_stats
            
        evolution_stats["final_generation"] = self.evolution_manager.current_generation
        evolution_stats["generations_evolved"] = generations
        
        # Save evolution results
        self.evolution_manager.export_results()
        
        self.logger.info(f"Completed evolution of {generations} generations")
        
        return evolution_stats
    
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
    
    def compare_strategies(self, original: BensBotStrategyAdapter, evolved: BensBotStrategyAdapter) -> Dict[str, Any]:
        """
        Run A/B test comparison between original and evolved strategies.
        
        Args:
            original: Original strategy
            evolved: Evolved strategy variant
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing strategies: {original.strategy_id} vs {evolved.strategy_id}")
        
        # Create AB test with config
        ab_test = ABTest(self.config.get("ab_testing", {}))
        
        # Run comparison
        comparison = ab_test.compare_strategies(original, evolved)
        
        return comparison
    
    def should_promote_strategy(self, comparison: Dict[str, Any]) -> bool:
        """
        Determine if an evolved strategy should be promoted based on comparison results.
        
        Args:
            comparison: Comparison results from A/B test
            
        Returns:
            True if strategy should be promoted, False otherwise
        """
        # Get promotion criteria from config
        promotion_config = self.config.get("strategy_promotion", {})
        min_improvement = promotion_config.get("min_improvement", 5.0)
        required_significance = promotion_config.get("required_significance", True)
        approval_threshold = promotion_config.get("approval_threshold", 0.7)
        
        # Check overall improvement
        is_improvement = comparison["overall"].get("is_improvement", False)
        is_significant = comparison["overall"].get("is_significant_improvement", False)
        improvement_score = comparison["overall"].get("improvement_score", 0)
        
        # Check if improvement meets minimum threshold
        metrics = comparison.get("metrics", {})
        profit_metric = metrics.get("profit", {})
        profit_improvement = profit_metric.get("percentage_diff", 0)
        
        # Determine if strategy should be promoted
        if not is_improvement:
            return False
            
        if required_significance and not is_significant:
            return False
            
        if profit_improvement < min_improvement:
            return False
            
        if improvement_score < approval_threshold:
            return False
            
        return True
    
    def promote_strategy(self, strategy: BensBotStrategyAdapter) -> Dict[str, Any]:
        """
        Promote a strategy to be used in BensBot.
        
        Args:
            strategy: Strategy to promote
            
        Returns:
            Promotion details
        """
        self.logger.info(f"Promoting strategy {strategy.strategy_id} to BensBot")
        
        # Save strategy to promotion directory
        promotion_dir = os.path.join(self.output_dir, "promoted_strategies")
        os.makedirs(promotion_dir, exist_ok=True)
        
        strategy_file = os.path.join(promotion_dir, f"{strategy.strategy_id}.json")
        strategy.save_to_file(strategy_file)
        
        # Record promotion in metadata
        promotion_metadata = {
            "strategy_id": strategy.strategy_id,
            "strategy_type": strategy.benbot_strategy.__class__.__name__ if strategy.benbot_strategy else "Unknown",
            "generation": strategy.metadata.get("generation", 0),
            "parent_ids": strategy.metadata.get("parent_ids", []),
            "promotion_timestamp": time.time(),
            "promotion_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_file = os.path.join(promotion_dir, f"{strategy.strategy_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(promotion_metadata, f, indent=2)
            
        self.logger.info(f"Strategy {strategy.strategy_id} promoted and saved to {strategy_file}")
        
        return promotion_metadata


def main():
    """Command line interface for EvoTrader bridge."""
    parser = argparse.ArgumentParser(description='BensBot-EvoTrader Bridge')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--evolve', type=int, default=0, help='Number of generations to evolve')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate current strategies')
    parser.add_argument('--promote', action='store_true', help='Promote best strategies after evaluation')
    
    args = parser.parse_args()
    
    # Create bridge
    bridge = EvoTraderBridge(args.config)
    
    # Perform requested operations
    if args.evolve > 0:
        bridge.evolve_strategies(args.evolve)
        
    if args.evaluate:
        # Get best strategies from each generation
        for gen in range(bridge.evolution_manager.current_generation + 1):
            best_strategies = bridge.get_best_strategies(generation=gen, count=3)
            
            for strategy in best_strategies:
                bridge.evaluate_strategy(strategy, f"cli_evaluation_gen_{gen}")
                
    if args.promote:
        # Compare best strategies against originals
        best_strategies = bridge.get_best_strategies(count=5)
        original_strategies = bridge.get_best_strategies(generation=0, count=5)
        
        if original_strategies and best_strategies:
            # Pick best original and best evolved
            original = original_strategies[0]
            evolved = best_strategies[0]
            
            # Compare them
            comparison = bridge.compare_strategies(original, evolved)
            
            # Promote if it passes criteria
            if bridge.should_promote_strategy(comparison):
                bridge.promote_strategy(evolved)


if __name__ == "__main__":
    main()
