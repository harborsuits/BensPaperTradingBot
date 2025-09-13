"""
BensBot-EvoTrader Bridge Module

This module connects BensBot with the EvoTrader evolutionary strategy research platform,
leveraging the existing evotrader_bridge components from the EvoTrader repository.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger("benbot.research.evotrader_integration")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EvoTraderIntegration:
    """Main integration class for connecting BensBot with EvoTrader."""
    
    def __init__(self, evotrader_repo_path: Optional[str] = None):
        """
        Initialize the EvoTrader integration.
        
        Args:
            evotrader_repo_path: Path to the EvoTrader repository
                (default: looks for /Evotrader at the project root)
        """
        # Find EvoTrader repository
        self.repo_path = self._find_evotrader_repo(evotrader_repo_path)
        
        if not self.repo_path:
            logger.error("EvoTrader repository not found. Integration unavailable.")
            self.available = False
            return

        # Add EvoTrader to Python path
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)
            logger.info(f"Added EvoTrader repo to Python path: {self.repo_path}")

        # Import path helper to ensure proper imports within EvoTrader
        try:
            import evotrader_path
            logger.debug("Imported evotrader_path helper")
        except ImportError:
            logger.warning("Could not import evotrader_path helper. Some imports may fail.")

        # Import bridge components
        try:
            # Import main bridge from EvoTrader repo
            from benbot.evotrader_bridge.main import EvoTraderBridge
            from benbot.evotrader_bridge.strategy_adapter import BensBotStrategyAdapter
            from benbot.evotrader_bridge.evolution_manager import EvolutionManager
            from benbot.evotrader_bridge.performance_tracker import PerformanceTracker
            from benbot.evotrader_bridge.testing_framework import SimulationEnvironment

            # Store bridge components for later use
            self.bridge_cls = EvoTraderBridge
            self.adapter_cls = BensBotStrategyAdapter
            self.evolution_manager_cls = EvolutionManager
            self.performance_tracker_cls = PerformanceTracker
            self.simulation_env_cls = SimulationEnvironment
            
            # Success flag
            self.available = True
            logger.info("Successfully imported EvoTrader bridge components")
            
        except ImportError as e:
            logger.error(f"Failed to import EvoTrader bridge components: {e}")
            self.available = False
    
    def _find_evotrader_repo(self, provided_path: Optional[str] = None) -> Optional[str]:
        """
        Find the EvoTrader repository path.
        
        Args:
            provided_path: User-provided path to the EvoTrader repo
            
        Returns:
            Absolute path to the EvoTrader repository or None if not found
        """
        # Use provided path if given
        if provided_path and os.path.exists(provided_path):
            return os.path.abspath(provided_path)
        
        # Look for Evotrader in project root
        project_root = Path(__file__).parent.parent.parent.parent  # Up to Trading:BenBot
        default_path = os.path.join(project_root, "Evotrader")
        
        if os.path.exists(default_path):
            return default_path
        
        # Not found
        return None
    
    def create_bridge(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create an instance of the EvoTraderBridge from the EvoTrader repo.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            EvoTraderBridge instance or None if unavailable
        """
        if not self.available:
            logger.error("EvoTrader integration unavailable. Bridge cannot be created.")
            return None
        
        try:
            # Create the bridge instance
            bridge = self.bridge_cls(config_path=None, config=config)
            return bridge
        except Exception as e:
            logger.error(f"Error creating EvoTraderBridge: {e}")
            return None

    def adapt_strategy(self, strategy: Any, strategy_id: Optional[str] = None) -> Any:
        """
        Adapt a BensBot strategy for use with EvoTrader.
        
        Args:
            strategy: BensBot strategy to adapt
            strategy_id: Optional unique identifier
            
        Returns:
            Adapted strategy or None if failed
        """
        if not self.available:
            logger.error("EvoTrader integration unavailable. Cannot adapt strategy.")
            return None
        
        try:
            # Create adapter
            adapter = self.adapter_cls(
                strategy_id=strategy_id,
                benbot_strategy=strategy
            )
            return adapter
        except Exception as e:
            logger.error(f"Error adapting strategy: {e}")
            return None

    def run_evolution(
        self, 
        asset_class: str = "forex", 
        generations: int = 10,
        population_size: int = 50,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        strategies: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Run strategy evolution using EvoTrader.
        
        Args:
            asset_class: "forex" or "crypto"
            generations: Number of generations to evolve
            population_size: Size of strategy population
            symbols: Trading symbols to use (pairs)
            timeframes: Timeframes to test on
            strategies: Initial strategies (optional)
            
        Returns:
            Evolution results dictionary or empty dict if failed
        """
        if not self.available:
            logger.error("EvoTrader integration unavailable. Cannot run evolution.")
            return {}
        
        # Create bridge
        bridge = self.create_bridge()
        if not bridge:
            return {}
        
        try:
            # Configure evolution
            if asset_class == "forex":
                default_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
                default_timeframes = ["1h", "4h", "1d"]
            else:  # crypto
                default_symbols = ["BTC/USD", "ETH/USD"]
                default_timeframes = ["15m", "1h", "4h", "1d"]
            
            # Use provided values or defaults
            symbols = symbols or default_symbols
            timeframes = timeframes or default_timeframes
            
            # Run evolution through the bridge
            results = bridge.evolve_strategies(
                strategies=strategies,
                asset_class=asset_class,
                symbols=symbols,
                timeframes=timeframes,
                generations=generations,
                population_size=population_size
            )
            
            return results
        except Exception as e:
            logger.error(f"Error running evolution: {e}")
            return {}
    
    def get_best_strategies(self, results: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
        """
        Extract the best strategies from evolution results.
        
        Args:
            results: Evolution results dictionary
            count: Number of top strategies to return
            
        Returns:
            List of top strategy dictionaries
        """
        if not results or "generations" not in results:
            return []
        
        # Get last generation
        generations = results.get("generations", [])
        if not generations:
            return []
        
        last_gen = generations[-1]
        
        # Extract strategies
        strategies = last_gen.get("strategies", [])
        
        # Sort by fitness
        sorted_strategies = sorted(
            strategies,
            key=lambda s: s.get("fitness", 0),
            reverse=True
        )
        
        # Return top N
        return sorted_strategies[:count]
    
    def deploy_strategy(self, strategy: Any, target_env: str = "paper") -> bool:
        """
        Deploy a strategy from EvoTrader to BensBot for live trading.
        
        Args:
            strategy: Strategy object or ID
            target_env: "paper" or "live"
            
        Returns:
            Success flag
        """
        if not self.available:
            logger.error("EvoTrader integration unavailable. Cannot deploy strategy.")
            return False
        
        # Create bridge
        bridge = self.create_bridge()
        if not bridge:
            return False
        
        try:
            # Deploy the strategy
            success = bridge.deploy_strategy_to_benbot(
                strategy=strategy,
                target_env=target_env
            )
            
            return success
        except Exception as e:
            logger.error(f"Error deploying strategy: {e}")
            return False
