"""
Strategy Loader

This module provides a loader class for dynamically loading trading strategies
based on configuration parameters.
"""

import importlib
import logging
from typing import Dict, Any, Type, List, Optional

logger = logging.getLogger(__name__)

class StrategyLoader:
    """
    Class responsible for loading trading strategies dynamically
    """
    
    def __init__(self):
        """Initialize the strategy loader"""
        self.loaded_strategies = {}
        logger.info("Strategy loader initialized")
    
    def load_strategy(self, strategy_name: str, **kwargs) -> Any:
        """
        Load a strategy by name
        
        Args:
            strategy_name: Name of the strategy to load
            **kwargs: Additional parameters to pass to the strategy
            
        Returns:
            Strategy instance
        """
        # Check if already loaded
        if strategy_name in self.loaded_strategies:
            return self.loaded_strategies[strategy_name]
            
        # Try to import from strategies module
        try:
            # Try different locations to find the strategy
            locations = [
                f"trading_bot.strategies.{strategy_name}",
                f"trading_bot.strategies.stocks.{strategy_name}",
                f"trading_bot.strategies.stocks.trend.{strategy_name}",
                f"trading_bot.strategies.stocks.momentum.{strategy_name}",
                f"trading_bot.strategies.crypto.{strategy_name}",
                f"trading_bot.strategies.forex.{strategy_name}",
                f"trading_bot.strategies.options.{strategy_name}",
                f"trading_bot.strategies.factory.{strategy_name}",
            ]
            
            strategy_module = None
            for location in locations:
                try:
                    strategy_module = importlib.import_module(location)
                    break
                except ImportError:
                    continue
                    
            if strategy_module is None:
                logger.error(f"Could not find strategy: {strategy_name}")
                return None
                
            # Look for the strategy class in the module
            strategy_class = getattr(strategy_module, strategy_name)
            
            # Instantiate the strategy
            strategy_instance = strategy_class(**kwargs)
            
            # Store in loaded strategies
            self.loaded_strategies[strategy_name] = strategy_instance
            
            logger.info(f"Successfully loaded strategy: {strategy_name}")
            
            return strategy_instance
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading strategy {strategy_name}: {str(e)}")
            return None
    
    def list_available_strategies(self) -> List[str]:
        """
        List available strategies
        
        Returns:
            List of available strategy names
        """
        # This would ideally scan directories to find all strategies
        # For now, just return a fixed list
        from trading_bot.strategies.factory.strategy_registry import StrategyRegistry
        
        try:
            return StrategyRegistry.get_all_strategy_names()
        except Exception as e:
            logger.error(f"Error getting strategy list: {str(e)}")
            return []
    
    def get_strategy_metadata(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get metadata for a strategy
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Dictionary of strategy metadata
        """
        from trading_bot.strategies.factory.strategy_registry import StrategyRegistry
        
        try:
            return StrategyRegistry.get_strategy_metadata(strategy_name) or {}
        except Exception as e:
            logger.error(f"Error getting strategy metadata: {str(e)}")
            return {} 