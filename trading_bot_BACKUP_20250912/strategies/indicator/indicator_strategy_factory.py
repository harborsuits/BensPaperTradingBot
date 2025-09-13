#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicator Strategy Factory

This module provides factory methods to create indicator-based strategies
from configuration files or dictionaries.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import importlib
import inspect

from trading_bot.strategies.indicator.indicator_strategy import IndicatorStrategy
from trading_bot.strategies.indicator.indicator_data_provider import IndicatorDataProvider
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.core.constants import SignalDirection

logger = logging.getLogger(__name__)

# Dictionary to register custom strategy classes
STRATEGY_REGISTRY = {}

def register_strategy(cls):
    """
    Decorator to register a strategy class.
    
    Args:
        cls: Strategy class to register
        
    Returns:
        The class itself (unchanged)
    """
    STRATEGY_REGISTRY[cls.__name__] = cls
    return cls


class IndicatorStrategyFactory:
    """
    Factory for creating indicator-based strategies from configuration.
    """
    
    def __init__(self, 
                 broker_manager: Optional[MultiBrokerManager] = None,
                 config_dir: Optional[str] = None):
        """
        Initialize the indicator strategy factory.
        
        Args:
            broker_manager: Optional broker manager for market data access
            config_dir: Directory containing strategy configuration files
        """
        self.broker_manager = broker_manager
        self.config_dir = config_dir or 'config/strategies'
        
        # Initialize data provider
        self.data_provider = IndicatorDataProvider(broker_manager=broker_manager)
        
        # Auto-discover custom strategy implementations
        self._discover_custom_strategies()
        
        logger.info(f"Initialized IndicatorStrategyFactory with {len(STRATEGY_REGISTRY)} registered strategies")
    
    def create_strategy_from_config(self, 
                                   config_file: str, 
                                   symbol: Optional[str] = None) -> Optional[IndicatorStrategy]:
        """
        Create a strategy from a configuration file.
        
        Args:
            config_file: Path to configuration file
            symbol: Optional symbol to override configuration
            
        Returns:
            Strategy instance
        """
        # Load configuration
        try:
            config_path = os.path.join(self.config_dir, config_file) if not os.path.isabs(config_file) else config_file
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading strategy configuration from {config_file}: {str(e)}")
            return None
        
        return self.create_strategy_from_dict(config, symbol)
    
    def create_strategy_from_dict(self, 
                                 config: Dict[str, Any], 
                                 symbol: Optional[str] = None) -> Optional[IndicatorStrategy]:
        """
        Create a strategy from a configuration dictionary.
        
        Args:
            config: Strategy configuration dictionary
            symbol: Optional symbol to override configuration
            
        Returns:
            Strategy instance
        """
        try:
            # Extract basic configuration
            strategy_name = config.get('name', 'unnamed_strategy')
            strategy_type = config.get('strategy_type', 'indicator')
            
            # Set symbol if provided
            if symbol:
                config['symbol'] = symbol
            
            # Validate required fields
            if 'indicators' not in config:
                logger.error(f"Missing required 'indicators' section in configuration for {strategy_name}")
                return None
            
            if 'entry_rules' not in config and 'rules' not in config:
                logger.error(f"Missing required rules section in configuration for {strategy_name}")
                return None
            
            # Some configs might use 'rules' instead of specific entry/exit rules
            # Convert if needed
            if 'rules' in config and ('entry_rules' not in config or 'exit_rules' not in config):
                rules = config.pop('rules')
                
                # Split rules into entry and exit rules
                entry_rules = []
                exit_rules = []
                
                for rule in rules:
                    # Determine rule type based on its content
                    if 'entry_signal' in rule or ('signal' in rule and rule.get('signal') in ['buy', 'sell']):
                        # Convert 'signal' to 'entry_signal' if needed
                        if 'signal' in rule and 'entry_signal' not in rule:
                            signal = rule.pop('signal')
                            rule['entry_signal'] = SignalDirection.BUY if signal == 'buy' else SignalDirection.SELL
                        
                        entry_rules.append(rule)
                    elif 'exit_signal' in rule or rule.get('action') == 'exit':
                        # Convert 'action' to 'exit_signal' if needed
                        if 'action' in rule and rule.get('action') == 'exit' and 'exit_signal' not in rule:
                            rule.pop('action')
                            # Default to opposite of the current position
                            rule['exit_signal'] = None  # Will be determined at runtime
                        
                        exit_rules.append(rule)
                    else:
                        # If no signal is specified, assume it's an entry rule with buy signal
                        rule['entry_signal'] = SignalDirection.BUY
                        entry_rules.append(rule)
                
                # Update config with split rules
                config['entry_rules'] = entry_rules
                config['exit_rules'] = exit_rules
            
            # Special handling for custom strategy types
            if strategy_type != 'indicator' and strategy_type in STRATEGY_REGISTRY:
                # Create custom strategy instance
                strategy_class = STRATEGY_REGISTRY[strategy_type]
                return strategy_class(strategy_name, config)
            
            # Standard indicator strategy
            return IndicatorStrategy(strategy_name, config)
            
        except Exception as e:
            logger.error(f"Error creating strategy from configuration: {str(e)}")
            return None
    
    def load_all_strategies(self) -> Dict[str, IndicatorStrategy]:
        """
        Load all strategy configurations from the config directory.
        
        Returns:
            Dictionary of strategy ID to strategy instance
        """
        strategies = {}
        
        if not os.path.exists(self.config_dir):
            logger.warning(f"Strategy config directory {self.config_dir} does not exist")
            return strategies
        
        # Load all JSON files in the config directory
        for filename in os.listdir(self.config_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                config_file = os.path.join(self.config_dir, filename)
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check if it's a single strategy or a bundle
                if 'strategies' in config:
                    # Bundle of strategies
                    for strategy_config in config['strategies']:
                        strategy = self.create_strategy_from_dict(strategy_config)
                        if strategy:
                            strategy_id = f"{strategy.name}_{strategy.symbol}"
                            strategies[strategy_id] = strategy
                else:
                    # Single strategy
                    strategy = self.create_strategy_from_dict(config)
                    if strategy:
                        strategy_id = f"{strategy.name}_{strategy.symbol}"
                        strategies[strategy_id] = strategy
            
            except Exception as e:
                logger.error(f"Error loading strategy from {filename}: {str(e)}")
        
        logger.info(f"Loaded {len(strategies)} indicator strategies")
        return strategies
    
    def update_strategy_data(self, 
                           strategy: IndicatorStrategy, 
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a strategy with latest market data and indicators.
        
        Args:
            strategy: Strategy instance
            market_data: Market data dictionary
            
        Returns:
            Updated market data with indicators
        """
        try:
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv')
            if ohlcv_data is None or len(ohlcv_data) == 0:
                logger.warning(f"No OHLCV data provided for {strategy.name}")
                return market_data
            
            # Calculate indicators using the data provider
            indicators = self.data_provider.get_multiple_indicators(
                symbol=strategy.symbol,
                indicators=strategy.indicators,
                timeframe=strategy.timeframe
            )
            
            # Add indicators to market data
            market_data['indicators'] = indicators
            
            return market_data
        
        except Exception as e:
            logger.error(f"Error updating strategy data: {str(e)}")
            return market_data
    
    def _discover_custom_strategies(self) -> None:
        """
        Discover and register custom strategy implementations.
        """
        # Look for custom strategies in the same directory
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            for filename in os.listdir(module_dir):
                if filename.endswith('.py') and filename != '__init__.py' and filename != os.path.basename(__file__):
                    try:
                        module_name = f"trading_bot.strategies.indicator.{filename[:-3]}"
                        module = importlib.import_module(module_name)
                        
                        # Find all classes that inherit from IndicatorStrategy
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (obj is not IndicatorStrategy and 
                                issubclass(obj, IndicatorStrategy) and 
                                obj.__module__ == module.__name__):
                                STRATEGY_REGISTRY[name] = obj
                                logger.debug(f"Discovered custom strategy: {name}")
                    except Exception as e:
                        logger.warning(f"Error importing module {filename}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error discovering custom strategies: {str(e)}")


# Create a factory instance with default parameters
def create_factory(broker_manager: Optional[MultiBrokerManager] = None, 
                  config_dir: Optional[str] = None) -> IndicatorStrategyFactory:
    """
    Create a strategy factory instance.
    
    Args:
        broker_manager: Optional broker manager
        config_dir: Optional config directory
        
    Returns:
        IndicatorStrategyFactory instance
    """
    return IndicatorStrategyFactory(broker_manager, config_dir)
