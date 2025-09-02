#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Strategies Package

This is the main entry point for all trading strategies in BensBot. It handles:
1. Registration and discovery of all strategy implementations
2. Initialization of strategy factory and registry
3. Dynamic strategy creation and instantiation

The package is organized by asset class:
- options: Options trading strategies (spreads, straddles, etc.)
- stocks: Stock trading strategies (trend, momentum, value, etc.)
- forex: Forex trading strategies (trend, range, carry, etc.)
- crypto: Cryptocurrency trading strategies (technical, onchain, etc.)

Each asset class module further organizes strategies by category.
"""

import logging
import importlib
import os
import sys
from typing import Dict, Any, List, Type, Optional

# Define __all__ at the beginning to avoid circular imports
__all__ = [
    'StrategyFactory',
    'StrategyRegistry',
    'AssetClass',
    'StrategyType', 
    'MarketRegime',
    'TimeFrame',
    'options',
    'forex',
    'stocks',
    'get_strategies_by_asset',
    'create_strategy',
    'micro_strategies',  # Added micro_strategies to expose it
    'Strategy',          # Added Strategy for backward compatibility
    'SignalType'         # Added SignalType for backward compatibility
]

# Configure logging
logging = logging.getLogger(__name__)

# Make sure the package is properly defined and has the __path__ attribute
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# Import the strategy factory and registry first
try:
    from .factory.strategy_factory import StrategyFactory
    from .factory.strategy_registry import (
        StrategyRegistry, 
        AssetClass, 
        StrategyType, 
        MarketRegime, 
        TimeFrame
    )
    # These are already defined in __all__ at the top
    
    # Flag to track if registry is properly initialized
    _REGISTRY_INITIALIZED = False
    
except ImportError as e:
    import sys
    import logging
    logging.warning(f"Error importing strategy components: {e}")
    # Create placeholder if import fails
    class StrategyFactory:
        @staticmethod
        def create_strategy(*args, **kwargs):
            return None
    sys.modules[__name__ + '.strategy_factory'] = __import__('types').ModuleType('strategy_factory')
    sys.modules[__name__ + '.strategy_factory'].StrategyFactory = StrategyFactory
    _REGISTRY_INITIALIZED = False

# Import Strategy and SignalType from base/base_strategy.py for backward compatibility
try:
    from .base.base_strategy import Strategy, SignalType
except ImportError as e:
    logging.warning(f"Error importing base strategy: {e}")
    # Create placeholders if import fails
    class Strategy:
        pass
    class SignalType:
        BUY = 1
        SELL = -1
        NEUTRAL = 0

# Now import all asset-specific strategy modules
# These imports will trigger the registration of strategies via their __init__ files
try:
    # Import main asset class packages
    from . import options  # Options strategies
    from . import forex    # Forex strategies 
    from . import stocks   # Stock strategies
    from . import micro_strategies  # Import micro strategies module
    
    # Import factory module for registry initialization
    from . import factory
    
    # Add crypto to __all__ if the module exists
    try:
        from . import crypto  # Crypto strategies
        if 'crypto' not in __all__:
            __all__.append('crypto')
    except ImportError:
        logging.info("Crypto strategies module not found. Will be implemented later.")
    
    # Mark registry as initialized
    _REGISTRY_INITIALIZED = True
    
    # Log successful initialization
    logging.info(f"Successfully initialized strategy registry with {len(StrategyRegistry.list_strategies())} strategies")
    
except Exception as e:
    logging.error(f"Error initializing strategy modules: {e}")

# Helper function to get all available strategies by asset class
def get_strategies_by_asset(asset_class: str) -> List[str]:
    """Return a list of available strategies for a specific asset class."""
    if not _REGISTRY_INITIALIZED:
        return []
    return StrategyRegistry.list_strategies_by_asset(asset_class)

# Helper function to create a strategy instance by name
def create_strategy(strategy_name: str, **kwargs) -> Any:
    """Create a strategy instance by name using the strategy factory."""
    if not _REGISTRY_INITIALIZED:
        return None
    return StrategyFactory.create_strategy(strategy_name, **kwargs)

# Helper functions are already in __all__
