#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct test script for options strategies
This bypasses the regular import infrastructure
"""

import os
import sys
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Define required enums directly to avoid circular imports
class StrategyType(Enum):
    VOLATILITY = "volatility"
    INCOME = "income"

class AssetClass(Enum):
    OPTIONS = "options"

class MarketRegime(Enum):
    VOLATILE = "volatile"
    RANGE_BOUND = "range_bound"
    EVENT_DRIVEN = "event_driven"

class TimeFrame(Enum):
    SWING = "swing"

# Simple mock strategy registry
class MockStrategyRegistry:
    strategies = {}
    
    @classmethod
    def register(cls, name, strategy_class, metadata=None):
        cls.strategies[name] = (strategy_class, metadata)
        logger.info(f"✓ Registered strategy: {name}")

def main():
    logger.info("----- TESTING DIRECT STRATEGY IMPORTS -----")
    
    # List of strategies to import and test
    strategies_to_test = [
        {
            "name": "StraddleStrangleStrategy",
            "import_path": "trading_bot.strategies.options.volatility_spreads.straddle_strangle_strategy",
            "class_name": "StraddleStrangleStrategy",
            "registry_name": "straddle_strangle",
            "metadata": {
                "type": StrategyType.VOLATILITY,
                "asset_class": AssetClass.OPTIONS,
                "market_regime": MarketRegime.VOLATILE,
                "timeframe": TimeFrame.SWING
            }
        },
        {
            "name": "IronCondorStrategy",
            "import_path": "trading_bot.strategies.options.complex_spreads.iron_condor_strategy_new",
            "class_name": "IronCondorStrategyNew",
            "registry_name": "iron_condor",
            "metadata": {
                "type": StrategyType.INCOME,
                "asset_class": AssetClass.OPTIONS,
                "market_regime": MarketRegime.RANGE_BOUND,
                "timeframe": TimeFrame.SWING
            }
        },
        {
            "name": "StrangleStrategy",
            "import_path": "trading_bot.strategies.options.volatility_spreads.strangle_strategy",
            "class_name": "StrangleStrategy",
            "registry_name": "strangle",
            "metadata": {
                "type": StrategyType.VOLATILITY,
                "asset_class": AssetClass.OPTIONS,
                "market_regime": MarketRegime.VOLATILE,
                "timeframe": TimeFrame.SWING
            }
        },
        {
            "name": "StraddleStrategy",
            "import_path": "trading_bot.strategies.options.volatility_spreads.straddle_strategy",
            "class_name": "StraddleStrategy",
            "registry_name": "straddle",
            "metadata": {
                "type": StrategyType.VOLATILITY,
                "asset_class": AssetClass.OPTIONS,
                "market_regime": MarketRegime.VOLATILE,
                "timeframe": TimeFrame.SWING
            }
        }
    ]
    
    success_count = 0
    
    for strategy_info in strategies_to_test:
        try:
            logger.info(f"Testing import of {strategy_info['name']}...")
            
            # Dynamic import of the strategy module
            module = __import__(strategy_info['import_path'], fromlist=[strategy_info['class_name']])
            
            # Get the strategy class from the module
            strategy_class = getattr(module, strategy_info['class_name'])
            
            # Register the strategy with our mock registry
            MockStrategyRegistry.register(
                strategy_info['registry_name'],
                strategy_class,
                strategy_info['metadata']
            )
            
            # Test instantiation
            strategy_instance = strategy_class()
            logger.info(f"✓ Successfully instantiated {strategy_info['name']}")
            
            # Test methods
            if hasattr(strategy_instance, 'generate_signals'):
                logger.info(f"✓ {strategy_info['name']} has generate_signals method")
            
            success_count += 1
            
        except ImportError as e:
            logger.error(f"✗ Could not import {strategy_info['name']}: {e}")
        except Exception as e:
            logger.error(f"✗ Error with {strategy_info['name']}: {e}")
    
    logger.info("----- TEST COMPLETE -----")
    logger.info(f"Successfully tested {success_count} out of {len(strategies_to_test)} strategies")
    
    return success_count == len(strategies_to_test)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
