#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Factory Bridge Module

This module re-exports the StrategyFactory class from the factory subdirectory
to maintain backward compatibility with code that imports from
trading_bot.strategies.strategy_factory
"""

# Re-export the StrategyFactory class
from trading_bot.strategies.factory.strategy_factory import StrategyFactory

# Re-export any other necessary classes
from trading_bot.strategies.factory.strategy_registry import (
    StrategyRegistry, 
    AssetClass, 
    StrategyType, 
    MarketRegime, 
    TimeFrame
)

# Make sure these are available when importing from this module
__all__ = [
    'StrategyFactory',
    'StrategyRegistry',
    'AssetClass',
    'StrategyType',
    'MarketRegime',
    'TimeFrame'
]
