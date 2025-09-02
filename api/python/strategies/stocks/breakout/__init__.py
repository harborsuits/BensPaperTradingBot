#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks breakout strategy package.
"""

from .stocks_breakout_strategy import StocksBreakoutStrategy

# Create mock implementations for validation purposes
class VolatilityBreakoutStrategy(StocksBreakoutStrategy):
    """Mock implementation of the VolatilityBreakoutStrategy for validation purposes."""
    pass

class PriceBreakoutStrategy(StocksBreakoutStrategy):
    """Mock implementation of the PriceBreakoutStrategy for validation purposes."""
    pass

class VolumeBreakoutStrategy(StocksBreakoutStrategy):
    """Mock implementation of the VolumeBreakoutStrategy for validation purposes."""
    pass

__all__ = [
    "StocksBreakoutStrategy",
    "VolatilityBreakoutStrategy",
    "PriceBreakoutStrategy",
    "VolumeBreakoutStrategy"
]
