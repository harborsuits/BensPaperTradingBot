#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks momentum strategy package.
"""

from .stocks_momentum_strategy import StocksMomentumStrategy

# Create aliases and mock implementations for backward compatibility
StockMomentumStrategy = StocksMomentumStrategy

class RelativeStrengthStrategy(StocksMomentumStrategy):
    """Mock implementation of the RelativeStrengthStrategy for validation purposes."""
    pass

class PriceVolumeStrategy(StocksMomentumStrategy):
    """Mock implementation of the PriceVolumeStrategy for validation purposes."""
    pass

__all__ = [
    "StocksMomentumStrategy",
    "StockMomentumStrategy",
    "RelativeStrengthStrategy",
    "PriceVolumeStrategy"
]
