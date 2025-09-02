#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks trend strategy package.
"""

from .stocks_trend_strategy import StocksTrendStrategy

# Create mock implementations for validation purposes
class MovingAverageCrossStrategy(StocksTrendStrategy):
    """Mock implementation of the MovingAverageCrossStrategy for validation purposes."""
    pass

class StocksTrendFollowingStrategy(StocksTrendStrategy):
    """Mock implementation of the StocksTrendFollowingStrategy for validation purposes."""
    pass

class MAACDStrategy(StocksTrendStrategy):
    """Mock implementation of the MAACDStrategy for validation purposes."""
    pass

class TrendChannelStrategy(StocksTrendStrategy):
    """Mock implementation of the TrendChannelStrategy for validation purposes."""
    pass

__all__ = [
    "StocksTrendStrategy",
    "MovingAverageCrossStrategy",
    "StocksTrendFollowingStrategy",
    "MAACDStrategy",
    "TrendChannelStrategy"
]
