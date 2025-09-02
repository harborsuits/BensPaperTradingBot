"""
Stock Breakout Strategies Module

This module provides breakout-based trading strategies for stocks, which aim to
capture significant price movements when an asset breaks through key support or
resistance levels.
"""

from trading_bot.strategies.stocks.breakout.breakout_strategy import PriceChannelBreakoutStrategy, VolumeBreakoutStrategy, VolatilityBreakoutStrategy

__all__ = [
    'PriceChannelBreakoutStrategy',
    'VolumeBreakoutStrategy',
    'VolatilityBreakoutStrategy'
] 