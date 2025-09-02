"""
Strategy Templates Module

This module provides template-based trading strategies that can be used as
starting points for creating new custom strategies across various asset classes.
"""

from trading_bot.strategies.templates.breakout_template import BreakoutTemplate
from trading_bot.strategies.templates.trend_following_template import TrendFollowingTemplate
from trading_bot.strategies.templates.mean_reversion_template import MeanReversionTemplate

__all__ = [
    'BreakoutTemplate',
    'TrendFollowingTemplate',
    'MeanReversionTemplate',
] 