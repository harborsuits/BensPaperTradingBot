"""
Stock Strategies Module

This module provides stock-specific trading strategies organized by trading style.
"""

from trading_bot.strategies.stocks.swing import StockSwingTradingStrategy, MultiTimeframeSwingStrategy
from trading_bot.strategies.stocks.momentum import MomentumStrategy
from trading_bot.strategies.stocks.breakout import PriceChannelBreakoutStrategy, VolumeBreakoutStrategy, VolatilityBreakoutStrategy
from trading_bot.strategies.stocks.mean_reversion import MeanReversionStrategy
from trading_bot.strategies.stocks.trend import MultiTimeframeCorrelationStrategy

__all__ = [
    # Swing strategies
    'StockSwingTradingStrategy',
    'MultiTimeframeSwingStrategy',
    
    # Momentum strategies
    'MomentumStrategy',
    
    # Breakout strategies
    'PriceChannelBreakoutStrategy',
    'VolumeBreakoutStrategy',
    'VolatilityBreakoutStrategy',
    
    # Mean reversion strategies
    'MeanReversionStrategy',
    
    # Trend strategies
    'MultiTimeframeCorrelationStrategy',
] 