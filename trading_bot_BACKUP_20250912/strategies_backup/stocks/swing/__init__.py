"""
Stock Swing Trading Strategies Module

This module provides swing trading strategies for stocks, which aim to
capture medium-term price movements over periods of days to weeks.
"""

from trading_bot.strategies.stocks.swing.swing_trading_strategy import StockSwingTradingStrategy
from trading_bot.strategies.stocks.swing.multi_timeframe_swing import MultiTimeframeSwingStrategy

__all__ = [
    'StockSwingTradingStrategy',
    'MultiTimeframeSwingStrategy'
] 