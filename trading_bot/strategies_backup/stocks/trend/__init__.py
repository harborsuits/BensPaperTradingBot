"""
Stock Trend Strategies Module

This module provides trend-following trading strategies for stocks, which aim to
capture sustained price movements in established market trends.
"""

from trading_bot.strategies.stocks.trend.multi_timeframe_correlation import MultiTimeframeCorrelationStrategy

__all__ = ['MultiTimeframeCorrelationStrategy'] 