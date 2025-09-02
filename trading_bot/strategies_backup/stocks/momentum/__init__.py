"""
Stock Momentum Strategies Module

This module provides momentum-based trading strategies for stocks, which aim to
capture continued price movement by buying assets that have shown strong recent
performance and selling those that have underperformed.
"""

from trading_bot.strategies.stocks.momentum.momentum_strategy import MomentumStrategy

__all__ = ['MomentumStrategy'] 