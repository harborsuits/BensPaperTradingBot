"""
Timeframe Strategies Package

This package contains trading strategies based on various timeframes,
from scalping (minutes) to position trading (months).
"""

# Import key strategies for easier access
from trading_bot.strategies.timeframe.swing_trading import SwingTradingStrategy
from trading_bot.strategies.timeframe.day_trading import DayTradingStrategy
from trading_bot.strategies.timeframe.position_trading import PositionTradingStrategy
from trading_bot.strategies.timeframe.scalping import ScalpingStrategy
from trading_bot.strategies.timeframe.momentum_trading import MomentumTradingStrategy
from trading_bot.strategies.timeframe.range_trading import RangeTradingStrategy
from trading_bot.strategies.timeframe.trend_trading import TrendTradingStrategy
from trading_bot.strategies.timeframe.breakout_trading import BreakoutTradingStrategy
from trading_bot.strategies.timeframe.reversal_trading import ReversalTradingStrategy
from trading_bot.strategies.timeframe.gap_trading import GapTradingStrategy

# Export the strategies
__all__ = [
    'SwingTradingStrategy',
    'DayTradingStrategy',
    'PositionTradingStrategy',
    'ScalpingStrategy',
    'MomentumTradingStrategy',
    'RangeTradingStrategy',
    'TrendTradingStrategy',
    'BreakoutTradingStrategy',
    'ReversalTradingStrategy',
    'GapTradingStrategy',
] 