#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy module for trend strategies.
"""

from .forex_trend_following_strategy import ForexTrendFollowingStrategy
from .forex_breakout_strategy import ForexBreakoutStrategy
from .forex_day_trading_strategy import ForexDayTradingStrategy
from .forex_carry_trade_strategy import ForexCarryTradeStrategy
from .forex_swing_trading_strategy import ForexSwingTradingStrategy
from .forex_momentum_strategy import ForexMomentumStrategy
from .forex_grid_trading_strategy import ForexGridTradingStrategy
from .forex_retracement_strategy import ForexRetracementStrategy
from .forex_position_trading_strategy import ForexPositionTradingStrategy
from .forex_range_trading_strategy import ForexRangeTradingStrategy
from .forex_counter_trend_strategy import ForexCounterTrendStrategy
from .forex_scalping_strategy import ForexScalpingStrategy

__all__ = [
    "ForexTrendFollowingStrategy",
    "ForexBreakoutStrategy",
    "ForexDayTradingStrategy",
    "ForexCarryTradeStrategy",
    "ForexSwingTradingStrategy",
    "ForexMomentumStrategy",
    "ForexGridTradingStrategy",
    "ForexRetracementStrategy",
    "ForexPositionTradingStrategy",
    "ForexRangeTradingStrategy",
    "ForexCounterTrendStrategy",
    "ForexScalpingStrategy",
]
