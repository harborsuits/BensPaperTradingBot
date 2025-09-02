#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Trading Strategies Package

This package contains implementations of various forex trading strategies:

1. Trend Strategies:
   - Trend Following: Standard trend following with moving averages
   - Counter Trend: Fading extreme moves with oscillators
   - Position Trading: Long-term trend following with weekly/monthly charts

2. Momentum Strategies:
   - Momentum: Trading based on price acceleration
   - Breakout: Trading significant price level breakouts

3. Range Strategies:
   - Range Trading: Trading between support and resistance levels
   - Grid Trading: Placing buy and sell orders at preset levels

4. Short-Term Strategies:
   - Scalping: Very short-term trading for small profits
   - Day Trading: Intraday trading with no overnight positions

5. Swing Strategies:
   - Swing Trading: Trading market swings over several days
   - Retracement: Trading pullbacks in trending markets

6. Specialized Strategies:
   - Carry Trade: Profiting from interest rate differentials
   - London Session Breakout: Trading the London session open
"""

# Import strategy modules
from . import momentum
from . import trend
from . import swing
from . import scalping
from . import breakout
from . import range

# Import specific strategy classes
try:
    # Trend strategies
    from .trend import ForexTrendFollowingStrategy
    from .trend import ForexCounterTrendStrategy
    from .trend import ForexPositionTradingStrategy
    
    # Momentum strategies
    from .momentum import ForexMomentumStrategy
    from .breakout import ForexBreakoutStrategy
    from .breakout import LondonSessionBreakoutStrategy
    
    # Range strategies
    from .range import ForexRangeTradingStrategy
    from .trend import ForexGridTradingStrategy
    
    # Short-term strategies
    from .scalping import ForexScalpingStrategy
    from .trend import ForexDayTradingStrategy
    
    # Swing strategies
    from .swing import ForexSwingTradingStrategy
    from .trend import ForexRetracementStrategy
    
    # Specialized strategies
    from .trend import ForexCarryTradeStrategy
except ImportError as e:
    import logging
    logging.warning(f"Some forex strategies could not be imported: {e}")

# Create a clean, deduplicated list of all strategy classes
__all__ = [
    # Trend strategies
    "ForexTrendFollowingStrategy",
    "ForexCounterTrendStrategy",
    "ForexPositionTradingStrategy",
    
    # Momentum strategies
    "ForexMomentumStrategy",
    "ForexBreakoutStrategy",
    "LondonSessionBreakoutStrategy",
    
    # Range strategies
    "ForexRangeTradingStrategy",
    "ForexGridTradingStrategy",
    
    # Short-term strategies
    "ForexScalpingStrategy",
    "ForexDayTradingStrategy",
    
    # Swing strategies
    "ForexSwingTradingStrategy",
    "ForexRetracementStrategy",
    
    # Specialized strategies
    "ForexCarryTradeStrategy",
]

# Register all strategies with the strategy registry
from trading_bot.core.strategy_registry import StrategyRegistry

try:
    # Register trend strategies
    StrategyRegistry.register("forex_trend_following", ForexTrendFollowingStrategy)
    StrategyRegistry.register("forex_counter_trend", ForexCounterTrendStrategy)
    StrategyRegistry.register("forex_position_trading", ForexPositionTradingStrategy)
    
    # Register momentum strategies
    StrategyRegistry.register("forex_momentum", ForexMomentumStrategy)
    StrategyRegistry.register("forex_breakout", ForexBreakoutStrategy)
    StrategyRegistry.register("forex_london_breakout", LondonSessionBreakoutStrategy)
    
    # Register range strategies
    StrategyRegistry.register("forex_range_trading", ForexRangeTradingStrategy)
    StrategyRegistry.register("forex_grid_trading", ForexGridTradingStrategy)
    
    # Register short-term strategies
    StrategyRegistry.register("forex_scalping", ForexScalpingStrategy)
    StrategyRegistry.register("forex_day_trading", ForexDayTradingStrategy)
    
    # Register swing strategies
    StrategyRegistry.register("forex_swing_trading", ForexSwingTradingStrategy)
    StrategyRegistry.register("forex_retracement", ForexRetracementStrategy)
    
    # Register specialized strategies
    StrategyRegistry.register("forex_carry_trade", ForexCarryTradeStrategy)
    
    logging.info("Successfully registered all forex strategies")
except (NameError, ImportError) as e:
    import logging
    logging.warning(f"Not all forex strategies could be registered: {e}")
