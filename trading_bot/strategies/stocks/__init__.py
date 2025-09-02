#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks Trading Strategies Package

This package contains implementations of various stock trading strategies:

1. Base Strategy:
   - StocksBaseStrategy: Base class for all stock strategies with common functionality

2. Trend-Following Strategies:
   - StocksTrendStrategy: Follows established trends using moving averages and other indicators
   - StocksTrendFollowingStrategy: Specific implementation focused on trend following

3. Momentum Strategies:
   - StocksMomentumStrategy: Captures price momentum across various timeframes
   - RelativeStrengthStrategy: Trades based on relative strength vs market/sector

4. Value & Growth Strategies:
   - StocksValueStrategy: Trades undervalued stocks based on fundamental metrics
   - StocksGrowthStrategy: Focuses on stocks with high growth potential 

5. Mean Reversion Strategies:
   - StocksMeanReversionStrategy: Trades mean reversion patterns
   - VWAPMeanReversionStrategy: Uses VWAP for mean reversion signals

6. Event & Breakout Strategies:
   - StocksBreakoutStrategy: Trades price breakouts from patterns
   - GapTradingStrategy: Trades opening price gaps
   - NewsSentimentStrategy: Trades based on news sentiment analysis

7. Technical & Statistical Strategies:
   - StocksTechnicalStrategy: Uses technical indicators for trading signals
   - StocksStatisticalStrategy: Applies statistical models for trading

8. Specialized Strategies:
   - StocksDividendStrategy: Focuses on dividend-paying stocks
   - StocksFundamentalStrategy: Uses fundamental data for trading decisions
   - ShortSellingStrategy: Specifically designed for short positions
   - VolumeSurgeStrategy: Identifies and trades on unusual volume
"""

# Import subpackages
from . import base
from . import trend
from . import momentum
from . import value
from . import growth
from . import dividend
from . import mean_reversion
from . import breakout
from . import technical
from . import fundamental
from . import sentiment
from . import statistical

# Import all strategies with proper error handling
try:
    # Base strategy
    from .base import StocksBaseStrategy
    
    # Trend & momentum strategies
    from .trend import StocksTrendStrategy, StocksTrendFollowingStrategy
    from .momentum import StocksMomentumStrategy
    
    # Value & growth strategies
    from .value import StocksValueStrategy
    from .growth import StocksGrowthStrategy
    
    # Mean reversion strategies
    from .mean_reversion import StocksMeanReversionStrategy
    from . import VWAPMeanReversionStrategy
    
    # Event & breakout strategies
    from .breakout import StocksBreakoutStrategy
    from . import GapTradingStrategy
    from .sentiment import NewsSentimentStrategy
    
    # Technical & statistical strategies
    from .technical import StocksTechnicalStrategy
    from .statistical import StocksStatisticalStrategy
    
    # Specialized strategies
    from .dividend import StocksDividendStrategy
    from .fundamental import StocksFundamentalStrategy
    
    # Correct naming conventions for strategies with underscores
    StocksMeanReversionStrategy = getattr(mean_reversion, "StocksMean_reversionStrategy", StocksMeanReversionStrategy)
    
except ImportError as e:
    import logging
    logging.warning(f"Some stock strategies could not be imported: {e}")

# Create a comprehensive list of all strategies
__all__ = [
    # Base strategy
    "StocksBaseStrategy",
    
    # Trend & momentum strategies
    "StocksTrendStrategy",
    "StocksTrendFollowingStrategy",
    "StocksMomentumStrategy",
    
    # Value & growth strategies
    "StocksValueStrategy",
    "StocksGrowthStrategy",
    
    # Mean reversion strategies
    "StocksMeanReversionStrategy",
    "VWAPMeanReversionStrategy",
    
    # Event & breakout strategies
    "StocksBreakoutStrategy",
    "GapTradingStrategy",
    "NewsSentimentStrategy",
    
    # Technical & statistical strategies
    "StocksTechnicalStrategy",
    "StocksStatisticalStrategy",
    
    # Specialized strategies
    "StocksDividendStrategy",
    "StocksFundamentalStrategy",
]

# Register all strategies with the strategy registry
from trading_bot.core.strategy_registry import StrategyRegistry

try:
    # Register base strategy
    StrategyRegistry.register("stocks_base", StocksBaseStrategy)
    
    # Register trend & momentum strategies
    StrategyRegistry.register("stocks_trend", StocksTrendStrategy)
    StrategyRegistry.register("stocks_trend_following", StocksTrendFollowingStrategy)
    StrategyRegistry.register("stocks_momentum", StocksMomentumStrategy)
    
    # Register value & growth strategies
    StrategyRegistry.register("stocks_value", StocksValueStrategy)
    StrategyRegistry.register("stocks_growth", StocksGrowthStrategy)
    
    # Register mean reversion strategies
    StrategyRegistry.register("stocks_mean_reversion", StocksMeanReversionStrategy)
    StrategyRegistry.register("stocks_vwap_mean_reversion", VWAPMeanReversionStrategy)
    
    # Register event & breakout strategies
    StrategyRegistry.register("stocks_breakout", StocksBreakoutStrategy)
    StrategyRegistry.register("stocks_gap_trading", GapTradingStrategy)
    StrategyRegistry.register("stocks_news_sentiment", NewsSentimentStrategy)
    
    # Register technical & statistical strategies
    StrategyRegistry.register("stocks_technical", StocksTechnicalStrategy)
    StrategyRegistry.register("stocks_statistical", StocksStatisticalStrategy)
    
    # Register specialized strategies
    StrategyRegistry.register("stocks_dividend", StocksDividendStrategy)
    StrategyRegistry.register("stocks_fundamental", StocksFundamentalStrategy)
    
    logging.info("Successfully registered all stocks strategies")
except (NameError, ImportError) as e:
    import logging
    logging.warning(f"Not all stocks strategies could be registered: {e}")
from .statistical import StocksStatisticalStrategy

__all__ = [
    "StocksBaseStrategy",
        "StocksTrendStrategy",
    "StocksMomentumStrategy",
    "StocksValueStrategy",
    "StocksGrowthStrategy",
    "StocksDividendStrategy",
    "StocksMean_reversionStrategy",
    "StocksBreakoutStrategy",
    "StocksTechnicalStrategy",
    "StocksFundamentalStrategy",
    "StocksSentimentStrategy",
    "StocksStatisticalStrategy",
]
