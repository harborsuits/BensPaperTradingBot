"""
Trading Strategies Module

This module provides a collection of trading strategies for different asset classes 
and market conditions. Strategies are organized by asset type and trading approach.
"""

# Import from asset-specific packages
from trading_bot.strategies.stocks import *
from trading_bot.strategies.options import *
from trading_bot.strategies.crypto import *
from trading_bot.strategies.forex import *

# Import from timeframe-based packages (legacy)
from trading_bot.strategies.timeframe import *

# Import ML and advanced strategies
from trading_bot.strategies.ml_strategy import MLStrategy

# Import base classes and common utilities
from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)

# Import feature engineering directly for easier access
from trading_bot.utils.feature_engineering import FeatureEngineering

# Import base strategy classes
from trading_bot.strategies.base import (
    StockBaseStrategy,
    OptionsBaseStrategy,
    CryptoBaseStrategy,
    ForexBaseStrategy
)

# Import mean reversion strategies directly from source files
from trading_bot.strategies.mean_reversion.zscore_strategy import ZScoreMeanReversionStrategy
from trading_bot.strategies.stocks.mean_reversion.mean_reversion_strategy import (
    MeanReversionStrategy, 
    RSIMeanReversionStrategy, 
    BollingerBandMeanReversionStrategy,
    StatisticalMeanReversionStrategy
)

# Import stock-specific strategies
from trading_bot.strategies.stocks import (
    StockSwingTradingStrategy,
    MultiTimeframeSwingStrategy,
    MomentumStrategy,
    PriceChannelBreakoutStrategy,
    VolumeBreakoutStrategy,
    VolatilityBreakoutStrategy,
    MultiTimeframeCorrelationStrategy
)

__all__ = [
    # Base classes
    'StrategyTemplate',
    'StrategyOptimizable',
    'Signal',
    'SignalType',
    'TimeFrame',
    'MarketRegime',
    
    # Asset-specific base classes
    'StockBaseStrategy',
    'OptionsBaseStrategy',
    'CryptoBaseStrategy',
    'ForexBaseStrategy',
    
    # Advanced strategies
    'MLStrategy',
    
    # Feature engineering
    'FeatureEngineering',
    
    # Mean reversion strategies 
    'ZScoreMeanReversionStrategy',
    'MeanReversionStrategy',
    'RSIMeanReversionStrategy',
    'BollingerBandMeanReversionStrategy',
    'StatisticalMeanReversionStrategy',
    
    # Stock strategies
    'StockSwingTradingStrategy',
    'MultiTimeframeSwingStrategy',
    'MomentumStrategy',
    'PriceChannelBreakoutStrategy',
    'VolumeBreakoutStrategy',
    'VolatilityBreakoutStrategy',
    'MultiTimeframeCorrelationStrategy',
    
    # Import all strategies from subpackages
    # These will be populated from the imports above
] 