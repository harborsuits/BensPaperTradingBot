"""
Mean Reversion Strategies Module

This module provides mean reversion trading strategies that aim to
capitalize on price movements that revert to a mean or average value after
deviating significantly.
"""

# Import the Z-Score strategy
from trading_bot.strategies.mean_reversion.zscore_strategy import ZScoreMeanReversionStrategy

# Import the stock-specific mean reversion strategies
from trading_bot.strategies.stocks.mean_reversion.mean_reversion_strategy import (
    MeanReversionStrategy, 
    RSIMeanReversionStrategy, 
    BollingerBandMeanReversionStrategy,
    StatisticalMeanReversionStrategy
)

__all__ = [
    'ZScoreMeanReversionStrategy',
    'MeanReversionStrategy',
    'RSIMeanReversionStrategy',
    'BollingerBandMeanReversionStrategy',
    'StatisticalMeanReversionStrategy'
] 