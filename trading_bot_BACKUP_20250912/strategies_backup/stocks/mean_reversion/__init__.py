"""
Stock Mean Reversion Strategies Module

This module provides mean reversion trading strategies for stocks, which aim to
capitalize on price movements that revert to a mean or average value after
deviating significantly.
"""

# Import strategy classes directly from the implementation file
from trading_bot.strategies.stocks.mean_reversion.mean_reversion_strategy import (
    RSIMeanReversionStrategy,
    BollingerBandMeanReversionStrategy,
    StatisticalMeanReversionStrategy,
    MeanReversionStrategy
)

__all__ = [
    'MeanReversionStrategy',
    'RSIMeanReversionStrategy',
    'BollingerBandMeanReversionStrategy',
    'StatisticalMeanReversionStrategy'
] 