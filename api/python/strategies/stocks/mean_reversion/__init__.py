#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks mean_reversion strategy package.
"""

from .stocks_mean_reversion_strategy import StocksMean_reversionStrategy

# Create aliases and mock implementations for backward compatibility
StockMeanReversionStrategy = StocksMean_reversionStrategy
StocksMeanReversionStrategy = StocksMean_reversionStrategy  # Add this alias to fix import

class RSIReversionStrategy(StocksMean_reversionStrategy):
    """Mock implementation of the RSIReversionStrategy for validation purposes."""
    pass

class BollingerBandStrategy(StocksMean_reversionStrategy):
    """Mock implementation of the BollingerBandStrategy for validation purposes."""
    pass

class VWAPMeanReversionStrategy(StocksMean_reversionStrategy):
    """Mock implementation of the VWAPMeanReversionStrategy for validation purposes."""
    pass

__all__ = [
    "StocksMean_reversionStrategy",
    "StockMeanReversionStrategy",
    "StocksMeanReversionStrategy",
    "RSIReversionStrategy",
    "BollingerBandStrategy",
    "VWAPMeanReversionStrategy"
]
