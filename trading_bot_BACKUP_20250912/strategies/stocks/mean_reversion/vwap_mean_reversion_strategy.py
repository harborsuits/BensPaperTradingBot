"""
VWAP Mean Reversion Strategy
"""

from .stocks_mean_reversion_strategy import StocksMean_reversionStrategy

class VWAPMeanReversionStrategy(StocksMean_reversionStrategy):
    """Mock implementation of the VWAPMeanReversionStrategy for validation purposes."""

    def __init__(self):
        super().__init__()
        self.name = "VWAP Mean Reversion Strategy"
