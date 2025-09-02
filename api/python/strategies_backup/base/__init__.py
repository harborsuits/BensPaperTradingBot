"""
Base Strategy Classes Module

This module exports the base strategy classes for different asset classes.
These base classes provide asset-specific functionality and defaults.
"""

from trading_bot.strategies.base.stock_base import StockBaseStrategy
from trading_bot.strategies.base.options_base import OptionsBaseStrategy, OptionType
from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.base.crypto_base import CryptoBaseStrategy, ExchangeType

__all__ = [
    'StockBaseStrategy',
    'OptionsBaseStrategy',
    'OptionType',
    'ForexBaseStrategy',
    'ForexSession',
    'CryptoBaseStrategy',
    'ExchangeType'
] 