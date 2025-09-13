"""
Options Strangles Strategies Module

This module provides strategies for trading options strangles, which involve
simultaneously buying or selling both a call and a put option with the same
expiration date but different strike prices.
"""

from trading_bot.strategies.options.strangles.strangle_strategy import StrangleStrategy

__all__ = [
    'StrangleStrategy',
] 