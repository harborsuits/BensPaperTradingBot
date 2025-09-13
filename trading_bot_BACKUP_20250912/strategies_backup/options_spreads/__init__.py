"""
Options Spreads Strategies Package

This package contains options trading strategies that involve multi-leg spreads
to capitalize on various market forecasts with defined risk/reward profiles.
"""

# Import key strategies for easier access
from trading_bot.strategies.options_spreads.butterfly_spread import ButterflySpreadStrategy
from trading_bot.strategies.options_spreads.iron_condor_strategy import IronCondorStrategy
from trading_bot.strategies.options_spreads.bull_call_spread_strategy import BullCallSpreadStrategy
from trading_bot.strategies.options_spreads.bear_put_spread_strategy import BearPutSpreadStrategy
from trading_bot.strategies.options_spreads.bull_put_spread_strategy import BullPutSpreadStrategy
from trading_bot.strategies.options_spreads.jade_lizard_strategy import JadeLizardStrategy
from trading_bot.strategies.options_spreads.iron_butterfly_strategy import IronButterflyStrategy

# Export the strategies
__all__ = [
    'ButterflySpreadStrategy',
    'IronCondorStrategy',
    'BullCallSpreadStrategy',
    'BearPutSpreadStrategy',
    'BullPutSpreadStrategy',
    'JadeLizardStrategy',
    'IronButterflyStrategy',
] 