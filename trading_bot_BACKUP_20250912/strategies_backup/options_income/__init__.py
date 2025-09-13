"""
Options Income Strategies Package

This package contains options trading strategies focused on generating income
through premium collection and risk-defined options positions.
"""

# Import key strategies for easier access
from trading_bot.strategies.options_income.cash_secured_put_strategy import CashSecuredPutStrategy
from trading_bot.strategies.options_income.collar_strategy import CollarStrategy
from trading_bot.strategies.options_income.married_put_strategy import MarriedPutStrategy

# Export the strategies
__all__ = [
    'CashSecuredPutStrategy',
    'CollarStrategy',
    'MarriedPutStrategy',
] 