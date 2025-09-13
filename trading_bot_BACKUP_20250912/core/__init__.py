"""
Trading Bot Core Module

Contains base classes, interfaces, and common abstractions.
"""

from trading_bot.core.interfaces import (
    DataProvider,
    IndicatorInterface,
    StrategyInterface,
    SignalInterface,
    RiskManager,
    OrderManager,
    PortfolioManager,
    NotificationManager
)

__all__ = [
    "DataProvider",
    "IndicatorInterface",
    "StrategyInterface",
    "SignalInterface",
    "RiskManager",
    "OrderManager",
    "PortfolioManager",
    "NotificationManager"
]
