"""
Strategy Factory Package

This package provides the infrastructure for strategy registration, discovery,
and instantiation.
"""

from trading_bot.strategies_new.factory.registry import register_strategy, get_registered_strategies
from trading_bot.strategies_new.factory.strategy_factory import StrategyFactory

__all__ = ['register_strategy', 'get_registered_strategies', 'StrategyFactory']
