"""
EvoTrader integration module for BensBot

This module provides integration between BensBot and the EvoTrader evolutionary
trading strategy research platform, allowing BensBot to leverage EvoTrader's
capabilities for strategy evolution, optimization, and evaluation.
"""

from trading_bot.research.evotrader.bridge import EvoTraderBridge
from trading_bot.research.evotrader.strategy_adapter import BensBotStrategyAdapter
from trading_bot.research.evotrader.data_connector import EvoTraderDataConnector

__all__ = ['EvoTraderBridge', 'BensBotStrategyAdapter', 'EvoTraderDataConnector']
