"""
Macro Guidance Module

This module provides comprehensive macro economic analysis and guidance
for trading decisions, integrating economic events, market regimes, and
volatility conditions into the trading system.
"""

from .macro_engine import MacroGuidanceEngine
from .integration import MacroGuidanceIntegration
from .macro_event_definitions import (
    MacroEvent, EventType, MarketImpact, EventImportance,
    TradingBias, MarketScenario, StrategyAdjustment
)
from .config import DEFAULT_MACRO_CONFIG

__all__ = [
    'MacroGuidanceEngine',
    'MacroGuidanceIntegration',
    'MacroEvent',
    'EventType',
    'MarketImpact',
    'EventImportance',
    'TradingBias',
    'MarketScenario',
    'StrategyAdjustment',
    'DEFAULT_MACRO_CONFIG'
] 