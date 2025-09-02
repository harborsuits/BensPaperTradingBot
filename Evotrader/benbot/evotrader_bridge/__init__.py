"""
BensBot-EvoTrader Integration Bridge

This package provides components to integrate EvoTrader's evolutionary trading
system with BensBot's existing strategy framework.
"""

# Add EvoTrader to Python path
import evotrader_path

from benbot.evotrader_bridge.strategy_adapter import BensBotStrategyAdapter
from benbot.evotrader_bridge.evolution_manager import EvolutionManager
from benbot.evotrader_bridge.performance_tracker import PerformanceTracker

__all__ = ['BensBotStrategyAdapter', 'EvolutionManager', 'PerformanceTracker']
