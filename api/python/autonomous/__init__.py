"""
Autonomous Trading Package

This package provides components for fully autonomous trading strategy
generation, backtesting, approval, and deployment to paper trading.
"""

from trading_bot.autonomous.autonomous_engine import AutonomousEngine
from trading_bot.autonomous.autonomous_ui import AutonomousUI

__all__ = ['AutonomousEngine', 'AutonomousUI']
