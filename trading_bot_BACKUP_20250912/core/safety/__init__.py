"""
Safety guardrails for the trading system.

This module provides components for implementing safety features:
- Circuit breakers to limit losses
- Cooldown periods after losses
- Emergency stop functionality
- Trading mode control (live vs paper)
"""

from trading_bot.core.safety.circuit_breakers import CircuitBreakerManager
from trading_bot.core.safety.cooldown import CooldownManager
from trading_bot.core.safety.emergency_stop import EmergencyStopManager
from trading_bot.core.safety.trading_mode import TradingModeManager
from trading_bot.core.safety.safety_manager import SafetyManager

__all__ = [
    'CircuitBreakerManager',
    'CooldownManager', 
    'EmergencyStopManager',
    'TradingModeManager',
    'SafetyManager',
]
