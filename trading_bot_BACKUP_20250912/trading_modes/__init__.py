#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Modes - Separates signal generation from trading execution logic.

This module provides a clean separation between signal generation (strategies),
trading logic (modes), and risk management, based on best practices from
FreqTrade and OctoBot trading systems.
"""

from trading_bot.trading_modes.base_trading_mode import BaseTradingMode, Order, OrderType
from trading_bot.trading_modes.standard_trading_mode import StandardTradingMode

__all__ = [
    'BaseTradingMode', 'Order', 'OrderType',
    'StandardTradingMode'
]
