#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume Surge Trading Strategies Package

This package contains implementations of volume-based trading strategies:
- Volume Surge/Spike Strategy
- Volume Breakout Strategy
- Relative Volume Strategy
- Volume Profile Trading Strategy
"""

# Import strategies for easy access
from trading_bot.strategies_new.stocks.volume.volume_surge_strategy import VolumeSurgeStrategy

__all__ = ['VolumeSurgeStrategy']
