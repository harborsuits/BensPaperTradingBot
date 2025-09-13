#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sector Rotation Trading Strategies Package

This package contains implementations of sector-based rotation strategies:
- Sector Rotation Strategy (economic cycle based)
- Sector Momentum Strategy
- Sector Relative Strength Strategy
- Cross-Sector Opportunities
"""

# Import strategies for easy access
from trading_bot.strategies_new.stocks.sector.sector_rotation_strategy import SectorRotationStrategy

__all__ = ['SectorRotationStrategy']
