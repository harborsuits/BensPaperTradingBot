#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap Trading Strategies Package

This package contains implementations of gap trading strategies:
- Gap and Go (continuation) strategy
- Gap Fade (reversal) strategy
"""

# Import strategies for easy access
from trading_bot.strategies_new.stocks.gap.gap_trading_strategy import GapTradingStrategy

__all__ = ['GapTradingStrategy']
