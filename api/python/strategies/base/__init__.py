#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Strategy Module

This is a compatibility module to provide imports required by the straddle/strangle strategy.
"""

# Import the base strategy implementations from their actual locations
from trading_bot.strategies.options.base.options_base_strategy import OptionsBaseStrategy

__all__ = [
    'OptionsBaseStrategy'
]
