#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Base Strategy Module

This is a compatibility module to re-export the OptionsBaseStrategy from its actual location.
"""

# Re-export the OptionsBaseStrategy
from trading_bot.strategies.options.base.options_base_strategy import OptionsBaseStrategy

# Export everything from the original module
__all__ = ['OptionsBaseStrategy']
