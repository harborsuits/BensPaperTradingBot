#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Strategy Templates Package

This package provides standard templates for creating trading strategies
that work with the BensBot backtester, strategy finder, and live trading system.

Using these templates ensures that all strategies have consistent interfaces
and can be seamlessly integrated into the trading system.
"""

from .strategy_template import StrategyTemplate, register_strategy_with_registry
from .options_strategy_template import OptionsStrategyTemplate
from .forex_strategy_template import ForexStrategyTemplate
from .stocks_strategy_template import StocksStrategyTemplate

__all__ = [
    'StrategyTemplate',
    'OptionsStrategyTemplate',
    'ForexStrategyTemplate',
    'StocksStrategyTemplate',
    'register_strategy_with_registry',
]
