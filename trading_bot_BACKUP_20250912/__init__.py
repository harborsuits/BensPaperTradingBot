#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot Framework

A framework for building, testing, and deploying automated trading strategies.
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Ben Dickinson"

# NOTE:
# Avoid importing heavy submodules at package import time.
# Many submodules pull in optional/system-specific dependencies (e.g. websockets, gym).
# We expose key symbols via lazy imports to prevent cascading import failures
# when users import subpackages like `trading_bot.core.strategy_registry`.
import importlib

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose logging from libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

__all__ = [
    'MainOrchestrator',
    'BenBotAssistant',
    'TradingBot',
    'TradingMode',
    'TradeResult',
    'TradeAction',
    'TradeType',
]


def __getattr__(name):  # PEP 562 lazy attribute access for modules
    if name == 'MainOrchestrator':
        return importlib.import_module('trading_bot.orchestration.main_orchestrator').MainOrchestrator
    if name == 'BenBotAssistant':
        return importlib.import_module('trading_bot.assistant.benbot_assistant').BenBotAssistant
    if name in {'TradingBot', 'TradingMode', 'TradeResult', 'TradeAction', 'TradeType'}:
        core = importlib.import_module('trading_bot.core.trading_bot')
        return getattr(core, name)
    raise AttributeError(f"module 'trading_bot' has no attribute {name!r}")